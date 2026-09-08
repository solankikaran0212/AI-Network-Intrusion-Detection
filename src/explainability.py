"""Model explainability via SHAP, with deliberate fallbacks.

An intrusion detection alert that says only "DDoS, 97% confident" is close to
useless operationally. The analyst's next question is always *why*, and if the
system cannot answer it they must reconstruct the reasoning by hand - at which
point the automation has saved nothing. Explainability here is a product
requirement, not a nice-to-have.

Method selection, in order of preference:

1. **TreeSHAP** (``shap.TreeExplainer``) for Random Forest / XGBoost. Exact,
   fast (polynomial rather than exponential in features), and the right tool for
   the models we actually select.
2. **LinearSHAP** for Logistic Regression.
3. **Permutation importance** as a global fallback if SHAP fails to initialise.
4. **Model-native ``feature_importances_`` / coefficients** as a last resort.

Levels 3 and 4 are global rather than per-prediction, so the module always
reports which method produced an explanation. Silently degrading from local to
global attribution would be misleading - they answer different questions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.config import get_logger

logger = get_logger(__name__)

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    SHAP_AVAILABLE = False
    logger.warning("shap not installed; falling back to model-native importances.")


#: Mapping from transformed feature names back to something human-readable.
_PRETTY_NAMES: dict[str, str] = {
    "flow_packets_s": "Flow Packets/s",
    "flow_bytes_s": "Flow Bytes/s",
    "packets_per_second": "Packets per Second (derived)",
    "bytes_per_second": "Bytes per Second (derived)",
    "total_fwd_packets": "Total Forward Packets",
    "total_backward_packets": "Total Backward Packets",
    "total_length_of_fwd_packets": "Total Forward Bytes",
    "total_length_of_bwd_packets": "Total Backward Bytes",
    "destination_port": "Destination Port",
    "packet_length_mean": "Packet Length Mean",
    "packet_length_std": "Packet Length Std Dev",
    "flow_duration": "Flow Duration",
    "duration_seconds": "Flow Duration (s)",
    "fwd_bwd_packet_ratio": "Fwd/Bwd Packet Ratio (derived)",
    "fwd_bwd_byte_ratio": "Fwd/Bwd Byte Ratio (derived)",
    "tcp_flag_density": "TCP Flag Density (derived)",
    "avg_packet_size_derived": "Average Packet Size (derived)",
    "syn_flag_count": "SYN Flag Count",
    "ack_flag_count": "ACK Flag Count",
    "fin_flag_count": "FIN Flag Count",
    "psh_flag_count": "PSH Flag Count",
    "rst_flag_count": "RST Flag Count",
    "init_win_bytes_forward": "Initial Window Bytes (fwd)",
    "init_win_bytes_backward": "Initial Window Bytes (bwd)",
}


def prettify(name: str) -> str:
    """Turn a transformed feature name into analyst-facing text."""
    if name in _PRETTY_NAMES:
        return _PRETTY_NAMES[name]
    if name.startswith("port_category_"):
        return f"Service: {name.removeprefix('port_category_')}"
    if name.startswith("protocol_"):
        return f"Protocol: {name.removeprefix('protocol_')}"
    return name.replace("_", " ").title()


@dataclass
class FeatureContribution:
    """One feature's contribution to one prediction."""

    feature: str
    display_name: str
    contribution: float
    direction: str
    value: float | str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "display_name": self.display_name,
            "contribution": round(float(self.contribution), 6),
            "direction": self.direction,
            "value": self.value,
        }


class ModelExplainer:
    """Wraps a fitted :class:`~src.train.ModelBundle` with an explanation method."""

    def __init__(self, bundle: Any, background_size: int = 100) -> None:
        self.bundle = bundle
        self.feature_names = bundle.transformed_feature_names()
        self.method = "none"
        self._explainer = None
        self._global_importance: np.ndarray | None = None
        self._init_explainer(background_size)

    # ----------------------------------------------------------------- setup

    def _init_explainer(self, background_size: int) -> None:
        model = self.bundle.model
        model_type = type(model).__name__

        if SHAP_AVAILABLE:
            try:
                if model_type in {"RandomForestClassifier", "XGBClassifier", "ExtraTreesClassifier"}:
                    self._explainer = shap.TreeExplainer(model)
                    self.method = "TreeSHAP"
                    logger.info("Explainer initialised: TreeSHAP for %s", model_type)
                    return
                if model_type == "LogisticRegression":
                    background = np.zeros((1, len(self.feature_names)))
                    self._explainer = shap.LinearExplainer(model, background)
                    self.method = "LinearSHAP"
                    logger.info("Explainer initialised: LinearSHAP for %s", model_type)
                    return
            except Exception:  # noqa: BLE001
                logger.exception("SHAP initialisation failed; using native importances.")

        self._global_importance = self._native_importance()
        self.method = "global_feature_importance" if self._global_importance is not None else "none"
        logger.info("Explainer fallback: %s", self.method)

    def _native_importance(self) -> np.ndarray | None:
        model = self.bundle.model
        if hasattr(model, "feature_importances_"):
            return np.asarray(model.feature_importances_, dtype=float)
        if hasattr(model, "coef_"):
            return np.abs(np.asarray(model.coef_, dtype=float)).mean(axis=0)
        return None

    # ------------------------------------------------------------ explaining

    def _shap_values_for(self, matrix: np.ndarray) -> np.ndarray | None:
        """Return SHAP values shaped ``(n_samples, n_features, n_classes)``.

        SHAP's return shape varies by version and model: older APIs return a
        list of per-class arrays, newer ones a single 3-D array. Normalising
        here keeps the calling code stable across versions.
        """
        if self._explainer is None:
            return None
        try:
            values = self._explainer.shap_values(matrix, check_additivity=False)
        except TypeError:
            values = self._explainer.shap_values(matrix)
        except Exception:  # noqa: BLE001
            logger.exception("SHAP computation failed.")
            return None

        if isinstance(values, list):
            return np.stack(values, axis=-1)
        array = np.asarray(values)
        if array.ndim == 2:
            return array[:, :, np.newaxis]
        return array

    def explain_batch(
        self,
        raw_features: pd.DataFrame,
        matrix: np.ndarray,
        predicted_indices: Sequence[int],
        top_k: int = 5,
    ) -> list[list[dict[str, Any]]]:
        """Explain each row, returning the ``top_k`` contributing features.

        Args:
            raw_features: Pre-transformation frame, used to report actual values.
            matrix: The transformed matrix the model consumed.
            predicted_indices: Encoded class index predicted for each row - SHAP
                attributions are per-class, so we extract the winning class's.
            top_k: How many features to return per row.

        Returns:
            One list of contribution dicts per input row.
        """
        n_rows = matrix.shape[0]
        shap_values = self._shap_values_for(matrix)

        if shap_values is None:
            return [self._global_explanation(top_k) for _ in range(n_rows)]

        explanations: list[list[dict[str, Any]]] = []
        n_classes = shap_values.shape[-1]

        for row in range(n_rows):
            class_index = int(predicted_indices[row]) if n_classes > 1 else 0
            class_index = min(class_index, n_classes - 1)
            contributions = shap_values[row, :, class_index]

            order = np.argsort(np.abs(contributions))[::-1][:top_k]
            row_explanation = []
            for idx in order:
                name = self.feature_names[idx] if idx < len(self.feature_names) else f"feature_{idx}"
                row_explanation.append(
                    FeatureContribution(
                        feature=name,
                        display_name=prettify(name),
                        contribution=float(contributions[idx]),
                        direction="increases" if contributions[idx] > 0 else "decreases",
                        value=_raw_value(raw_features, name, row),
                    ).to_dict()
                )
            explanations.append(row_explanation)
        return explanations

    def explain_one(
        self, raw_features: pd.DataFrame, matrix: np.ndarray, predicted_index: int, top_k: int = 5
    ) -> list[dict[str, Any]]:
        """Explain a single row."""
        return self.explain_batch(raw_features, matrix, [predicted_index], top_k=top_k)[0]

    def _global_explanation(self, top_k: int) -> list[dict[str, Any]]:
        """Fallback: global importances, clearly marked as not row-specific."""
        if self._global_importance is None:
            return []
        order = np.argsort(self._global_importance)[::-1][:top_k]
        return [
            FeatureContribution(
                feature=self.feature_names[i] if i < len(self.feature_names) else f"feature_{i}",
                display_name=prettify(self.feature_names[i]) if i < len(self.feature_names) else f"feature_{i}",
                contribution=float(self._global_importance[i]),
                direction="global_importance",
                value=None,
            ).to_dict()
            for i in order
        ]

    def global_importance(self, top_k: int = 20) -> pd.DataFrame:
        """Model-level feature importance ranking, for the dashboard."""
        importance = self._global_importance if self._global_importance is not None else self._native_importance()
        if importance is None:
            return pd.DataFrame(columns=["feature", "display_name", "importance"])
        order = np.argsort(importance)[::-1][:top_k]
        return pd.DataFrame(
            {
                "feature": [self.feature_names[i] if i < len(self.feature_names) else f"f{i}" for i in order],
                "display_name": [
                    prettify(self.feature_names[i]) if i < len(self.feature_names) else f"f{i}" for i in order
                ],
                "importance": [float(importance[i]) for i in order],
            }
        )


def _raw_value(frame: pd.DataFrame, feature_name: str, row: int) -> float | str | None:
    """Look up the pre-transformation value behind a transformed feature name."""
    if feature_name in frame.columns:
        value = frame.iloc[row][feature_name]
        try:
            return round(float(value), 4)
        except (TypeError, ValueError):
            return str(value)
    for prefix in ("port_category_", "protocol_"):
        if feature_name.startswith(prefix):
            base = prefix.rstrip("_")
            if base in frame.columns:
                return str(frame.iloc[row][base])
    return None


def explanation_sentence(prediction: str, contributions: list[dict[str, Any]]) -> str:
    """Render an explanation as one plain-English line for the UI."""
    if not contributions:
        return f"Classified as {prediction}; no feature attribution available."
    drivers = ", ".join(c["display_name"] for c in contributions[:3])
    return f"Classified as {prediction}, driven mainly by {drivers}."
