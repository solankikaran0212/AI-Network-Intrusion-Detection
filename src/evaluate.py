"""Model evaluation with metrics chosen for a security context.

The headline point this module encodes: **accuracy is the wrong metric for
intrusion detection.** CIC-IDS2017 is roughly 80% benign, so a model that
predicts BENIGN for every single flow scores 80% accuracy while detecting
nothing at all. Worse, that failure mode is invisible if accuracy is the only
number reported.

We therefore lead with:

* **Recall per attack class** - what fraction of real attacks did we catch? A
  missed attack (false negative) means an intrusion proceeds undetected. That is
  a breach.
* **Macro-F1** - the unweighted mean across classes, so a rare class such as
  Infiltration counts as much as the dominant benign class. This is the model
  selection metric.
* **Precision** - the alert-fatigue axis. A detector with 20% precision
  generates four false alarms per true one; analysts learn to ignore it, which
  turns a technically-working system into a non-working one.

Both error types matter, in different ways. False negatives cost a breach; false
positives cost analyst attention, and enough of them cost the system's
credibility. F1 is reported because it forces that trade-off into a single
comparable number.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import BENIGN_LABEL, PATHS, get_logger

logger = get_logger(__name__)


@dataclass
class EvaluationResult:
    """All metrics for one model on one split."""

    model_name: str
    split: str
    accuracy: float
    balanced_accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_precision: float
    weighted_recall: float
    weighted_f1: float
    roc_auc_ovr: float | None
    per_class: dict[str, dict[str, float]]
    confusion_matrix: list[list[int]]
    labels: list[str]
    attack_detection_rate: float
    false_alarm_rate: float
    false_negatives: int
    false_positives: int
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "split": self.split,
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "macro_precision": self.macro_precision,
            "macro_recall": self.macro_recall,
            "macro_f1": self.macro_f1,
            "weighted_precision": self.weighted_precision,
            "weighted_recall": self.weighted_recall,
            "weighted_f1": self.weighted_f1,
            "roc_auc_ovr": self.roc_auc_ovr,
            "per_class": self.per_class,
            "confusion_matrix": self.confusion_matrix,
            "labels": self.labels,
            "attack_detection_rate": self.attack_detection_rate,
            "false_alarm_rate": self.false_alarm_rate,
            "false_negatives": self.false_negatives,
            "false_positives": self.false_positives,
            **self.extra,
        }

    def summary_line(self) -> str:
        return (
            f"{self.model_name:<22} acc={self.accuracy:.4f}  macroF1={self.macro_f1:.4f}  "
            f"macroRecall={self.macro_recall:.4f}  detection={self.attack_detection_rate:.4f}  "
            f"falseAlarm={self.false_alarm_rate:.4f}"
        )


def binary_security_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Collapse multiclass results to the attack/benign view a SOC cares about.

    Returns detection rate (recall on "is an attack"), false alarm rate
    (benign flows wrongly alerted on), and raw FN/FP counts.
    """
    true_attack = np.asarray(y_true) != BENIGN_LABEL
    pred_attack = np.asarray(y_pred) != BENIGN_LABEL

    true_positives = int(np.sum(true_attack & pred_attack))
    false_negatives = int(np.sum(true_attack & ~pred_attack))
    false_positives = int(np.sum(~true_attack & pred_attack))
    true_negatives = int(np.sum(~true_attack & ~pred_attack))

    detection_rate = true_positives / max(true_positives + false_negatives, 1)
    false_alarm_rate = false_positives / max(false_positives + true_negatives, 1)

    return {
        "attack_detection_rate": float(detection_rate),
        "false_alarm_rate": float(false_alarm_rate),
        "false_negatives": false_negatives,
        "false_positives": false_positives,
        "true_positives": true_positives,
        "true_negatives": true_negatives,
    }


def evaluate_model(
    model_name: str,
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    y_proba: np.ndarray | None = None,
    class_labels: list[str] | None = None,
    split: str = "test",
) -> EvaluationResult:
    """Compute the full metric suite for one model.

    ``roc_auc_ovr`` uses one-vs-rest with macro averaging. It is skipped (set to
    ``None``) when probabilities are unavailable or when a class is missing from
    the split, since AUC is undefined for a class with no positive samples.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = class_labels or sorted(set(y_true.tolist()) | set(y_pred.tolist()))

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0, labels=labels
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0, labels=labels
    )

    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    per_class = {
        label: {
            "precision": float(values["precision"]),
            "recall": float(values["recall"]),
            "f1": float(values["f1-score"]),
            "support": int(values["support"]),
        }
        for label, values in report.items()
        if label in labels and isinstance(values, dict)
    }

    roc_auc: float | None = None
    if y_proba is not None:
        try:
            present = sorted(set(y_true.tolist()))
            if len(present) == len(labels) and y_proba.shape[1] == len(labels):
                roc_auc = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro", labels=labels))
        except (ValueError, IndexError) as exc:
            logger.debug("ROC-AUC unavailable for %s: %s", model_name, exc)

    security = binary_security_metrics(y_true, y_pred)

    return EvaluationResult(
        model_name=model_name,
        split=split,
        accuracy=float(accuracy_score(y_true, y_pred)),
        balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
        macro_precision=float(macro_p),
        macro_recall=float(macro_r),
        macro_f1=float(macro_f1),
        weighted_precision=float(weighted_p),
        weighted_recall=float(weighted_r),
        weighted_f1=float(weighted_f1),
        roc_auc_ovr=roc_auc,
        per_class=per_class,
        confusion_matrix=confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        labels=list(labels),
        attack_detection_rate=security["attack_detection_rate"],
        false_alarm_rate=security["false_alarm_rate"],
        false_negatives=int(security["false_negatives"]),
        false_positives=int(security["false_positives"]),
        extra={"true_positives": security["true_positives"], "true_negatives": security["true_negatives"]},
    )


def comparison_table(results: list[EvaluationResult]) -> pd.DataFrame:
    """Side-by-side model comparison, sorted by the selection metric."""
    rows = [
        {
            "Model": r.model_name,
            "Accuracy": round(r.accuracy, 4),
            "Balanced Acc": round(r.balanced_accuracy, 4),
            "Macro Precision": round(r.macro_precision, 4),
            "Macro Recall": round(r.macro_recall, 4),
            "Macro F1": round(r.macro_f1, 4),
            "Weighted F1": round(r.weighted_f1, 4),
            "ROC-AUC (OvR)": round(r.roc_auc_ovr, 4) if r.roc_auc_ovr is not None else None,
            "Detection Rate": round(r.attack_detection_rate, 4),
            "False Alarm Rate": round(r.false_alarm_rate, 4),
            "False Negatives": r.false_negatives,
        }
        for r in results
    ]
    return pd.DataFrame(rows).sort_values("Macro F1", ascending=False).reset_index(drop=True)


def confusion_matrix_frame(result: EvaluationResult) -> pd.DataFrame:
    """Confusion matrix as a labelled frame (rows=true, cols=predicted)."""
    return pd.DataFrame(
        result.confusion_matrix,
        index=[f"true_{label}" for label in result.labels],
        columns=[f"pred_{label}" for label in result.labels],
    )


def save_metrics(payload: dict[str, Any], path: Path | None = None) -> Path:
    """Write the metrics report as JSON for the dashboard and README to read."""
    target = Path(path) if path is not None else PATHS.metrics_report
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=_json_default)
    logger.info("Saved metrics report to %s", target)
    return target


def load_metrics(path: Path | None = None) -> dict[str, Any]:
    """Read a previously written metrics report."""
    target = Path(path) if path is not None else PATHS.metrics_report
    if not target.exists():
        raise FileNotFoundError(f"No metrics report at {target}. Run scripts/train_model.py first.")
    with target.open(encoding="utf-8") as handle:
        return json.load(handle)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    return str(value)
