"""FastAPI service exposing the intrusion detection model over HTTP.

Endpoints
---------
``GET  /health``         Liveness and readiness.
``GET  /model-info``     Metadata about the deployed model.
``POST /predict``        Score one flow, with explanation.
``POST /predict/batch``  Score an uploaded CSV of flows.

Design notes
------------
*Model loading happens once, at startup*, through the lifespan handler. Loading
per request would add hundreds of milliseconds and defeat the point of a
service. A missing model does not crash the process: the API starts in a
degraded state and returns ``503`` from the prediction routes, so an
orchestrator can distinguish "container is broken" from "model not deployed
yet".

*Uploads are bounded twice* - once on declared size before reading, once on row
count after parsing. Both limits are configuration, not literals, so a
deployment can tighten them without a code change.
"""

from __future__ import annotations

import io
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.schemas import (
    BatchPredictionResponse,
    BatchSummary,
    ErrorResponse,
    HealthResponse,
    ModelInfoResponse,
    NetworkFlowRequest,
    PredictionResponse,
)
from src.config import PATHS, SERVICE, get_logger
from src.predict import ThreatPredictor, get_predictor

logger = get_logger("api")

_STARTED_AT = time.time()
_PREDICTOR: ThreatPredictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load model artifacts once at startup, release them at shutdown."""
    global _PREDICTOR
    try:
        _PREDICTOR = get_predictor()
        logger.info("API startup complete: model artifacts loaded.")
    except FileNotFoundError:
        _PREDICTOR = None
        logger.error(
            "No trained model found at %s. The API will start in DEGRADED mode; "
            "prediction endpoints will return 503 until `python scripts/train_model.py` is run.",
            PATHS.model_bundle,
        )
    except Exception:  # noqa: BLE001
        _PREDICTOR = None
        logger.exception("Unexpected failure while loading model artifacts.")
    yield
    _PREDICTOR = None
    logger.info("API shutdown complete.")


app = FastAPI(
    title="AI-Powered Network Intrusion Detection API",
    description=(
        "Classifies network flow records into benign traffic or an attack family, "
        "scores each flow for anomalousness with an unsupervised detector, and "
        "returns a composite risk rating.\n\n"
        "**Risk scores are model-derived engineering heuristics, not calibrated "
        "probabilities and not a production security standard.**"
    ),
    version=SERVICE.model_version,
    lifespan=lifespan,
    responses={
        422: {"model": ErrorResponse, "description": "Request failed validation"},
        503: {"model": ErrorResponse, "description": "Model not loaded"},
    },
)

app.add_middleware(
    CORSMiddleware,
    # Defaults to the local Streamlit origin; override with NIDS_CORS_ORIGINS.
    allow_origins=[
        origin.strip()
        for origin in __import__("os")
        .getenv("NIDS_CORS_ORIGINS", "http://localhost:8501,http://127.0.0.1:8501")
        .split(",")
        if origin.strip()
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Middleware & error handling
# --------------------------------------------------------------------------- #


@app.middleware("http")
async def add_request_context(request: Request, call_next):  # type: ignore[no-untyped-def]
    """Attach a request id and log timing.

    Deliberately logs the *shape* of a request (path, status, duration) and
    never the body. Flow records are traffic metadata and may be sensitive, so
    they must not end up in application logs.
    """
    request_id = uuid.uuid4().hex[:12]
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:  # noqa: BLE001
        elapsed = (time.perf_counter() - started) * 1000
        logger.exception("request_id=%s %s %s failed after %.1fms",
                         request_id, request.method, request.url.path, elapsed)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error.", "error_type": "internal_error"},
        )
    elapsed = (time.perf_counter() - started) * 1000
    response.headers["X-Request-ID"] = request_id
    logger.info(
        "request_id=%s %s %s -> %d (%.1fms)",
        request_id, request.method, request.url.path, response.status_code, elapsed,
    )
    return response


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Return errors in the uniform :class:`ErrorResponse` envelope."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "error_type": "http_error"},
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Return a 422 that describes *where* validation failed, not what was sent.

    FastAPI's default handler echoes the offending input back in the response
    body. That is a problem here for two reasons. Practically, a payload
    containing ``Infinity`` cannot be re-serialised to JSON, so the default
    handler raises and the clean 422 degrades into a 500. From a security
    standpoint, reflecting attacker-controlled input back into a response is a
    habit worth not forming in a service whose whole job is handling hostile
    traffic. We therefore report the field location and the reason, and drop
    the raw value.
    """
    problems = [
        {
            "field": ".".join(str(part) for part in error.get("loc", ())),
            "message": error.get("msg", "Invalid value."),
            "type": error.get("type", "value_error"),
        }
        for error in exc.errors()
    ]
    logger.info("Rejected invalid request to %s: %d field error(s)", request.url.path, len(problems))
    return JSONResponse(
        status_code=422,
        content={
            "detail": "Request validation failed.",
            "error_type": "validation_error",
            "errors": problems,
        },
    )


def require_predictor() -> ThreatPredictor:
    """Dependency that yields the loaded predictor or fails with 503."""
    if _PREDICTOR is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Model is not loaded. Train one with `python scripts/train_model.py` "
                "and restart the service."
            ),
        )
    return _PREDICTOR


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health() -> HealthResponse:
    """Liveness probe. Returns 200 even when degraded so the container is not
    killed by an orchestrator merely because no model has been deployed yet."""
    loaded = _PREDICTOR is not None
    return HealthResponse(
        status="healthy" if loaded else "degraded",
        model_loaded=loaded,
        anomaly_detector_loaded=bool(loaded and _PREDICTOR is not None and _PREDICTOR.anomaly is not None),
        api_version=SERVICE.model_version,
        uptime_seconds=round(time.time() - _STARTED_AT, 2),
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["System"])
async def model_info(predictor: ThreatPredictor = Depends(require_predictor)) -> ModelInfoResponse:
    """Describe the deployed model: name, version, classes, features, metrics."""
    info: dict[str, Any] = predictor.model_info()
    return ModelInfoResponse(**{
        key: info.get(key)
        for key in ModelInfoResponse.model_fields
        if info.get(key) is not None
    })


@app.post("/predict", response_model=PredictionResponse, tags=["Detection"])
async def predict(
    flow: NetworkFlowRequest,
    explain: bool = Query(True, description="Include SHAP feature attributions."),
    top_k: int = Query(5, ge=1, le=20, description="Number of contributing features."),
    predictor: ThreatPredictor = Depends(require_predictor),
) -> PredictionResponse:
    """Classify a single network flow and return its risk assessment.

    The response carries three independent signals so an analyst can triage
    without trusting a single number: the supervised class prediction with its
    confidence, the unsupervised anomaly verdict, and the composite risk score
    that blends them with model-independent traffic heuristics.
    """
    try:
        result = predictor.predict_one(flow.to_flow_dict(), explain=explain, top_k=top_k)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Prediction failed.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed. See server logs for details.",
        ) from exc
    return PredictionResponse(**result.to_dict())


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Detection"])
async def predict_batch(
    file: UploadFile = File(..., description="CSV of network flow records."),
    preview_rows: int = Query(100, ge=0, le=5000, description="Per-row results to return."),
    predictor: ThreatPredictor = Depends(require_predictor),
) -> BatchPredictionResponse:
    """Score an uploaded CSV of flows and return predictions plus summary stats.

    Upload handling is defensive by design:

    * only ``.csv`` is accepted, checked on the filename suffix;
    * the payload is size-capped before parsing, so a large upload cannot
      exhaust memory during read;
    * the parsed frame is row-capped, bounding scoring time;
    * the full result set is summarised, while only ``preview_rows`` rows are
      echoed back, keeping the response bounded regardless of input size.
    """
    filename = Path(file.filename or "upload.csv").name
    if Path(filename).suffix.lower() not in SERVICE.allowed_upload_suffixes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type. Allowed: {', '.join(SERVICE.allowed_upload_suffixes)}",
        )

    raw = await file.read()
    if len(raw) > SERVICE.max_upload_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds the {SERVICE.max_upload_mb:.0f} MB limit.",
        )
    if not raw.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty.")

    try:
        frame = pd.read_csv(io.BytesIO(raw))
    except UnicodeDecodeError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File is not valid UTF-8 text.",
        ) from exc
    except Exception as exc:  # noqa: BLE001 - pandas raises many parser errors
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not parse the file as CSV.",
        ) from exc

    if frame.empty:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="CSV contains no rows.")

    rows_received = len(frame)
    truncated = rows_received > SERVICE.max_upload_rows
    if truncated:
        frame = frame.head(SERVICE.max_upload_rows)
        logger.warning("Upload truncated from %d to %d rows.", rows_received, len(frame))

    try:
        scored = predictor.predict_csv(frame, explain=False)
        summary = predictor.summarize(scored)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Batch scoring failed for %s", filename)
        raise HTTPException(
            status_code=422,
            detail="Scoring failed. Check that the CSV contains network flow feature columns.",
        ) from exc

    result_columns = [
        column
        for column in ("prediction", "confidence", "risk_score", "risk_level",
                       "anomaly_score", "is_anomaly", "is_malicious")
        if column in scored.columns
    ]
    preview = (
        scored[result_columns].head(preview_rows).to_dict(orient="records") if preview_rows else []
    )

    return BatchPredictionResponse(
        filename=filename,
        rows_received=rows_received,
        rows_scored=int(len(scored)),
        summary=BatchSummary(**{
            key: value for key, value in summary.items() if key in BatchSummary.model_fields
        }),
        predictions=preview,
        truncated=truncated or len(preview) < len(scored),
    )


@app.get("/", include_in_schema=False)
async def root() -> dict[str, str]:
    """Point callers at the interactive documentation."""
    return {
        "service": "AI-Powered Network Intrusion Detection API",
        "version": SERVICE.model_version,
        "docs": "/docs",
        "health": "/health",
    }


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run("api.main:app", host=SERVICE.api_host, port=SERVICE.api_port, reload=False)
