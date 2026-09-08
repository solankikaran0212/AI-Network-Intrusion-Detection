# syntax=docker/dockerfile:1

# Multi-stage build: wheels are compiled in a builder stage and only the
# installed packages are copied forward. This keeps the runtime image free of
# compilers and build headers, which shrinks it and reduces attack surface -
# relevant for a security tool.

# --------------------------------------------------------------------------- #
# Stage 1: build dependencies
# --------------------------------------------------------------------------- #
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# libgomp is required by both XGBoost and scikit-learn for OpenMP threading.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt .
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install -r requirements.txt

# --------------------------------------------------------------------------- #
# Stage 2: runtime
# --------------------------------------------------------------------------- #
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

# Run as an unprivileged user. A container that parses untrusted CSV uploads
# should not be doing so as root.
RUN useradd --create-home --uid 10001 nids
WORKDIR /app

COPY --chown=nids:nids . .

# Writable directories for artifacts produced at runtime.
RUN mkdir -p data/processed models reports logs \
    && chown -R nids:nids /app

USER nids

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
