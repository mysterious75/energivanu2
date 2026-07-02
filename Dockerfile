# =============================================================================
# Energivanu — Production Dockerfile
# Multi-stage build for minimal image size and security
# =============================================================================
# Usage:
#   docker build -t energivanu:latest .
#   docker run -p 8000:8000 energivanu:latest
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1: Builder — Install dependencies and build the package
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /build

# Install system build dependencies required by numpy/torch compilation
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project metadata and source code
COPY pyproject.toml README.md LICENSE LICENSE-COMMERCIAL ./
COPY src/ ./src/

# Install the package with API dependencies (FastAPI + Uvicorn)
# Using --no-cache-dir to minimize layer size
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e ".[api]"

# ---------------------------------------------------------------------------
# Stage 2: Production — Minimal runtime image
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS production

# -- Metadata labels --------------------------------------------------------
LABEL maintainer="Ved Kumar <support@voraprotocol.com>" \
      description="Energivanu — ML-powered GPU data center power optimization" \
      version="0.1.0" \
      license="AGPL-3.0-or-later" \
      url="https://github.com/mysterious75/Energivanu" \
      vendor="VORAPROTOCOL"

# -- Environment variables ---------------------------------------------------
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ENERGIVANU_ENV=production

# -- Install runtime dependencies only --------------------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

# -- Create non-root user for security --------------------------------------
RUN groupadd --gid 1000 energivanu && \
    useradd --uid 1000 --gid energivanu --shell /bin/bash --create-home energivanu

WORKDIR /app

# -- Copy installed Python packages from builder stage -----------------------
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# -- Copy application source and configuration -------------------------------
COPY --chown=energivanu:energivanu src/ ./src/
COPY --chown=energivanu:energivanu config/ ./config/
COPY --chown=energivanu:energivanu models/ ./models/

# -- Create logs directory ----------------------------------------------------
RUN mkdir -p /app/logs && chown energivanu:energivanu /app/logs

# -- Expose the API port -----------------------------------------------------
EXPOSE 8000

# -- Health check — verifies the API is responding ----------------------------
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# -- Switch to non-root user -------------------------------------------------
USER energivanu

# -- Default entrypoint: start the FastAPI server ----------------------------
CMD ["uvicorn", "energivanu.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
