# ===========================================================================
# AgentOS Demo
# ===========================================================================
# Multi-agent system built with Agno.
# Runs as a non-root user (app) with:
#   /app    - application code
# ===========================================================================

FROM python:3.12-slim

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

# ---------------------------------------------------------------------------
# Create non-root user
# ---------------------------------------------------------------------------
RUN groupadd -r app && useradd -r -g app -m -s /bin/bash app

# ---------------------------------------------------------------------------
# System libraries + uv
# ---------------------------------------------------------------------------
# docling pulls in opencv, which needs a few native shared libs not present in
# the slim base image (otherwise: "libxcb.so.1: cannot open shared object file").
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libxcb1 \
        libxext6 \
        libsm6 \
        libxrender1 \
        gcc \
        g++ \
        make \
        git \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv

# ---------------------------------------------------------------------------
# Application code
# ---------------------------------------------------------------------------
WORKDIR /app
COPY requirements.txt .
RUN uv pip sync requirements.txt --system \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    --index-strategy unsafe-best-match
COPY . .

# ---------------------------------------------------------------------------
# Directory setup & permissions
# ---------------------------------------------------------------------------
RUN chmod 755 /app

# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
RUN chmod +x /app/scripts/entrypoint.sh
ENTRYPOINT ["/app/scripts/entrypoint.sh"]

# ---------------------------------------------------------------------------
# Switch to non-root user
# ---------------------------------------------------------------------------
USER app
WORKDIR /app

EXPOSE 8000

# ---------------------------------------------------------------------------
# Default command (overridden by compose)
# ---------------------------------------------------------------------------
CMD ["chill"]
