# ===========================================================================
# AgentOS Demo
# ===========================================================================
# Multi-agent system built with Agno.
# Runs as a non-root user (app) with:
#   /app    - application code
# ===========================================================================

FROM node:20-bookworm-slim AS node-runtime

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

RUN groupadd -r app && useradd -r -g app -m -s /bin/bash app

# Git/uv prepare Python MCP repositories. Node/npm are copied from the
# official Node image so the MCP wizard supports Node repositories without
# pulling Debian's very large npm dependency graph into the final image.
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

COPY --from=node-runtime /usr/local/bin/node /usr/local/bin/node
COPY --from=node-runtime /usr/local/lib/node_modules /usr/local/lib/node_modules
RUN ln -sf /usr/local/lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
    && ln -sf /usr/local/lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx \
    && node --version \
    && npm --version

WORKDIR /app
COPY requirements.txt .
RUN uv pip sync requirements.txt --system \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    --index-strategy unsafe-best-match
COPY . .

RUN chmod 755 /app
RUN chmod +x /app/scripts/entrypoint.sh
ENTRYPOINT ["/app/scripts/entrypoint.sh"]

USER app
WORKDIR /app

EXPOSE 8000
CMD ["chill"]
