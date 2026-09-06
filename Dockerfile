# URA-Shree production image.
#
# Three stages, for three different reasons:
#
#   web      Node builds the frontend. Node is 300MB of toolchain that has no
#            business in the runtime image, so only `dist/` crosses over.
#   deps     Wheels are compiled once into a virtualenv. Build tools stay here.
#   runtime  CUDA runtime (not devel: no nvcc, no headers, roughly a third of
#            the size) plus the venv and the source.
#
# Build:
#   docker build -t shree-api:latest .
#   docker build -t shree-api:cpu --build-arg CUDA_IMAGE=ubuntu:24.04 \
#                --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cpu .

ARG CUDA_IMAGE=nvidia/cuda:12.6.3-runtime-ubuntu24.04

# -- stage: frontend ----------------------------------------------------------
FROM node:22-alpine AS web

WORKDIR /build
# Manifests first: this layer is cached until a dependency actually changes,
# which is far less often than the source does.
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./
# Baked into the bundle at build time. Empty means "same origin as the page",
# which is what the reverse proxy setup wants.
ARG VITE_API_BASE_URL=""
ENV VITE_API_BASE_URL=$VITE_API_BASE_URL
RUN npm run build


# -- stage: python dependencies ----------------------------------------------
FROM ${CUDA_IMAGE} AS deps

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-dev build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Torch first and on its own: it is by far the largest wheel, and pinning its
# index here keeps the CUDA build matched to the base image.
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu126
RUN pip install --upgrade pip && pip install torch --index-url ${TORCH_INDEX}

COPY requirements.txt requirements-cloud.txt ./
RUN pip install -r requirements.txt -r requirements-cloud.txt


# -- stage: runtime -----------------------------------------------------------
FROM ${CUDA_IMAGE} AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    SHREE_MODE=cloud \
    SHREE_LOG_JSON=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 git curl ca-certificates \
    # The Docker CLI, not the daemon: the ContainerDriver drives sandboxes on a
    # socket mounted from outside. See docker-compose.yml for why that socket
    # must be a filtered proxy and not the raw host one.
        docker.io \
    && rm -rf /var/lib/apt/lists/*

COPY --from=deps /opt/venv /opt/venv

WORKDIR /app
COPY agent/ ./agent/
COPY inference/ ./inference/
COPY model/ ./model/
COPY providers/ ./providers/
COPY server/ ./server/
COPY tokenizer/ ./tokenizer/
COPY tools/ ./tools/
COPY training/ ./training/
COPY scripts/ ./scripts/
COPY configs/ ./configs/
COPY checkpoints/ ./checkpoints/
COPY deploy/gunicorn.conf.py ./deploy/gunicorn.conf.py
COPY --from=web /build/dist ./frontend/dist

# The API never needs to write to its own source tree. Workspaces live on a
# mounted volume, owned by this user.
RUN useradd --uid 10001 --create-home --shell /usr/sbin/nologin shree \
    && mkdir -p /var/lib/shree/workspaces \
    && chown -R shree:shree /var/lib/shree /app
USER shree

ENV SHREE_WORKSPACES_ROOT=/var/lib/shree/workspaces
VOLUME ["/var/lib/shree/workspaces"]

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/api/health || exit 1

CMD ["gunicorn", "server.api:app", "--config", "deploy/gunicorn.conf.py"]
