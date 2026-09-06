# The ephemeral execution sandbox.
#
# One container per session, created on first command and destroyed when the
# session goes idle. It holds a shell, a toolchain, and nothing else: no
# credentials, no Docker socket, no route off the host, and no write access
# anywhere except the workspace volume and /tmp.
#
# Keep it small. Every session pays for this image's size in page cache, and
# every package in it is something an escaped process could use.
#
#   docker build -f deploy/sandbox.Dockerfile -t shree-sandbox:latest .

FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        bash coreutils findutils grep sed gawk \
        git curl ca-certificates \
        python3 python3-pip python3-venv \
        build-essential \
        nodejs npm \
    && rm -rf /var/lib/apt/lists/*

# uid 1000 matches the --user the driver passes, so files the agent creates on
# the mounted volume are owned by the same account the API writes as.
RUN userdel -r ubuntu 2>/dev/null || true \
    && useradd --uid 1000 --create-home --shell /bin/bash agent \
    && mkdir -p /workspace && chown agent:agent /workspace

USER agent
WORKDIR /workspace

# The driver keeps the container alive itself and runs commands through
# `docker exec`, so the entrypoint only has to not exit.
CMD ["sleep", "infinity"]
