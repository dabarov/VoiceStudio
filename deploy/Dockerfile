# Base image for the Python/PyTorch runtime stage. The default builds the
# CUDA variant; CI also builds a ROCm/AMD variant (issue #1165) by overriding:
#   BASE_IMAGE=rocm/pytorch:rocm10.0_ubuntu24.04_py3.12_pytorch_release_2.13.0
#   GPU_FLAVOR=rocm
# Both bases ship patched torch 2.13.0 preinstalled; the dependency install
# below deliberately preserves them (see the GPU_FLAVOR guard).
ARG BASE_IMAGE=pytorch/pytorch:2.13.0-cuda13.0-cudnn9-runtime@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407

# ==========================================
# Builder Stage: Compile React Frontend
# ==========================================
# Keep in lockstep with package.json packageManager (tests/test_bun_toolchain_pins.py).
FROM oven/bun:1.4.2-alpine AS frontend-builder
WORKDIR /app

# Monorepo — bun workspace with lockfile at repo root. Copy manifests first
# so `bun install` caches independently of source edits.
COPY package.json bun.lock ./
COPY electron/package.json ./electron/
COPY config/ ./config/
COPY patches/ ./patches/

RUN bun install --frozen-lockfile

# Build the maintained Electron renderer as the browser UI. `electron/src/shared`
# contains transitional shared modules only; it has no app entry point.
COPY frontend/ ./frontend/
COPY electron/ ./electron/
RUN bun run --cwd electron build:web

# ==========================================
# Runtime Stage: Python & PyTorch Backend
# ==========================================
FROM ${BASE_IMAGE} AS runtime
WORKDIR /app

# Enable unbuffered logs and optimizations
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/omnivoice_data/huggingface
# Allow bare imports (from core.config, from services.*, etc.) when
# uvicorn is started as `backend.main:app` from WORKDIR /app.
ENV PYTHONPATH=/app/backend
# Headless server deployment: relax the desktop-only loopback origin gate.
# Docker's network NAT rewrites the client host to the bridge gateway, so the
# gate would otherwise 403 the operator out of /system/* and /api/settings/*
# ("Loopback origin required", issue #261). Exposure is governed by the
# operator's `-p` port mapping plus the optional share PIN. Desktop builds
# never set this, so their loopback boundary is unchanged.
ENV OMNIVOICE_SERVER_MODE=1

# Install system dependencies (FFmpeg is critical for torchaudio/scene splitting)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# PEP 668: the ROCm base (Ubuntu 24.04) marks its system Python
# EXTERNALLY-MANAGED, which would refuse installing into the selected base
# interpreter. Inside a single-purpose container image that is exactly what
# we want. No-ops on the conda-based CUDA image.
ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV UV_BREAK_SYSTEM_PACKAGES=1

# Install `uv` for blazing-fast reliable pip resolution
# (`python3 -m pip` — not every base symlinks a bare `pip` onto PATH)
RUN python3 -m pip install --no-cache-dir uv

# Copy python packaging specs (README.md required by hatchling metadata)
COPY pyproject.toml uv.lock README.md ./
COPY deploy/torch-constraints.txt ./deploy/torch-constraints.txt
COPY vendor/python/wheels/ ./vendor/python/wheels/

# Install the project (non-editable — no need for -e in containers).
# Uses `uv` for exponentially faster resolution than plain pip.
#
# Target the exact interpreter selected by the base image. The ROCm image has
# both /opt/venv/bin/python3 (ROCm torch) and /usr/bin/python (a CUDA-default
# environment); `--system` used the latter while the build guard used the
# former, so a green image launched a CPU-only backend on AMD (#1274).
#
# NOTE: `uv pip install` (without --upgrade) keeps already-installed packages
# that satisfy the requirements, so the base image's GPU-built torch/torchaudio
# (2.13.0, satisfying our security floor) survive this step instead of being
# clobbered by PyPI's CUDA-default wheels. That property is what makes the
# ROCm variant possible at all — the guard below pins it down.
#
# --constraint because `uv pip install` ignores `[tool.uv]
# constraint-dependencies` (that is a project-API setting), so without it the
# trio resolves on its bare lower bounds and torch may move while torchvision
# stays put — an ABI mismatch at import (#1357). The pins carry no local
# segment, so they match the base image's +cu130 / +rocm10.0 builds rather than
# replacing them.
RUN uv pip install --python "$(command -v python3)" --no-cache \
    --constraint deploy/torch-constraints.txt --override deploy/torch-constraints.txt --find-links vendor/python/wheels .

# Guard (fails the build, not the user at runtime): assert the dependency
# install did NOT replace the base image's GPU torch. A future dep bump that
# forces a different torch version would otherwise silently ship a CUDA build
# in the ROCm image (= CPU-only for AMD users) — catch it here instead.
ARG GPU_FLAVOR=cuda
RUN python3 -c "import os, torch, torchaudio, torchvision; \
    flavor = os.environ['GPU_FLAVOR']; \
    accel = torch.version.hip if flavor == 'rocm' else torch.version.cuda; \
    print(f'torch={torch.__version__} torchaudio={torchaudio.__version__} torchvision={torchvision.__version__} {flavor}={accel}'); \
    assert accel, f'base image {flavor} torch was clobbered (now {torch.__version__})'; \
    import torchvision.ops; torchvision.ops.nms; \
    print('torchvision C++ ops resolve against this torch')"

# CTranslate2 (WhisperX, faster-whisper) links CUDA 12's cuDNN 9 / cuBLAS 12,
# while the base image supplies CUDA 13 libraries. Missing native libraries
# can abort the backend process outright rather than raising. scripts/setup.py
# side-loads the cuDNN 9 / cuBLAS 12 libraries for source installs; the image needs the same
# shim or Docker users lose every CTranslate2 ASR engine (#2050).
#
# The target is derived from sys.prefix rather than hardcoded: backend/core/
# cudnn8.py looks for <sys.prefix>/lib/pythonX.Y/site-packages/ctranslate2_cuda12,
# and sys.prefix differs between the conda-based CUDA image and the ROCm venv.
# --no-deps installs just these two wheels, leaving the base image's torch
# stack untouched. Skipped for ROCm, which does not use cuDNN.
RUN if [ "$GPU_FLAVOR" = "cuda" ]; then \
      target="$(python3 -c "import os, sys; print(os.path.join(sys.prefix, 'lib', 'python%d.%d' % sys.version_info[:2], 'site-packages', 'ctranslate2_cuda12'))")" && \
      uv pip install --python "$(command -v python3)" --no-cache --no-deps \
        --target "$target" nvidia-cudnn-cu12==9.20.0.48 nvidia-cublas-cu12==12.9.1.4 && \
      python3 -c "import os, sys; root = os.path.join(sys.prefix, 'lib', 'python%d.%d' % sys.version_info[:2], 'site-packages', 'ctranslate2_cuda12', 'nvidia'); \
        d = os.path.join(root, 'cudnn', 'lib'); libs = [f for f in os.listdir(d) if '.so.9' in f]; \
        assert len(libs) >= 5, 'Incomplete cuDNN 9 libraries in ' + d; \
        assert all(os.path.isfile(os.path.join(root, 'cublas', 'lib', n)) for n in ('libcublas.so.12', 'libcublasLt.so.12')), 'Incomplete cuBLAS 12 libraries'; \
        print('cuDNN 9 / cuBLAS 12 compat libraries: %d' % len(libs))"; \
    fi

# Copy application source
COPY backend/ ./backend/
COPY omnivoice/ ./omnivoice/
COPY config/ ./config/
# Alembic config so schema migrations run natively on existing volumes
# (without it the backend fell back to the additive-column self-heal —
# functional, but the real migration chain is the first-class path).
COPY alembic.ini ./

# Copy the pre-built React frontend from the builder stage
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# Stable container interface for `docker run -P` and published-image consumers.
EXPOSE 3900

# Image-level health probe (compose files define their own; this covers plain
# `docker run`). Generous start period: first boot creates the venv-less
# schema + may pull model metadata before /health answers.
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=5 \
  CMD ["python3", "backend/docker_entrypoint.py", "--health-check"]

# Mount points for persistent data (sqlite db, user voices, huggingface cache)
VOLUME ["/app/omnivoice_data"]

# Bind to 0.0.0.0 for external access. The entrypoint reads the manifest default
# (or OMNIVOICE_PORT) and forwards any explicit Uvicorn options after the image.
ENV OMNIVOICE_BIND_HOST=0.0.0.0
ENTRYPOINT ["python3", "backend/docker_entrypoint.py"]
