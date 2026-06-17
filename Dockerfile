# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# Cross-Layer Transcoder (CLT) — training image
#
# Base: pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel
#   • PyTorch 2.4.1  • CUDA 12.1  • cuDNN 9  • NCCL 2.21 (compatible)
#   • Python 3.11
#
# Supports both single-GPU and 2-GPU DDP (torchrun) out of the box.
# ---------------------------------------------------------------------------
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel

# ── system ──────────────────────────────────────────────────────────────────
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ── NCCL / runtime env ──────────────────────────────────────────────────────
# Single-node, no InfiniBand. P2P (NVLink/PCIe) enabled.
ENV NCCL_DEBUG=WARN
ENV NCCL_IB_DISABLE=1
ENV NCCL_P2P_DISABLE=0
# Prevent CPU thread oversubscription when DDP spawns multiple workers
ENV OMP_NUM_THREADS=1
# HuggingFace cache inside the workspace volume
ENV HF_HOME=/workspace/.cache/huggingface
# Avoid HF tokenizer fork warnings
ENV TOKENIZERS_PARALLELISM=false
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /workspace

# ── Python deps ──────────────────────────────────────────────────────────────
# torch is already provided by the base image (2.4.1, CUDA 12.1). requirements.txt
# pins `torch>=2.1`, which is already satisfied, so pip will NOT reinstall torch
# and the CUDA-matched build from the base image is preserved.
COPY requirements.txt constraints.txt ./
RUN pip install --no-cache-dir -r requirements.txt -c constraints.txt

# ── package install ──────────────────────────────────────────────────────────
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir -e . --no-deps

# ── project assets ───────────────────────────────────────────────────────────
COPY configs ./configs
COPY scripts ./scripts

CMD ["bash"]
