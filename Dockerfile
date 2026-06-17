FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV NCCL_DEBUG=WARN
ENV NCCL_IB_DISABLE=1
ENV NCCL_P2P_DISABLE=0
ENV OMP_NUM_THREADS=1
ENV HF_HOME=/workspace/.cache/huggingface
ENV TOKENIZERS_PARALLELISM=false
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /workspace

COPY requirements.txt constraints.txt ./
RUN pip install --no-cache-dir -r requirements.txt -c constraints.txt

COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir -e . --no-deps

COPY configs ./configs
COPY scripts ./scripts

CMD ["bash"]
