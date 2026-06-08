FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYENV_ROOT=/root/.pyenv
ENV PATH="${PYENV_ROOT}/shims:${PYENV_ROOT}/bin:/root/.local/bin:${PATH}"
ENV HF_HOME=/workspace/.cache/huggingface

ARG PYTHON_VERSION=3.12.0

RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN curl https://pyenv.run | bash \
    && pyenv install "${PYTHON_VERSION}" \
    && pyenv global "${PYTHON_VERSION}"

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /workspace/qwen_clt_circuit_baseline

COPY requirements.txt constraints.txt pyproject.toml ./
COPY src ./src

RUN uv pip install --system --upgrade pip \
    && uv pip install --system -r requirements.txt -c constraints.txt \
    && uv pip install --system -e .

COPY configs ./configs
COPY scripts ./scripts
COPY README.md EXPERIMENT_PIPELINE.md EXPERIMENT_PLAN.md ./

CMD ["/bin/bash"]
