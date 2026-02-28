# Custom vLLM image with PoC v2 endpoints
FROM ghcr.io/gonka-ai/vllm:v0.9.1-poc-v2-post6

WORKDIR /app

# Install node worker dependencies using vLLM's Python
RUN /usr/bin/python3.12 -m pip install --no-cache-dir \
    fastapi>=0.109.0 \
    "uvicorn[standard]>=0.27.0" \
    pydantic>=2.5.0 \
    pydantic-settings>=2.1.0 \
    httpx>=0.26.0

# Copy node worker code
COPY config.py models.py vllm_client.py worker.py artifact_buffer.py main.py /app/
COPY startup.sh /app/
RUN chmod +x /app/startup.sh

# Python
ENV PYTHONUNBUFFERED=1

# PoC v2 model settings
ENV MODEL_NAME=Qwen/Qwen3-235B-A22B-Instruct-2507-FP8
ENV K_DIM=12
ENV SEQ_LEN=1024

# vLLM settings
ENV VLLM_PORT=8000
ENV VLLM_HOST=127.0.0.1
ENV VLLM_RPC_TIMEOUT=120000

# Node worker settings
ENV API_KEY=""
ENV PORT=9000
ENV LOG_LEVEL=INFO

# NCCL settings for multi-GPU
ENV NCCL_NVLS_ENABLE=0
ENV NCCL_P2P_DISABLE=1
ENV NCCL_IB_DISABLE=1
ENV NCCL_SHM_DISABLE=0
ENV NCCL_DEBUG=WARN

# HuggingFace cache — model downloaded on first run, cached on volume
ENV HF_HOME=/root/.cache/huggingface
ENV VLLM_USE_V1=0

# Clear vLLM default entrypoint
ENTRYPOINT []

# Expose vLLM + node worker ports
EXPOSE 8000 9000

CMD ["/app/startup.sh"]
