FROM runpod/base:0.6.3-cuda12.2.0

ENV PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    HF_HOME=/runpod-volume/huggingface \
    TRANSFORMERS_CACHE=/runpod-volume/huggingface \
    DIFFUSERS_CACHE=/runpod-volume/huggingface

WORKDIR /app

COPY requirements.txt .
RUN python3.11 -m pip install --no-cache-dir --upgrade pip && \
    python3.11 -m pip install --no-cache-dir hf-transfer && \
    python3.11 -m pip install --no-cache-dir -r requirements.txt

COPY handler.py .

CMD ["python3.11", "-u", "handler.py"]
