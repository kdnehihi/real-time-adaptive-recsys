FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-serving.txt /app/requirements-serving.txt
RUN python3 -m pip install --no-cache-dir -r /app/requirements-serving.txt

COPY recommender /app/recommender
COPY scripts/serve_adaptive_api.py /app/scripts/serve_adaptive_api.py

EXPOSE 8000
CMD ["python3", "/app/scripts/serve_adaptive_api.py"]
