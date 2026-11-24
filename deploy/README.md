# Adaptive Serving Deployment

The service loads three immutable artifacts from the private Hugging Face model repository:

- `v4_corrected/v1/best_model.pt`
- `serving/corrected_v4_2_exact_v1/`
- `serving/corrected_v4_2_preprocessing_v1/`

It does not read Google Drive, Gold data, or the locked Test protocol.

Build from the project root:

```bash
docker build -f deploy/Dockerfile -t adaptive-recsys:latest .
```

Run one process on one GPU:

```bash
docker run --rm --gpus all \
  -p 8000:8000 \
  -e HF_TOKEN="$HF_TOKEN" \
  -v recsys-hf-cache:/tmp/recsys-artifacts \
  adaptive-recsys:latest
```

Selected initial settings:

```text
RECSYS_MAX_BATCH_SIZE=8
RECSYS_BATCH_WAIT_MS=2
```

Override them with environment variables only after a controlled load test. Keep one Uvicorn worker per GPU because each worker would otherwise duplicate the checkpoint and catalog in GPU memory. Scale horizontally with GPU replicas.

Health check:

```bash
curl http://localhost:8000/health
```

The first container start downloads the private artifacts. Reusing the mounted cache avoids downloading the multi-gigabyte checkpoint on every restart.
