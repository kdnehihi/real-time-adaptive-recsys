from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import uvicorn
from huggingface_hub import snapshot_download

from recommender.serving.api import create_app
from recommender.serving.runtime import AdaptiveServingRuntime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve Corrected V4.2 from private Hugging Face artifacts.")
    parser.add_argument("--repo-id", default=os.getenv("HF_REPO_ID", "khoa1505/recsys-artifacts"))
    parser.add_argument("--cache-dir", type=Path, default=Path(os.getenv("RECSYS_CACHE_DIR", "/tmp/recsys-artifacts")))
    parser.add_argument("--host", default=os.getenv("RECSYS_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("RECSYS_PORT", "8000")))
    parser.add_argument("--device", default=os.getenv("RECSYS_DEVICE", "cuda"))
    parser.add_argument("--max-batch-size", type=int, default=int(os.getenv("RECSYS_MAX_BATCH_SIZE", "8")))
    parser.add_argument("--batch-wait-ms", type=float, default=float(os.getenv("RECSYS_BATCH_WAIT_MS", "2")))
    return parser.parse_args()


def build_app(args: argparse.Namespace):
    """Download/cache immutable artifacts and create one single-GPU API process."""
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN must contain a read token for the private artifact repository")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    root = Path(
        snapshot_download(
            args.repo_id,
            repo_type="model",
            token=token,
            local_dir=args.cache_dir,
            allow_patterns=[
                "v4_corrected/v1/best_model.pt",
                "serving/corrected_v4_2_exact_v1/**",
                "serving/corrected_v4_2_preprocessing_v1/**",
            ],
        )
    )
    runtime = AdaptiveServingRuntime.from_artifacts(
        root / "serving/corrected_v4_2_exact_v1",
        root / "serving/corrected_v4_2_preprocessing_v1",
        root / "v4_corrected/v1/best_model.pt",
        torch.device(args.device),
    )
    return create_app(
        runtime,
        max_batch_size=args.max_batch_size,
        batch_wait_ms=args.batch_wait_ms,
    )


def main() -> None:
    args = parse_args()
    app = build_app(args)
    # One process owns one GPU/model copy. Scale with replicas, not Uvicorn workers.
    uvicorn.run(app, host=args.host, port=args.port, workers=1, log_level="info")


if __name__ == "__main__":
    main()
