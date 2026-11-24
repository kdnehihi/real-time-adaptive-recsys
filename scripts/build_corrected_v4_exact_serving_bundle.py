from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from recommender.models.adaptive.checkpoint import encode_items, load_artifacts, parquet_files, save_json, select_device
from scripts.audit_adaptive_content_readiness import STATIC_COLUMNS
from scripts.evaluate_adaptive_protocol_v2 import build_safe_candidate_pool, read_shard


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the fixed Corrected V4.2 exact serving catalog.")
    parser.add_argument("--adaptive-root", type=Path, required=True)
    parser.add_argument("--base-gold-root", type=Path, required=True)
    parser.add_argument("--corrected-v4-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--item-batch-size", type=int, default=8192)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def video_authors(static_path: Path) -> dict[int, int]:
    """Load static video-to-author metadata using bounded shard reads."""
    output: dict[int, int] = {}
    files = parquet_files(static_path)
    for index, path in enumerate(files, start=1):
        part = read_shard(path, list(STATIC_COLUMNS)).dropna(subset=["video_id", "author_id"])
        output.update(zip(part["video_id"].astype("int64"), part["author_id"].astype("int64")))
        print(f"author metadata: {index}/{len(files)}", flush=True)
    return output


def normalized_popularity(candidate_pool) -> np.ndarray:
    values = np.log1p(np.maximum(candidate_pool["item_hist_events"].to_numpy("float32"), 0.0))
    return np.nan_to_num((values - values.mean()) / (values.std() + 1e-6)).astype("float32")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)
    candidates = build_safe_candidate_pool(args.adaptive_root, "validation")
    artifacts = load_artifacts(args.adaptive_root, args.base_gold_root, args.corrected_v4_dir, device=device)
    # Item encoding ignores user inputs; provide the minimum valid stub required
    # by the shared batch encoder instead of reading an unrelated query row.
    source = {"user_id": 0}
    item_ids, embeddings = encode_items(
        artifacts.v4, candidates, source, artifacts, args.item_batch_size
    )
    authors_by_video = video_authors(args.base_gold_root / "item_features/static")
    arrays = {
        "item_ids.npy": item_ids.astype("int64"),
        "item_embeddings_fp16.npy": embeddings.astype("float16"),
        "author_ids.npy": np.asarray([authors_by_video.get(int(item), 0) for item in item_ids], dtype="int64"),
        "popularity.npy": normalized_popularity(candidates),
        "eligible_from_ms.npy": candidates["eligible_from_ms"].to_numpy("int64"),
    }
    for name, values in arrays.items():
        np.save(args.output_dir / name, values, allow_pickle=False)
    files = {
        name: {"bytes": (args.output_dir / name).stat().st_size, "sha256": sha256(args.output_dir / name)}
        for name in arrays
    }
    manifest = {
        "bundle": "corrected_v4_2_exact_serving_v1",
        "candidate_items": len(item_ids),
        "embedding_dim": int(embeddings.shape[1]),
        "embedding_storage_dtype": "float16",
        "serving_compute_dtype": "float32",
        "model_checkpoint": "v4_corrected/v1/best_model.pt",
        "policy": {"author_weight": 0.2, "popularity_weight": 0.02, "recency_tau": 3.0},
        "files": files,
        "locked_test_used": False,
    }
    save_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
