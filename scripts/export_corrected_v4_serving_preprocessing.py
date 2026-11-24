from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch

from recommender.models.adaptive.checkpoint import (
    ITEM_NUMERIC,
    SEQUENCE_COLUMNS,
    USER_NUMERIC,
    build_vocabs,
    prepare_train_item_vocab_frame,
    prepare_train_vocab_frame,
    save_json,
)
from recommender.serving.preprocessing import ServingUserEncoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export train-fitted V4 serving preprocessing.")
    parser.add_argument("--adaptive-root", type=Path, required=True)
    parser.add_argument("--base-gold-root", type=Path, required=True)
    parser.add_argument("--corrected-v4-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def ordered_values(index: dict[object, int]) -> list[object]:
    return [value for value, _ in sorted(index.items(), key=lambda pair: pair[1])]


def json_value(value: object) -> object:
    """Convert numpy scalar vocabulary values to portable JSON scalars."""
    return value.item() if isinstance(value, np.generic) else value


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(args.corrected_v4_dir / "best_model.pt", map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    users = prepare_train_vocab_frame(args.adaptive_root, config.get("max_train_examples"))
    items = prepare_train_item_vocab_frame(args.adaptive_root, config.get("max_item_pool_examples"))
    vocabs = build_vocabs(args.base_gold_root, users, items)
    expected = {key: int(value) for key, value in checkpoint.get("vocab_sizes", {}).items()}
    actual = {key: vocab.size for key, vocab in vocabs.items()}
    if actual != expected:
        raise RuntimeError(f"Exported vocab sizes differ from checkpoint: {actual} != {expected}")
    for feature in ("user_id", "video_id"):
        np.save(
            args.output_dir / f"vocab_{feature}.npy",
            np.asarray(ordered_values(vocabs[feature].index), dtype="int64"),
            allow_pickle=False,
        )
    for feature in ("user_active_degree", "video_type", "upload_type", "event_type"):
        rows = [
            {"value": json_value(value), "index": int(index)}
            for value, index in sorted(vocabs[feature].index.items(), key=lambda pair: pair[1])
        ]
        (args.output_dir / f"vocab_{feature}.json").write_text(json.dumps(rows, indent=2))
    shutil.copy2(args.base_gold_root / "transforms/numeric_stats.json", args.output_dir / "numeric_stats.json")
    schema = {
        "user_numeric": list(USER_NUMERIC),
        "item_numeric": list(ITEM_NUMERIC),
        "sequence_columns": list(SEQUENCE_COLUMNS),
        "sequence_max_len": int(config["sequence_max_len"]),
        "point_in_time_rule": "every recent_time_ms must be strictly less than current_time_ms",
    }
    (args.output_dir / "request_schema.json").write_text(json.dumps(schema, indent=2))
    files = {}
    for path in sorted(args.output_dir.iterdir()):
        if path.name != "manifest.json":
            files[path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    manifest = {
        "bundle": "corrected_v4_2_serving_preprocessing_v1",
        "checkpoint": "v4_corrected/v1/best_model.pt",
        "vocab_sizes": actual,
        "files": files,
        "gold_data_required_at_runtime": False,
        "locked_test_used": False,
    }
    save_json(args.output_dir / "manifest.json", manifest)
    encoder = ServingUserEncoder(
        args.output_dir, args.corrected_v4_dir / "best_model.pt", torch.device("cpu")
    )
    smoke = pd.DataFrame(
        {
            "user_id": [int(ordered_values(vocabs["user_id"].index)[0])],
            "current_time_ms": [1],
            "recent_video_ids": [[]],
            "recent_target_classes": [[]],
            "recent_engagement_strengths": [[]],
            "recent_long_views": [[]],
            "recent_likes": [[]],
            "recent_hates": [[]],
            "recent_clicks": [[]],
            "recent_time_ms": [[]],
        }
    )
    vector = encoder.encode(smoke)
    if vector.shape != (1, int(config["retrieval_dim"])) or not np.isfinite(vector).all():
        raise RuntimeError("Standalone serving user encoder smoke test failed")
    manifest["standalone_encoder_smoke"] = {
        "passed": True,
        "shape": list(vector.shape),
        "l2_norm": float(np.linalg.norm(vector[0])),
    }
    save_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
