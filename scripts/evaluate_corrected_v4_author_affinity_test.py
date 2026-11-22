from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch

from recommender.models.adaptive.author_branch import encode_raw_recent_authors
from recommender.models.adaptive.checkpoint import (
    encode_items,
    encode_users,
    load_artifacts,
    parquet_files,
    save_json,
    select_device,
)
from scripts.audit_adaptive_content_readiness import STATIC_COLUMNS
from scripts.compare_corrected_adaptive_v4 import corrected_queries
from scripts.evaluate_adaptive_author_affinity import (
    config_name,
    paired_bootstrap,
    ranks_to_frame,
    score_configs,
)
from scripts.evaluate_adaptive_protocol_v2 import (
    build_safe_candidate_pool,
    load_event_history,
    load_protocol_queries,
    metric_columns,
    read_shard,
    segment_summary,
    user_macro_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Final locked-test comparison for fixed Corrected V4 author affinity."
    )
    parser.add_argument("--adaptive-root", type=Path, required=True)
    parser.add_argument("--train-source", type=Path, required=True)
    parser.add_argument("--base-gold-root", type=Path, required=True)
    parser.add_argument("--corrected-root", type=Path, required=True)
    parser.add_argument("--corrected-v4-dir", type=Path, required=True)
    parser.add_argument("--legacy-v4-dir", type=Path, required=True)
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--author-weight", type=float, default=0.2)
    parser.add_argument("--popularity-weight", type=float, default=0.02)
    parser.add_argument("--recency-tau", type=float, default=3.0)
    parser.add_argument("--top-ks", default="10,50,100,500,1000")
    parser.add_argument("--query-batch-size", type=int, default=16)
    parser.add_argument("--item-batch-size", type=int, default=8192)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def load_video_author_mapping(static_path: Path) -> dict[int, int]:
    """Load immutable item authors shard-by-shard with deterministic handle closure."""
    mapping: dict[int, int] = {}
    files = parquet_files(static_path)
    for index, path in enumerate(files, start=1):
        part = read_shard(path, list(STATIC_COLUMNS)).dropna(subset=["video_id", "author_id"])
        mapping.update(zip(part["video_id"].astype("int64"), part["author_id"].astype("int64")))
        if index % 16 == 0 or index == len(files):
            print(f"author metadata: {index}/{len(files)} shards; {len(mapping):,} videos", flush=True)
    return mapping


def normalized_popularity(candidate_pool: pd.DataFrame) -> np.ndarray:
    values = np.log1p(np.maximum(candidate_pool["item_hist_events"].to_numpy("float32"), 0.0))
    return np.nan_to_num((values - values.mean()) / (values.std() + 1e-6)).astype("float32")


def cold_warm_summary(frame: pd.DataFrame, metrics: list[str]) -> dict[str, dict[str, float | int]]:
    return {
        ("warm" if bool(value) else "cold"): user_macro_summary(part, metrics)
        for value, part in frame.groupby("target_warm")
    }


def final_acceptance(
    author_delta: dict[str, dict[str, float]],
    legacy_delta: dict[str, dict[str, float]],
    selected_segments: dict[str, dict[str, float | int]],
    matched_segments: dict[str, dict[str, float | int]],
    legacy_segments: dict[str, dict[str, float | int]],
) -> dict[str, bool]:
    """Apply predeclared primary and cold-item promotion gates on locked test."""
    matched_gate = bool(author_delta["hitrate@1000"]["ci95_low"] > 0)
    legacy_gate = bool(legacy_delta["hitrate@1000"]["ci95_low"] > 0)
    selected_cold = selected_segments.get("cold", {}).get("hitrate@1000", 0.0)
    cold_gate = bool(
        selected_cold >= matched_segments.get("cold", {}).get("hitrate@1000", 0.0)
        and selected_cold >= legacy_segments.get("cold", {}).get("hitrate@1000", 0.0)
    )
    return {
        "author_affinity_test_gate": matched_gate,
        "legacy_promotion_test_gate": legacy_gate,
        "cold_item_test_non_regression_gate": cold_gate,
        "accepted": matched_gate and legacy_gate and cold_gate,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)
    top_ks = tuple(sorted({int(value) for value in args.top_ks.split(",")}))
    metrics = metric_columns(top_ks)
    selected_config = (float(args.author_weight), float(args.popularity_weight))
    matched_config = (0.0, float(args.popularity_weight))
    raw_config = (0.0, 0.0)
    print("device/fixed config:", device, selected_config, flush=True)

    legacy_queries, protocol = load_protocol_queries(
        args.adaptive_root, args.protocol_dir, "test", None
    )
    corrected = corrected_queries(legacy_queries, args.corrected_root, split="test")
    candidate_pool = build_safe_candidate_pool(
        args.adaptive_root, "test", train_source=args.train_source
    )
    video_to_author = load_video_author_mapping(args.base_gold_root / "item_features/static")
    popularity = normalized_popularity(candidate_pool)
    history = load_event_history(
        args.adaptive_root,
        "test",
        set(protocol["user_id"].astype("int64")),
        int(protocol["query_time_ms"].max()) + 1,
        train_source=args.train_source,
    )
    indices = np.arange(len(protocol), dtype="int64")
    print("locked test queries/users:", len(protocol), protocol["user_id"].nunique(), flush=True)

    print("encoding Corrected V4...", flush=True)
    corrected_artifacts = load_artifacts(
        args.adaptive_root,
        args.base_gold_root,
        args.corrected_v4_dir,
        device=device,
        train_source=args.train_source,
    )
    item_ids, corrected_items = encode_items(
        corrected_artifacts.v4,
        candidate_pool,
        corrected.iloc[0].to_dict(),
        corrected_artifacts,
        args.item_batch_size,
    )
    corrected_users = encode_users(
        corrected_artifacts.v4, corrected, corrected_artifacts, batch_size=512
    )
    recent_authors = encode_raw_recent_authors(
        corrected, video_to_author, int(corrected_artifacts.v4_cfg["sequence_max_len"])
    )
    candidate_authors = torch.as_tensor(
        [int(video_to_author.get(int(video_id), 0) or 0) for video_id in item_ids],
        dtype=torch.long,
    )
    eligible = candidate_pool["eligible_from_ms"].to_numpy("int64")
    corrected_configs = tuple(dict.fromkeys((raw_config, matched_config, selected_config)))
    corrected_scores = score_configs(
        indices,
        corrected_configs,
        protocol,
        corrected_users,
        corrected_items,
        recent_authors,
        candidate_authors,
        popularity,
        item_ids,
        eligible,
        history,
        args.query_batch_size,
        args.recency_tau,
        device,
    )
    del corrected_artifacts, corrected_items, corrected_users
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("encoding Legacy V4.2...", flush=True)
    legacy_artifacts = load_artifacts(
        args.adaptive_root,
        args.base_gold_root,
        args.legacy_v4_dir,
        device=device,
        train_source=args.train_source,
    )
    legacy_ids, legacy_items = encode_items(
        legacy_artifacts.v4,
        candidate_pool,
        legacy_queries.iloc[0].to_dict(),
        legacy_artifacts,
        args.item_batch_size,
    )
    if not np.array_equal(item_ids, legacy_ids):
        raise AssertionError("Legacy and corrected candidate item order differs")
    legacy_users = encode_users(
        legacy_artifacts.v4, legacy_queries, legacy_artifacts, batch_size=512
    )
    legacy_recent_authors = encode_raw_recent_authors(
        legacy_queries, video_to_author, int(legacy_artifacts.v4_cfg["sequence_max_len"])
    )
    legacy_scores = score_configs(
        indices,
        (selected_config,),
        protocol,
        legacy_users,
        legacy_items,
        legacy_recent_authors,
        candidate_authors,
        popularity,
        item_ids,
        eligible,
        history,
        args.query_batch_size,
        args.recency_tau,
        device,
    )

    frames: dict[str, pd.DataFrame] = {}
    for config, (ranks, repeats) in corrected_scores.items():
        name = f"corrected_{config_name(config)}"
        frame = ranks_to_frame(protocol, indices, ranks, repeats, top_ks)
        frame.insert(0, "variant", name)
        frames[name] = frame
    legacy_name = f"legacy_{config_name(selected_config)}"
    legacy_ranks, legacy_repeats = legacy_scores[selected_config]
    legacy_frame = ranks_to_frame(protocol, indices, legacy_ranks, legacy_repeats, top_ks)
    legacy_frame.insert(0, "variant", legacy_name)
    frames[legacy_name] = legacy_frame

    summary_rows = [
        {"variant": name, **user_macro_summary(frame, metrics)}
        for name, frame in frames.items()
    ]
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.output_dir / "test_summary.csv", index=False)
    segment_rows = []
    for name, frame in frames.items():
        segment = segment_summary(frame, metrics)
        segment.insert(0, "variant", name)
        segment_rows.append(segment)
        frame.to_parquet(args.output_dir / f"per_query_{name}.parquet", index=False)
    pd.concat(segment_rows, ignore_index=True).to_csv(
        args.output_dir / "test_segments.csv", index=False
    )

    selected_name = f"corrected_{config_name(selected_config)}"
    matched_name = f"corrected_{config_name(matched_config)}"
    comparisons = {
        "selected_vs_corrected_matched_popularity": paired_bootstrap(
            frames[selected_name], frames[matched_name], metrics, args.bootstrap_samples, args.seed
        ),
        "selected_corrected_vs_legacy_v4_2": paired_bootstrap(
            frames[selected_name], frames[legacy_name], metrics, args.bootstrap_samples, args.seed + 1
        ),
    }
    segments = {name: cold_warm_summary(frame, metrics) for name, frame in frames.items()}
    gates = final_acceptance(
        comparisons["selected_vs_corrected_matched_popularity"],
        comparisons["selected_corrected_vs_legacy_v4_2"],
        segments[selected_name],
        segments[matched_name],
        segments[legacy_name],
    )
    report = {
        "protocol": "adaptive_eval_v2_locked_test",
        "selection_policy": "weights frozen from Notebook 31 validation; no test tuning",
        "fixed_config": {
            "author_weight": selected_config[0],
            "popularity_weight": selected_config[1],
            "recency_tau": args.recency_tau,
        },
        "queries": int(len(protocol)),
        "users": int(protocol["user_id"].nunique()),
        "candidate_items": int(len(item_ids)),
        "test_summary": {
            row["variant"]: {key: value for key, value in row.items() if key != "variant"}
            for row in summary_rows
        },
        "cold_warm_summary": segments,
        "paired_user_bootstrap": comparisons,
        "acceptance_gates": gates,
        "acceptance_rule": "positive paired HR@1000 CI versus matched Corrected V4 and Legacy V4.2, plus cold HR@1000 non-regression versus both",
        "accepted": gates["accepted"],
    }
    save_json(args.output_dir / "locked_test_evaluation.json", report)
    print(summary.to_string(index=False), flush=True)
    print(json.dumps({"fixed_config": report["fixed_config"], "gates": gates}, indent=2), flush=True)


if __name__ == "__main__":
    main()
