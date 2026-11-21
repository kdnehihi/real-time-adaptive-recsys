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
    deterministic_user_split,
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


def parse_floats(raw: str) -> tuple[float, ...]:
    return tuple(sorted({float(value.strip()) for value in raw.split(",") if value.strip()}))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate author affinity over corrected V4 and compare legacy V4.2.")
    parser.add_argument("--adaptive-root", type=Path, required=True)
    parser.add_argument("--base-gold-root", type=Path, required=True)
    parser.add_argument("--corrected-root", type=Path, required=True)
    parser.add_argument("--corrected-v4-dir", type=Path, required=True)
    parser.add_argument("--legacy-v4-dir", type=Path, required=True)
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--author-weights", default="0,0.05,0.1,0.15,0.2,0.25,0.3,0.4")
    parser.add_argument("--popularity-weights", default="0,0.01,0.02,0.03")
    parser.add_argument("--legacy-author-weight", type=float, default=0.2)
    parser.add_argument("--legacy-popularity-weight", type=float, default=0.02)
    parser.add_argument("--top-ks", default="10,50,100,500,1000")
    parser.add_argument("--calibration-user-fraction", type=float, default=0.35)
    parser.add_argument("--recency-tau", type=float, default=3.0)
    parser.add_argument("--query-batch-size", type=int, default=16)
    parser.add_argument("--item-batch-size", type=int, default=8192)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def load_video_author_mapping(static_path: Path) -> dict[int, int]:
    """Load immutable item authors shard-by-shard to tolerate Drive FUSE."""
    mapping: dict[int, int] = {}
    files = parquet_files(static_path)
    for index, path in enumerate(files, start=1):
        part = read_shard(path, list(STATIC_COLUMNS)).dropna(subset=["video_id", "author_id"])
        mapping.update(zip(part["video_id"].astype("int64"), part["author_id"].astype("int64")))
        if index % 16 == 0 or index == len(files):
            print(f"author metadata: {index}/{len(files)} shards; {len(mapping):,} videos", flush=True)
    return mapping


def candidate_authors(item_ids: np.ndarray, mapping: dict[int, int]) -> torch.Tensor:
    return torch.as_tensor(
        [int(mapping.get(int(video_id), 0) or 0) for video_id in item_ids], dtype=torch.long
    )


def normalized_popularity(candidate_pool: pd.DataFrame) -> np.ndarray:
    values = np.log1p(np.maximum(candidate_pool["item_hist_events"].to_numpy("float32"), 0.0))
    return np.nan_to_num((values - values.mean()) / (values.std() + 1e-6)).astype("float32")


def cold_warm_summary(frame: pd.DataFrame, metrics: list[str]) -> dict[str, dict[str, float | int]]:
    return {
        ("warm" if bool(value) else "cold"): user_macro_summary(part, metrics)
        for value, part in frame.groupby("target_warm")
    }


def acceptance_decision(
    selected_author_weight: float,
    author_delta: dict[str, dict[str, float]],
    promotion_delta: dict[str, dict[str, float]],
    selected_segments: dict[str, dict[str, float | int]],
    matched_segments: dict[str, dict[str, float | int]],
) -> dict[str, bool]:
    """Apply explicit statistical and cold-item non-regression gates."""
    author_gate = bool(
        selected_author_weight > 0
        and author_delta["hitrate@1000"]["delta"] > 0
        and author_delta["hitrate@1000"]["ci95_low"] > 0
    )
    promotion_gate = bool(
        promotion_delta["hitrate@1000"]["delta"] > 0
        and promotion_delta["hitrate@1000"]["ci95_low"] > 0
    )
    cold_gate = bool(
        selected_segments.get("cold", {}).get("hitrate@1000", 0.0)
        >= matched_segments.get("cold", {}).get("hitrate@1000", 0.0)
    )
    return {
        "author_affinity_gate": author_gate,
        "legacy_promotion_gate": promotion_gate,
        "cold_item_non_regression_gate": cold_gate,
        "accepted": author_gate and promotion_gate and cold_gate,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)
    top_ks = tuple(sorted({int(value) for value in args.top_ks.split(",")}))
    metrics = metric_columns(top_ks)
    configs = tuple(
        (author, popularity)
        for author in parse_floats(args.author_weights)
        for popularity in parse_floats(args.popularity_weights)
    )
    print("device/calibration configs:", device, len(configs), flush=True)

    legacy_queries, protocol = load_protocol_queries(
        args.adaptive_root, args.protocol_dir, "validation", None
    )
    corrected = corrected_queries(legacy_queries, args.corrected_root)
    candidate_pool = build_safe_candidate_pool(args.adaptive_root, "validation")
    video_to_author = load_video_author_mapping(args.base_gold_root / "item_features/static")
    popularity = normalized_popularity(candidate_pool)
    history = load_event_history(
        args.adaptive_root,
        "validation",
        set(protocol["user_id"].astype("int64")),
        int(protocol["query_time_ms"].max()) + 1,
    )
    calibration_mask = deterministic_user_split(
        protocol["user_id"], args.calibration_user_fraction, args.seed
    )
    calibration_indices = np.flatnonzero(calibration_mask)
    holdout_indices = np.flatnonzero(~calibration_mask)
    if set(protocol.iloc[calibration_indices]["user_id"]) & set(protocol.iloc[holdout_indices]["user_id"]):
        raise AssertionError("User leakage between calibration and holdout")
    print(
        "calibration/holdout queries/users:",
        len(calibration_indices), protocol.iloc[calibration_indices]["user_id"].nunique(),
        len(holdout_indices), protocol.iloc[holdout_indices]["user_id"].nunique(),
        flush=True,
    )

    print("encoding corrected V4...", flush=True)
    corrected_artifacts = load_artifacts(
        args.adaptive_root, args.base_gold_root, args.corrected_v4_dir, device=device
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
    corrected_recent_authors = encode_raw_recent_authors(
        corrected, video_to_author, int(corrected_artifacts.v4_cfg["sequence_max_len"])
    )
    authors = candidate_authors(item_ids, video_to_author)
    eligible = candidate_pool["eligible_from_ms"].to_numpy("int64")

    calibration_scores = score_configs(
        calibration_indices, configs, protocol, corrected_users, corrected_items,
        corrected_recent_authors, authors, popularity, item_ids, eligible, history,
        args.query_batch_size, args.recency_tau, device,
    )
    calibration_rows: list[dict[str, object]] = []
    for config, (ranks, repeats) in calibration_scores.items():
        frame = ranks_to_frame(protocol, calibration_indices, ranks, repeats, top_ks)
        calibration_rows.append(
            {
                "config": config_name(config),
                "author_weight": config[0],
                "popularity_weight": config[1],
                **user_macro_summary(frame, metrics),
            }
        )
    calibration = pd.DataFrame(calibration_rows).sort_values(
        ["hitrate@1000", "ndcg@100", "ndcg@1000"], ascending=False
    ).reset_index(drop=True)
    calibration.to_csv(args.output_dir / "calibration_grid.csv", index=False)
    selected = (
        float(calibration.iloc[0]["author_weight"]),
        float(calibration.iloc[0]["popularity_weight"]),
    )
    print("selected corrected config:", selected, flush=True)

    corrected_holdout_configs = tuple(
        dict.fromkeys(((0.0, 0.0), (0.0, selected[1]), selected))
    )
    corrected_scores = score_configs(
        holdout_indices, corrected_holdout_configs, protocol, corrected_users, corrected_items,
        corrected_recent_authors, authors, popularity, item_ids, eligible, history,
        args.query_batch_size, args.recency_tau, device,
    )
    del corrected_artifacts, corrected_items, corrected_users
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("encoding legacy V4.2 baseline...", flush=True)
    legacy_artifacts = load_artifacts(
        args.adaptive_root, args.base_gold_root, args.legacy_v4_dir, device=device
    )
    legacy_item_ids, legacy_items = encode_items(
        legacy_artifacts.v4,
        candidate_pool,
        legacy_queries.iloc[0].to_dict(),
        legacy_artifacts,
        args.item_batch_size,
    )
    if not np.array_equal(item_ids, legacy_item_ids):
        raise AssertionError("Legacy and corrected candidate item order differs")
    legacy_users = encode_users(
        legacy_artifacts.v4, legacy_queries, legacy_artifacts, batch_size=512
    )
    legacy_recent_authors = encode_raw_recent_authors(
        legacy_queries, video_to_author, int(legacy_artifacts.v4_cfg["sequence_max_len"])
    )
    legacy_config = (args.legacy_author_weight, args.legacy_popularity_weight)
    legacy_scores = score_configs(
        holdout_indices, (legacy_config,), protocol, legacy_users, legacy_items,
        legacy_recent_authors, authors, popularity, item_ids, eligible, history,
        args.query_batch_size, args.recency_tau, device,
    )

    frames: dict[str, pd.DataFrame] = {}
    for config, (ranks, repeats) in corrected_scores.items():
        name = f"corrected_{config_name(config)}"
        frame = ranks_to_frame(protocol, holdout_indices, ranks, repeats, top_ks)
        frame.insert(0, "variant", name)
        frames[name] = frame
    legacy_name = f"legacy_{config_name(legacy_config)}"
    legacy_ranks, legacy_repeats = legacy_scores[legacy_config]
    legacy_frame = ranks_to_frame(
        protocol, holdout_indices, legacy_ranks, legacy_repeats, top_ks
    )
    legacy_frame.insert(0, "variant", legacy_name)
    frames[legacy_name] = legacy_frame

    summary_rows = [
        {"variant": name, **user_macro_summary(frame, metrics)}
        for name, frame in frames.items()
    ]
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.output_dir / "holdout_summary.csv", index=False)
    segments = []
    for name, frame in frames.items():
        segment = segment_summary(frame, metrics)
        segment.insert(0, "variant", name)
        segments.append(segment)
        frame.to_parquet(args.output_dir / f"per_query_{name}.parquet", index=False)
    pd.concat(segments, ignore_index=True).to_csv(
        args.output_dir / "holdout_segments.csv", index=False
    )

    selected_name = f"corrected_{config_name(selected)}"
    matched_name = f"corrected_{config_name((0.0, selected[1]))}"
    comparisons = {
        "selected_vs_corrected_matched_popularity": paired_bootstrap(
            frames[selected_name], frames[matched_name], metrics,
            args.bootstrap_samples, args.seed,
        ),
        "selected_corrected_vs_legacy_v4_2": paired_bootstrap(
            frames[selected_name], frames[legacy_name], metrics,
            args.bootstrap_samples, args.seed + 1,
        ),
    }
    segment_metrics = {
        name: cold_warm_summary(frame, metrics) for name, frame in frames.items()
    }
    gates = acceptance_decision(
        selected[0],
        comparisons["selected_vs_corrected_matched_popularity"],
        comparisons["selected_corrected_vs_legacy_v4_2"],
        segment_metrics[selected_name],
        segment_metrics[matched_name],
    )
    report = {
        "protocol": "adaptive_eval_v2_corrected_v4_author_affinity",
        "selection_policy": "user-disjoint calibration; maximize HR@1000 then NDCG@100/NDCG@1000",
        "calibration_queries": int(len(calibration_indices)),
        "holdout_queries": int(len(holdout_indices)),
        "candidate_items": int(len(item_ids)),
        "selected": {"author_weight": selected[0], "popularity_weight": selected[1]},
        "legacy_v4_2": {
            "author_weight": legacy_config[0],
            "popularity_weight": legacy_config[1],
        },
        "recency_tau": args.recency_tau,
        "holdout_summary": {
            row["variant"]: {key: value for key, value in row.items() if key != "variant"}
            for row in summary_rows
        },
        "cold_warm_summary": segment_metrics,
        "paired_user_bootstrap": comparisons,
        "acceptance_gates": gates,
        "acceptance_rule": "positive paired HR@1000 CI vs matched corrected V4 and legacy V4.2, plus cold HR@1000 non-regression",
        "accepted": gates["accepted"],
    }
    save_json(args.output_dir / "corrected_v4_author_affinity_evaluation.json", report)
    print(summary.to_string(index=False), flush=True)
    print(json.dumps({"selected": report["selected"], "gates": gates}, indent=2), flush=True)


if __name__ == "__main__":
    main()
