from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch

from recommender.models.adaptive.author_branch import encode_raw_recent_authors
from recommender.models.adaptive.checkpoint import encode_items, encode_users, load_artifacts, save_json, select_device
from scripts.evaluate_adaptive_author_affinity import (
    deterministic_user_split,
    load_video_author_mapping,
    paired_bootstrap,
    ranks_to_frame,
    score_configs,
)
from scripts.evaluate_adaptive_protocol_v2 import (
    build_safe_candidate_pool,
    load_event_history,
    load_protocol_queries,
    make_static_ablation,
    metric_columns,
    user_macro_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay selected V4.2 in strict point-in-time order.")
    parser.add_argument("--adaptive-root", type=Path, required=True)
    parser.add_argument("--base-gold-root", type=Path, required=True)
    parser.add_argument("--v4-dir", type=Path, required=True)
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--author-weight", type=float, default=0.2)
    parser.add_argument("--popularity-weight", type=float, default=0.02)
    parser.add_argument("--recency-tau", type=float, default=3.0)
    parser.add_argument("--calibration-user-fraction", type=float, default=0.35)
    parser.add_argument("--top-ks", default="10,50,100,500,1000")
    parser.add_argument("--query-batch-size", type=int, default=16)
    parser.add_argument("--item-batch-size", type=int, default=8192)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def validate_replay_order(features: pd.DataFrame, protocol: pd.DataFrame) -> dict[str, int]:
    """Assert timestamp order and strict-prior sequence timestamps before scoring."""
    query_times = protocol["query_time_ms"].to_numpy("int64")
    if len(query_times) > 1 and np.any(query_times[1:] < query_times[:-1]):
        raise AssertionError("Replay queries are not in non-decreasing timestamp order")
    checked = 0
    violations = 0
    for current, times in zip(query_times, features["recent_time_ms"]):
        values = np.asarray(times if times is not None else [], dtype=np.int64)
        checked += len(values)
        violations += int(np.sum(values >= current))
    if violations:
        raise AssertionError(f"Point-in-time replay leakage in {violations} recent events")
    return {"queries": int(len(query_times)), "sequence_timestamps_checked": checked, "violations": 0}


def add_replay_segments(frame: pd.DataFrame, features: pd.DataFrame, indices: np.ndarray) -> pd.DataFrame:
    """Attach online-state and preference-shift diagnostics to per-query metrics."""
    output = frame.copy()
    selected = features.iloc[indices].reset_index(drop=True)
    def numeric_column(name: str) -> pd.Series:
        if name not in selected:
            return pd.Series(np.zeros(len(selected)), index=selected.index, dtype="float64")
        return pd.to_numeric(selected[name], errors="coerce").fillna(0.0)

    long_view_delta = numeric_column("session_vs_user_long_view_delta")
    watch_delta = numeric_column("session_vs_user_watch_ratio_delta")
    shift = long_view_delta.abs() + watch_delta.abs()
    positive = shift[shift > 0]
    threshold = float(positive.quantile(0.75)) if len(positive) else float("inf")
    output["preference_shift_score"] = shift.to_numpy("float32")
    output["preference_shift_bucket"] = np.where(
        shift >= threshold, "recent_shift", "stable_or_low_signal"
    )
    output["recent_event_count"] = pd.to_numeric(
        selected["recent_event_count"], errors="coerce"
    ).fillna(0).astype("int64")
    output["query_time_ms"] = selected["current_time_ms"].astype("int64")
    return output


def segment_table(frames: dict[str, pd.DataFrame], metrics: list[str]) -> pd.DataFrame:
    """Summarize replay quality across online-state and item segments."""
    rows: list[dict[str, object]] = []
    columns = (
        "target_warm",
        "target_repeated",
        "timeline_bucket",
        "session_history_bucket",
        "preference_shift_bucket",
    )
    for variant, frame in frames.items():
        for column in columns:
            for value, part in frame.groupby(column, dropna=False):
                rows.append(
                    {
                        "variant": variant,
                        "segment_type": column,
                        "segment": str(value),
                        **user_macro_summary(part, metrics),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    top_ks = tuple(sorted({int(value) for value in args.top_ks.split(",")}))
    metrics = metric_columns(top_ks)
    device = select_device(args.device)
    started = time.perf_counter()

    features, protocol = load_protocol_queries(
        args.adaptive_root, args.protocol_dir, "validation", None
    )
    order = np.argsort(protocol["query_time_ms"].to_numpy("int64"), kind="stable")
    features = features.iloc[order].reset_index(drop=True)
    protocol = protocol.iloc[order].reset_index(drop=True)
    replay_validation = validate_replay_order(features, protocol)

    holdout_mask = ~deterministic_user_split(
        protocol["user_id"], args.calibration_user_fraction, args.seed
    )
    holdout_indices = np.flatnonzero(holdout_mask)
    holdout_users = set(protocol.iloc[holdout_indices]["user_id"].astype("int64"))
    print("ordered holdout queries/users:", len(holdout_indices), len(holdout_users), flush=True)

    candidate_pool = build_safe_candidate_pool(args.adaptive_root, "validation")
    artifacts = load_artifacts(args.adaptive_root, args.base_gold_root, args.v4_dir, device=device)
    item_started = time.perf_counter()
    item_ids, item_embeddings = encode_items(
        artifacts.v4, candidate_pool, features.iloc[0].to_dict(), artifacts, args.item_batch_size
    )
    item_encoding_seconds = time.perf_counter() - item_started

    user_started = time.perf_counter()
    adaptive_users = encode_users(artifacts.v4, features, artifacts, batch_size=512)
    static_users = encode_users(
        artifacts.v4, make_static_ablation(features), artifacts, batch_size=512
    )
    user_encoding_seconds = time.perf_counter() - user_started

    video_to_author = load_video_author_mapping(args.base_gold_root / "item_features/static")
    recent_authors = encode_raw_recent_authors(
        features, video_to_author, int(artifacts.v4_cfg["sequence_max_len"])
    )
    candidate_authors = torch.as_tensor(
        [int(video_to_author.get(int(video), 0) or 0) for video in item_ids], dtype=torch.long
    )
    popularity = np.log1p(
        np.maximum(candidate_pool["item_hist_events"].to_numpy("float32"), 0.0)
    )
    popularity = np.nan_to_num(
        (popularity - popularity.mean()) / (popularity.std() + 1e-6)
    ).astype("float32")
    history = load_event_history(
        args.adaptive_root,
        "validation",
        holdout_users,
        int(protocol.iloc[holdout_indices]["query_time_ms"].max()) + 1,
    )
    eligible = candidate_pool["eligible_from_ms"].to_numpy("int64")

    scoring_started = time.perf_counter()
    static_result = score_configs(
        holdout_indices, ((0.0, 0.0),), protocol, static_users, item_embeddings,
        recent_authors, candidate_authors, popularity, item_ids, eligible, history,
        args.query_batch_size, args.recency_tau, device,
    )[(0.0, 0.0)]
    adaptive_configs = ((0.0, 0.0), (0.0, args.popularity_weight),
                        (args.author_weight, args.popularity_weight))
    adaptive_result = score_configs(
        holdout_indices, adaptive_configs, protocol, adaptive_users, item_embeddings,
        recent_authors, candidate_authors, popularity, item_ids, eligible, history,
        args.query_batch_size, args.recency_tau, device,
    )
    scoring_seconds = time.perf_counter() - scoring_started

    result_map = {
        "v4_static_ablation": static_result,
        "v4_adaptive": adaptive_result[(0.0, 0.0)],
        "v4_adaptive_popularity": adaptive_result[(0.0, args.popularity_weight)],
        "v4_2_author_affinity": adaptive_result[(args.author_weight, args.popularity_weight)],
    }
    frames: dict[str, pd.DataFrame] = {}
    summaries: list[dict[str, object]] = []
    for variant, (ranks, repeats) in result_map.items():
        frame = ranks_to_frame(protocol, holdout_indices, ranks, repeats, top_ks)
        frame = add_replay_segments(frame, features, holdout_indices)
        frame.insert(0, "variant", variant)
        frames[variant] = frame
        summaries.append({"variant": variant, **user_macro_summary(frame, metrics)})
    summary = pd.DataFrame(summaries)
    summary.to_csv(args.output_dir / "online_replay_summary.csv", index=False)
    segments = segment_table(frames, metrics)
    segments.to_csv(args.output_dir / "online_replay_segments.csv", index=False)

    comparisons = {
        "v4_adaptive_vs_static": paired_bootstrap(
            frames["v4_adaptive"], frames["v4_static_ablation"], metrics,
            args.bootstrap_samples, args.seed,
        ),
        "v4_2_vs_v4_adaptive": paired_bootstrap(
            frames["v4_2_author_affinity"], frames["v4_adaptive"], metrics,
            args.bootstrap_samples, args.seed + 1,
        ),
        "v4_2_vs_same_popularity_v4": paired_bootstrap(
            frames["v4_2_author_affinity"], frames["v4_adaptive_popularity"], metrics,
            args.bootstrap_samples, args.seed + 2,
        ),
    }

    selected = frames["v4_2_author_affinity"].copy()
    baseline = frames["v4_adaptive"].set_index("example_id")
    selected["v4_exact_rank"] = selected["example_id"].map(baseline["exact_rank"])
    selected["rank_gain_vs_v4"] = selected["v4_exact_rank"].fillna(len(item_ids) + 1) - selected[
        "exact_rank"
    ].fillna(len(item_ids) + 1)
    examples = selected.sort_values(
        ["rank_gain_vs_v4", "preference_shift_score"], ascending=False
    ).head(100)
    examples.to_csv(args.output_dir / "recommendation_shift_examples.csv", index=False)
    selected.to_parquet(args.output_dir / "online_replay_per_query.parquet", index=False)

    latency = {
        "item_encoding_seconds": item_encoding_seconds,
        "user_encoding_seconds": user_encoding_seconds,
        "exact_scoring_seconds": scoring_seconds,
        "exact_scoring_queries_per_second": len(holdout_indices) / max(scoring_seconds, 1e-9),
        "total_seconds": time.perf_counter() - started,
        "note": "Offline exact-matrix throughput; not an ANN serving latency measurement.",
    }
    report = {
        "protocol": "v6_point_in_time_replay_v4_2_holdout",
        "ordering": "global non-decreasing query timestamp",
        "state_policy": "strict-prior sanitized state only",
        "v3_status": "not evaluated: no V3 inference checkpoint exists on the locked A2 protocol",
        "queries": int(len(holdout_indices)),
        "users": int(len(holdout_users)),
        "candidate_items": int(len(item_ids)),
        "selected_v4_2": {
            "author_weight": args.author_weight,
            "popularity_weight": args.popularity_weight,
            "recency_tau": args.recency_tau,
        },
        "point_in_time_validation": replay_validation,
        "summary": {
            row["variant"]: {key: value for key, value in row.items() if key != "variant"}
            for row in summaries
        },
        "paired_user_bootstrap": comparisons,
        "latency": latency,
    }
    save_json(args.output_dir / "v6_online_replay.json", report)
    print(summary.to_string(index=False), flush=True)
    print(json.dumps({"latency": latency, "point_in_time": replay_validation}, indent=2))


if __name__ == "__main__":
    main()
