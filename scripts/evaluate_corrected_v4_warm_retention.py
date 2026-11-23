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

from recommender.models.adaptive.author_branch import (
    author_affinity_scores,
    encode_raw_recent_authors,
)
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
    build_query_mask,
    deterministic_user_split,
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
    read_shard,
    segment_summary,
    user_macro_summary,
)


RetentionConfig = tuple[float, str]


def parse_floats(raw: str) -> tuple[float, ...]:
    return tuple(sorted({float(value.strip()) for value in raw.split(",") if value.strip()}))


def parse_modes(raw: str) -> tuple[str, ...]:
    modes = tuple(dict.fromkeys(value.strip() for value in raw.split(",") if value.strip()))
    invalid = sorted(set(modes) - {"all", "warm"})
    if invalid:
        raise ValueError(f"Unknown static-score modes: {invalid}")
    return modes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate a production-safe static/long-term score over fixed Corrected V4."
    )
    parser.add_argument("--adaptive-root", type=Path, required=True)
    parser.add_argument("--base-gold-root", type=Path, required=True)
    parser.add_argument("--corrected-root", type=Path, required=True)
    parser.add_argument("--corrected-v4-dir", type=Path, required=True)
    parser.add_argument("--legacy-v4-dir", type=Path, required=True)
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--static-weights", default="0,0.05,0.1,0.2,0.3,0.5,0.75,1.0")
    parser.add_argument("--static-modes", default="all,warm")
    parser.add_argument("--author-weight", type=float, default=0.2)
    parser.add_argument("--popularity-weight", type=float, default=0.02)
    parser.add_argument("--recency-tau", type=float, default=3.0)
    parser.add_argument("--top-ks", default="10,50,100,500,1000")
    parser.add_argument("--calibration-user-fraction", type=float, default=0.35)
    parser.add_argument("--query-batch-size", type=int, default=16)
    parser.add_argument("--item-batch-size", type=int, default=8192)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def config_name(config: RetentionConfig) -> str:
    weight, mode = config
    return f"static_{weight:g}_{mode}".replace(".", "p")


def apply_static_retention(
    base_scores: torch.Tensor,
    static_scores: torch.Tensor,
    candidate_warm: torch.Tensor,
    config: RetentionConfig,
) -> torch.Tensor:
    """Add one ANN-compatible long-term score, optionally only to warm items."""
    weight, mode = config
    if weight == 0:
        return base_scores
    if mode == "all":
        return base_scores + float(weight) * static_scores
    if mode == "warm":
        return base_scores + float(weight) * static_scores * candidate_warm.to(static_scores.dtype)
    raise ValueError(f"Unknown static retention mode: {mode}")


def load_video_author_mapping(static_path: Path) -> dict[int, int]:
    """Read immutable item authors shard-by-shard without retaining file handles."""
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


def score_retention_configs(
    indices: np.ndarray,
    configs: tuple[RetentionConfig, ...],
    protocol: pd.DataFrame,
    adaptive_users: np.ndarray,
    static_users: np.ndarray,
    item_embeddings: np.ndarray,
    recent_authors: torch.Tensor,
    candidate_authors: torch.Tensor,
    popularity: np.ndarray,
    candidate_warm: np.ndarray,
    item_ids: np.ndarray,
    eligible_from_ms: np.ndarray,
    history: dict[int, tuple[np.ndarray, np.ndarray]],
    author_weight: float,
    popularity_weight: float,
    batch_size: int,
    recency_tau: float,
    device: torch.device,
) -> dict[RetentionConfig, tuple[np.ndarray, np.ndarray]]:
    """Rank exact targets for every static-score configuration with one item matrix."""
    item_tensor = torch.as_tensor(item_embeddings, dtype=torch.float32, device=device)
    author_tensor = candidate_authors.to(device)
    popularity_tensor = torch.as_tensor(popularity, dtype=torch.float32, device=device)
    warm_tensor = torch.as_tensor(candidate_warm, dtype=torch.float32, device=device)[None, :]
    item_to_index = {int(value): index for index, value in enumerate(item_ids)}
    ranks = {config: np.zeros(len(indices), dtype=np.int64) for config in configs}
    repeats_all = np.zeros(len(indices), dtype=bool)
    for start in range(0, len(indices), batch_size):
        end = min(start + batch_size, len(indices))
        query_indices = indices[start:end]
        part = protocol.iloc[query_indices]
        mask, targets, repeats = build_query_mask(
            part, item_ids, eligible_from_ms, item_to_index, history
        )
        repeats_all[start:end] = repeats
        mask_tensor = torch.as_tensor(mask, dtype=torch.bool, device=device)
        target_tensor = torch.as_tensor(targets, dtype=torch.long, device=device)
        adaptive = torch.as_tensor(adaptive_users[query_indices], dtype=torch.float32, device=device)
        static = torch.as_tensor(static_users[query_indices], dtype=torch.float32, device=device)
        adaptive_scores = adaptive @ item_tensor.T
        static_scores = static @ item_tensor.T
        base = adaptive_scores
        if author_weight:
            base = base + float(author_weight) * author_affinity_scores(
                recent_authors[query_indices].to(device), author_tensor, recency_tau
            )
        if popularity_weight:
            base = base + float(popularity_weight) * popularity_tensor
        for config in configs:
            scores = apply_static_retention(base, static_scores, warm_tensor, config)
            scores = scores.masked_fill(mask_tensor, -torch.inf)
            target_scores = scores.gather(1, target_tensor[:, None])
            ranks[config][start:end] = (1 + scores.gt(target_scores).sum(dim=1)).cpu().numpy()
        del adaptive, static, adaptive_scores, static_scores, base
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if end % (batch_size * 20) == 0 or end == len(indices):
            print(f"scored queries: {end}/{len(indices)} across {len(configs)} retention configs", flush=True)
    return {config: (values, repeats_all.copy()) for config, values in ranks.items()}


def split_summary(frame: pd.DataFrame, metrics: list[str]) -> dict[str, dict[str, float | int]]:
    return {
        ("warm" if bool(value) else "cold"): user_macro_summary(part, metrics)
        for value, part in frame.groupby("target_warm")
    }


def select_calibration_config(rows: list[dict[str, object]]) -> RetentionConfig:
    """Select only configs that improve overall/warm HR and preserve cold HR."""
    table = pd.DataFrame(rows)
    baseline = table[(table["static_weight"] == 0.0)].iloc[0]
    eligible = table[
        (table["static_weight"] > 0.0)
        & (table["hitrate@1000"] >= baseline["hitrate@1000"])
        & (table["warm_hitrate@1000"] >= baseline["warm_hitrate@1000"])
        & (table["cold_hitrate@1000"] >= baseline["cold_hitrate@1000"])
    ]
    if eligible.empty:
        return (0.0, "none")
    selected = eligible.sort_values(
        ["hitrate@1000", "ndcg@1000", "warm_hitrate@1000"], ascending=False
    ).iloc[0]
    return (float(selected["static_weight"]), str(selected["static_mode"]))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)
    top_ks = tuple(sorted({int(value) for value in args.top_ks.split(",")}))
    metrics = metric_columns(top_ks)
    weights = parse_floats(args.static_weights)
    modes = parse_modes(args.static_modes)
    configs = tuple(dict.fromkeys([(0.0, "none"), *[(w, m) for w in weights if w > 0 for m in modes]]))
    print("device/retention configs:", device, len(configs), flush=True)

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

    print("encoding Corrected V4 adaptive/static users and items...", flush=True)
    artifacts = load_artifacts(
        args.adaptive_root, args.base_gold_root, args.corrected_v4_dir, device=device
    )
    item_ids, item_embeddings = encode_items(
        artifacts.v4, candidate_pool, corrected.iloc[0].to_dict(), artifacts, args.item_batch_size
    )
    adaptive_users = encode_users(artifacts.v4, corrected, artifacts, batch_size=512)
    static_users = encode_users(
        artifacts.v4, make_static_ablation(corrected), artifacts, batch_size=512
    )
    recent_authors = encode_raw_recent_authors(
        corrected, video_to_author, int(artifacts.v4_cfg["sequence_max_len"])
    )
    candidate_authors = torch.as_tensor(
        [int(video_to_author.get(int(video_id), 0) or 0) for video_id in item_ids],
        dtype=torch.long,
    )
    eligible = candidate_pool["eligible_from_ms"].to_numpy("int64")
    candidate_warm = candidate_pool["target_warm"].to_numpy(bool)

    calibration_scores = score_retention_configs(
        calibration_indices, configs, protocol, adaptive_users, static_users,
        item_embeddings, recent_authors, candidate_authors, popularity, candidate_warm,
        item_ids, eligible, history, args.author_weight, args.popularity_weight,
        args.query_batch_size, args.recency_tau, device,
    )
    calibration_rows: list[dict[str, object]] = []
    for config, (ranks, repeats) in calibration_scores.items():
        frame = ranks_to_frame(protocol, calibration_indices, ranks, repeats, top_ks)
        segments = split_summary(frame, metrics)
        calibration_rows.append(
            {
                "config": config_name(config),
                "static_weight": config[0],
                "static_mode": config[1],
                **user_macro_summary(frame, metrics),
                "warm_hitrate@1000": segments.get("warm", {}).get("hitrate@1000", 0.0),
                "cold_hitrate@1000": segments.get("cold", {}).get("hitrate@1000", 0.0),
            }
        )
    calibration = pd.DataFrame(calibration_rows).sort_values(
        ["hitrate@1000", "ndcg@1000"], ascending=False
    )
    calibration.to_csv(args.output_dir / "calibration_grid.csv", index=False)
    selected = select_calibration_config(calibration_rows)
    print("selected retention config:", selected, flush=True)

    holdout_configs = tuple(dict.fromkeys(((0.0, "none"), selected)))
    holdout_scores = score_retention_configs(
        holdout_indices, holdout_configs, protocol, adaptive_users, static_users,
        item_embeddings, recent_authors, candidate_authors, popularity, candidate_warm,
        item_ids, eligible, history, args.author_weight, args.popularity_weight,
        args.query_batch_size, args.recency_tau, device,
    )
    del artifacts, adaptive_users, static_users
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("encoding Legacy V4.2 reference...", flush=True)
    legacy_artifacts = load_artifacts(
        args.adaptive_root, args.base_gold_root, args.legacy_v4_dir, device=device
    )
    legacy_item_ids, legacy_items = encode_items(
        legacy_artifacts.v4, candidate_pool, legacy_queries.iloc[0].to_dict(),
        legacy_artifacts, args.item_batch_size,
    )
    if not np.array_equal(item_ids, legacy_item_ids):
        raise AssertionError("Legacy and Corrected candidate item order differs")
    legacy_users = encode_users(legacy_artifacts.v4, legacy_queries, legacy_artifacts, batch_size=512)
    legacy_recent_authors = encode_raw_recent_authors(
        legacy_queries, video_to_author, int(legacy_artifacts.v4_cfg["sequence_max_len"])
    )
    legacy_config = (args.author_weight, args.popularity_weight)
    legacy_result = score_configs(
        holdout_indices, (legacy_config,), protocol, legacy_users, legacy_items,
        legacy_recent_authors, candidate_authors, popularity, item_ids, eligible, history,
        args.query_batch_size, args.recency_tau, device,
    )[legacy_config]

    frames: dict[str, pd.DataFrame] = {}
    for config, (ranks, repeats) in holdout_scores.items():
        name = "corrected_base" if config[0] == 0 else f"corrected_{config_name(config)}"
        frame = ranks_to_frame(protocol, holdout_indices, ranks, repeats, top_ks)
        frame.insert(0, "variant", name)
        frames[name] = frame
    legacy_ranks, legacy_repeats = legacy_result
    legacy = ranks_to_frame(protocol, holdout_indices, legacy_ranks, legacy_repeats, top_ks)
    legacy.insert(0, "variant", "legacy_v4_2")
    frames["legacy_v4_2"] = legacy

    selected_name = "corrected_base" if selected[0] == 0 else f"corrected_{config_name(selected)}"
    summary_rows = [
        {"variant": name, **user_macro_summary(frame, metrics)} for name, frame in frames.items()
    ]
    pd.DataFrame(summary_rows).to_csv(args.output_dir / "holdout_summary.csv", index=False)
    segment_rows = []
    for name, frame in frames.items():
        segment = segment_summary(frame, metrics)
        segment.insert(0, "variant", name)
        segment_rows.append(segment)
        frame.to_parquet(args.output_dir / f"per_query_{name}.parquet", index=False)
    pd.concat(segment_rows, ignore_index=True).to_csv(
        args.output_dir / "holdout_segments.csv", index=False
    )

    base = frames["corrected_base"]
    selected_frame = frames[selected_name]
    comparisons = {
        "selected_vs_corrected_base": paired_bootstrap(
            selected_frame, base, metrics, args.bootstrap_samples, args.seed
        ),
        "selected_vs_legacy_v4_2": paired_bootstrap(
            selected_frame, legacy, metrics, args.bootstrap_samples, args.seed + 1
        ),
    }
    segment_comparisons = {}
    for label, value in (("cold", False), ("warm", True)):
        selected_part = selected_frame[selected_frame["target_warm"].eq(value)]
        base_part = base[base["target_warm"].eq(value)]
        segment_comparisons[f"selected_vs_base_{label}"] = paired_bootstrap(
            selected_part, base_part, metrics, args.bootstrap_samples, args.seed + (2 if value else 3)
        )
    selected_segments = split_summary(selected_frame, metrics)
    base_segments = split_summary(base, metrics)
    overall_delta = comparisons["selected_vs_corrected_base"]["hitrate@1000"]
    warm_delta = segment_comparisons["selected_vs_base_warm"]["hitrate@1000"]
    gates = {
        "nonzero_static_branch": selected[0] > 0,
        "overall_gain_gate": overall_delta["delta"] > 0 and overall_delta["ci95_low"] > 0,
        "warm_retention_gate": warm_delta["delta"] > 0 and warm_delta["ci95_low"] > 0,
        "cold_non_regression_gate": selected_segments["cold"]["hitrate@1000"]
        >= base_segments["cold"]["hitrate@1000"],
        "legacy_reference_gate": comparisons["selected_vs_legacy_v4_2"]["hitrate@1000"]["ci95_low"] > 0,
    }
    gates["accepted_for_next_stage"] = bool(
        gates["nonzero_static_branch"]
        and gates["overall_gain_gate"]
        and gates["warm_retention_gate"]
        and gates["cold_non_regression_gate"]
    )
    gates["production_promotion"] = bool(
        gates["accepted_for_next_stage"] and gates["legacy_reference_gate"]
    )
    report = {
        "protocol": "adaptive_eval_v2_corrected_v4_3_warm_retention",
        "selection_policy": "user-disjoint validation calibration; improve overall and warm HR@1000 while preserving cold HR@1000",
        "test_data_used": False,
        "calibration_queries": int(len(calibration_indices)),
        "holdout_queries": int(len(holdout_indices)),
        "candidate_items": int(len(item_ids)),
        "fixed_author_weight": args.author_weight,
        "fixed_popularity_weight": args.popularity_weight,
        "selected": {"static_weight": selected[0], "static_mode": selected[1]},
        "holdout_summary": {
            row["variant"]: {key: value for key, value in row.items() if key != "variant"}
            for row in summary_rows
        },
        "cold_warm_summary": {
            name: split_summary(frame, metrics) for name, frame in frames.items()
        },
        "paired_user_bootstrap": {**comparisons, **segment_comparisons},
        "acceptance_gates": gates,
        "accepted": gates["accepted_for_next_stage"],
    }
    save_json(args.output_dir / "corrected_v4_3_warm_retention.json", report)
    print(pd.DataFrame(summary_rows).to_string(index=False), flush=True)
    print(json.dumps({"selected": report["selected"], "gates": gates}, indent=2), flush=True)


if __name__ == "__main__":
    main()
