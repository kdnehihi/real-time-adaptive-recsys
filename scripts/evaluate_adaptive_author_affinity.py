from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch

from recommender.models.adaptive.author_branch import author_affinity_scores, encode_raw_recent_authors
from recommender.models.adaptive.checkpoint import encode_items, encode_users, load_artifacts, save_json, select_device
from scripts.audit_adaptive_content_readiness import STATIC_COLUMNS
from scripts.evaluate_adaptive_protocol_v2 import (
    build_safe_candidate_pool,
    load_event_history,
    load_protocol_queries,
    metric_columns,
    prior_seen_items,
    read_shard,
    segment_summary,
    user_macro_summary,
)
from recommender.models.adaptive.checkpoint import parquet_files


def parse_floats(raw: str) -> tuple[float, ...]:
    return tuple(sorted({float(value.strip()) for value in raw.split(",") if value.strip()}))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate and evaluate point-in-time V4.2 author affinity.")
    parser.add_argument("--adaptive-root", type=Path, required=True)
    parser.add_argument("--base-gold-root", type=Path, required=True)
    parser.add_argument("--v4-dir", type=Path, required=True)
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--author-weights", default="0,0.005,0.01,0.02,0.05,0.1,0.2")
    parser.add_argument("--popularity-weights", default="0,0.01,0.02")
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
    """Load immutable video-to-author metadata one Drive shard at a time."""
    mapping: dict[int, int] = {}
    files = parquet_files(static_path)
    for index, file_path in enumerate(files, start=1):
        part = read_shard(file_path, list(STATIC_COLUMNS)).dropna(subset=["video_id", "author_id"])
        mapping.update(
            zip(part["video_id"].astype("int64"), part["author_id"].astype("int64"))
        )
        if index % 16 == 0 or index == len(files):
            print(f"author metadata: {index}/{len(files)} shards; {len(mapping):,} videos", flush=True)
    return mapping


def deterministic_user_split(user_ids: pd.Series, fraction: float, seed: int) -> np.ndarray:
    """Assign all queries from one user to exactly one evaluation partition."""
    if not 0.0 < fraction < 1.0:
        raise ValueError("calibration-user-fraction must be between zero and one")
    values = user_ids.astype("int64").to_numpy()
    hashed = (values * 2_654_435_761 + int(seed) * 97_531) % 1_000_003
    return hashed < int(fraction * 1_000_003)


def build_query_mask(
    part: pd.DataFrame,
    item_ids: np.ndarray,
    eligible_from_ms: np.ndarray,
    item_to_index: dict[int, int],
    history: dict[int, tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build identical time/seen masks and restore the exact target when repeated."""
    mask = eligible_from_ms[None, :] > part["query_time_ms"].to_numpy("int64")[:, None]
    targets = np.empty(len(part), dtype=np.int64)
    repeats = np.zeros(len(part), dtype=bool)
    for offset, row in enumerate(part.itertuples(index=False)):
        target = item_to_index.get(int(row.video_id))
        if target is None:
            raise RuntimeError(f"Target {row.video_id} is missing from the candidate catalog")
        seen = prior_seen_items(history, int(row.user_id), int(row.query_time_ms))
        seen_indices = [item_to_index[item] for item in seen if item in item_to_index]
        if seen_indices:
            mask[offset, seen_indices] = True
        repeats[offset] = int(row.video_id) in seen
        mask[offset, target] = False
        targets[offset] = target
    return mask, targets, repeats


def ranks_to_frame(
    protocol: pd.DataFrame,
    indices: np.ndarray,
    ranks: np.ndarray,
    repeats: np.ndarray,
    top_ks: tuple[int, ...],
) -> pd.DataFrame:
    """Convert exact ranks into query-level metrics without materializing Top-K IDs."""
    selected = protocol.iloc[indices].reset_index(drop=True)
    rows = pd.DataFrame(
        {
            "example_id": selected["example_id"],
            "user_id": selected["user_id"].astype("int64"),
            "target_video_id": selected["video_id"].astype("int64"),
            "target_warm": selected["target_warm"].astype(bool),
            "target_repeated": repeats,
            "timeline_bucket": selected["timeline_bucket"],
            "session_history_bucket": selected["session_history_bucket_v2"],
            "exact_rank": np.where(ranks > 0, ranks, np.nan),
        }
    )
    for k in top_ks:
        hit = (ranks > 0) & (ranks <= k)
        rows[f"hitrate@{k}"] = hit.astype("float32")
        rows[f"ndcg@{k}"] = np.where(hit, 1.0 / np.log2(ranks + 1), 0.0)
        rows[f"mrr@{k}"] = np.where(hit, 1.0 / ranks, 0.0)
    return rows


def score_configs(
    indices: np.ndarray,
    configs: tuple[tuple[float, float], ...],
    protocol: pd.DataFrame,
    user_embeddings: np.ndarray,
    item_embeddings: np.ndarray,
    recent_authors: torch.Tensor,
    candidate_authors: torch.Tensor,
    popularity: np.ndarray,
    item_ids: np.ndarray,
    eligible_from_ms: np.ndarray,
    history: dict[int, tuple[np.ndarray, np.ndarray]],
    batch_size: int,
    recency_tau: float,
    device: torch.device,
) -> dict[tuple[float, float], tuple[np.ndarray, np.ndarray]]:
    """Compute exact target ranks for each weight pair using bounded GPU memory."""
    item_tensor = torch.as_tensor(item_embeddings, dtype=torch.float32, device=device)
    candidate_authors = candidate_authors.to(device)
    popularity_tensor = torch.as_tensor(popularity, dtype=torch.float32, device=device)
    item_to_index = {int(value): index for index, value in enumerate(item_ids)}
    ranks = {config: np.zeros(len(indices), dtype=np.int64) for config in configs}
    repeat_flags = np.zeros(len(indices), dtype=bool)
    for start in range(0, len(indices), batch_size):
        end = min(start + batch_size, len(indices))
        query_indices = indices[start:end]
        part = protocol.iloc[query_indices]
        mask, targets, repeats = build_query_mask(
            part, item_ids, eligible_from_ms, item_to_index, history
        )
        repeat_flags[start:end] = repeats
        mask_tensor = torch.as_tensor(mask, dtype=torch.bool, device=device)
        target_tensor = torch.as_tensor(targets, dtype=torch.long, device=device)
        users = torch.as_tensor(user_embeddings[query_indices], dtype=torch.float32, device=device)
        base = users @ item_tensor.T
        affinity = author_affinity_scores(
            recent_authors[query_indices].to(device), candidate_authors, recency_tau
        )
        for config in configs:
            author_weight, popularity_weight = config
            scores = base
            if author_weight:
                scores = scores + float(author_weight) * affinity
            if popularity_weight:
                scores = scores + float(popularity_weight) * popularity_tensor
            scores = scores.masked_fill(mask_tensor, -torch.inf)
            target_scores = scores.gather(1, target_tensor[:, None])
            exact_rank = 1 + scores.gt(target_scores).sum(dim=1)
            ranks[config][start:end] = exact_rank.cpu().numpy()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if end % (batch_size * 20) == 0 or end == len(indices):
            print(f"scored queries: {end}/{len(indices)} across {len(configs)} configs", flush=True)
    return {config: (values, repeat_flags.copy()) for config, values in ranks.items()}


def paired_bootstrap(
    candidate: pd.DataFrame,
    baseline: pd.DataFrame,
    metrics: list[str],
    samples: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    """Estimate candidate-minus-baseline deltas with user-clustered resampling."""
    joined = candidate.merge(
        baseline,
        on=["example_id", "user_id"],
        suffixes=("_candidate", "_baseline"),
        validate="one_to_one",
    )
    delta_columns: dict[str, pd.Series] = {}
    for metric in metrics:
        delta_columns[metric] = (
            joined[f"{metric}_candidate"] - joined[f"{metric}_baseline"]
        )
    deltas = pd.DataFrame({"user_id": joined["user_id"], **delta_columns})
    per_user = deltas.groupby("user_id", sort=False)[metrics].mean()
    values = per_user.to_numpy("float64")
    rng = np.random.default_rng(seed)
    draws = np.empty((samples, len(metrics)), dtype=np.float64)
    for index in range(samples):
        draws[index] = values[rng.integers(0, len(values), size=len(values))].mean(axis=0)
    return {
        metric: {
            "delta": float(values[:, offset].mean()),
            "ci95_low": float(np.quantile(draws[:, offset], 0.025)),
            "ci95_high": float(np.quantile(draws[:, offset], 0.975)),
        }
        for offset, metric in enumerate(metrics)
    }


def config_name(config: tuple[float, float]) -> str:
    return f"author_{config[0]:g}_pop_{config[1]:g}".replace(".", "p")


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
    print("device/configs:", device, len(configs), flush=True)

    queries, protocol = load_protocol_queries(
        args.adaptive_root, args.protocol_dir, "validation", None
    )
    candidate_pool = build_safe_candidate_pool(args.adaptive_root, "validation")
    artifacts = load_artifacts(args.adaptive_root, args.base_gold_root, args.v4_dir, device=device)
    item_ids, item_embeddings = encode_items(
        artifacts.v4, candidate_pool, queries.iloc[0].to_dict(), artifacts, args.item_batch_size
    )
    users = encode_users(artifacts.v4, queries, artifacts, batch_size=512)

    video_to_author = load_video_author_mapping(args.base_gold_root / "item_features/static")
    recent_authors = encode_raw_recent_authors(
        queries, video_to_author, int(artifacts.v4_cfg["sequence_max_len"])
    )
    candidate_authors = torch.as_tensor(
        [int(video_to_author.get(int(video), 0) or 0) for video in item_ids], dtype=torch.long
    )
    popularity = np.log1p(np.maximum(candidate_pool["item_hist_events"].to_numpy("float32"), 0.0))
    popularity = np.nan_to_num(
        (popularity - popularity.mean()) / (popularity.std() + 1e-6)
    ).astype("float32")
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
    if not len(calibration_indices) or not len(holdout_indices):
        raise RuntimeError("Deterministic user split produced an empty partition")
    if set(protocol.iloc[calibration_indices]["user_id"]) & set(protocol.iloc[holdout_indices]["user_id"]):
        raise AssertionError("User leakage between calibration and holdout")
    print(
        "calibration/holdout queries/users:",
        len(calibration_indices), protocol.iloc[calibration_indices]["user_id"].nunique(),
        len(holdout_indices), protocol.iloc[holdout_indices]["user_id"].nunique(),
        flush=True,
    )

    calibration_scores = score_configs(
        calibration_indices, configs, protocol, users, item_embeddings, recent_authors,
        candidate_authors, popularity, item_ids,
        candidate_pool["eligible_from_ms"].to_numpy("int64"), history,
        args.query_batch_size, args.recency_tau, device,
    )
    calibration_rows: list[dict[str, object]] = []
    for config, (ranks, repeats) in calibration_scores.items():
        frame = ranks_to_frame(protocol, calibration_indices, ranks, repeats, top_ks)
        calibration_rows.append(
            {"config": config_name(config), "author_weight": config[0], "popularity_weight": config[1],
             **user_macro_summary(frame, metrics)}
        )
    calibration = pd.DataFrame(calibration_rows).sort_values(
        ["hitrate@1000", "ndcg@100", "ndcg@1000"], ascending=False
    ).reset_index(drop=True)
    calibration.to_csv(args.output_dir / "calibration_grid.csv", index=False)
    selected = (
        float(calibration.iloc[0]["author_weight"]),
        float(calibration.iloc[0]["popularity_weight"]),
    )
    print("selected config:", selected, flush=True)

    holdout_configs = tuple(dict.fromkeys(((0.0, 0.0), (0.0, 0.02), (0.0, selected[1]), selected)))
    holdout_scores = score_configs(
        holdout_indices, holdout_configs, protocol, users, item_embeddings, recent_authors,
        candidate_authors, popularity, item_ids,
        candidate_pool["eligible_from_ms"].to_numpy("int64"), history,
        args.query_batch_size, args.recency_tau, device,
    )
    frames: dict[tuple[float, float], pd.DataFrame] = {}
    summary_rows: list[dict[str, object]] = []
    segment_rows: list[pd.DataFrame] = []
    for config, (ranks, repeats) in holdout_scores.items():
        frame = ranks_to_frame(protocol, holdout_indices, ranks, repeats, top_ks)
        frame.insert(0, "variant", config_name(config))
        frames[config] = frame
        frame.to_parquet(args.output_dir / f"per_query_{config_name(config)}.parquet", index=False)
        summary_rows.append({"variant": config_name(config), **user_macro_summary(frame, metrics)})
        segment = segment_summary(frame, metrics)
        segment.insert(0, "variant", config_name(config))
        segment_rows.append(segment)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.output_dir / "holdout_summary.csv", index=False)
    pd.concat(segment_rows, ignore_index=True).to_csv(args.output_dir / "holdout_segments.csv", index=False)

    matched_baseline = (0.0, selected[1])
    comparisons = {
        "selected_vs_matched_popularity_baseline": paired_bootstrap(
            frames[selected], frames[matched_baseline], metrics, args.bootstrap_samples, args.seed
        ),
        "selected_vs_raw_v4": paired_bootstrap(
            frames[selected], frames[(0.0, 0.0)], metrics, args.bootstrap_samples, args.seed + 1
        ),
    }
    primary = comparisons["selected_vs_matched_popularity_baseline"]["hitrate@1000"]
    accepted = bool(selected[0] > 0 and primary["delta"] > 0 and primary["ci95_low"] > 0)
    report = {
        "protocol": "adaptive_eval_v2_v4_2_author_affinity",
        "selection_policy": "user-disjoint calibration; maximize HR@1000 then NDCG@100/NDCG@1000",
        "calibration_queries": int(len(calibration_indices)),
        "holdout_queries": int(len(holdout_indices)),
        "candidate_items": int(len(item_ids)),
        "selected": {"author_weight": selected[0], "popularity_weight": selected[1]},
        "recency_tau": float(args.recency_tau),
        "holdout_summary": {
            row["variant"]: {key: value for key, value in row.items() if key != "variant"}
            for row in summary_rows
        },
        "paired_user_bootstrap": comparisons,
        "acceptance_rule": "author_weight > 0 and HR@1000 paired CI95 lower bound > 0 vs same-popularity V4",
        "accepted": accepted,
    }
    save_json(args.output_dir / "v4_2_author_affinity_evaluation.json", report)
    print(summary.to_string(index=False), flush=True)
    print(json.dumps({"selected": report["selected"], "accepted": accepted, "primary": primary}, indent=2))


if __name__ == "__main__":
    main()
