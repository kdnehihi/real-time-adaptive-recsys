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
import pyarrow.dataset as ds

from recommender.models.adaptive.checkpoint import (
    INPUT_COLUMNS,
    encode_items,
    load_artifacts,
    prepare_item_pool,
    read_parquet,
    retrieve_for_queries,
    sanitize_point_in_time_sequences,
    save_json,
    select_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit adaptive retrieval coverage before training a higher-capacity model."
    )
    parser.add_argument("--adaptive-root", type=Path, required=True)
    parser.add_argument("--base-gold-root", type=Path, required=True)
    parser.add_argument("--v4-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="validation", choices=["validation", "test"])
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--max-candidate-items", type=int, default=0)
    parser.add_argument("--top-ks", default="10,50,100,500,1000")
    parser.add_argument("--popularity-blend-grid", default="0.0,0.02")
    parser.add_argument("--item-batch-size", type=int, default=8192)
    parser.add_argument("--query-batch-size", type=int, default=32)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def parse_ints(raw: str) -> tuple[int, ...]:
    return tuple(sorted({int(value.strip()) for value in raw.split(",") if value.strip()}))


def parse_floats(raw: str) -> tuple[float, ...]:
    return tuple(float(value.strip()) for value in raw.split(",") if value.strip())


def load_queries(root: Path, split: str, limit: int | None) -> pd.DataFrame:
    frame = read_parquet(
        root / "sequence_examples" / split,
        INPUT_COLUMNS,
        None,
        ds.field("target_class") == "STRONG_POSITIVE",
    )
    frame = sanitize_point_in_time_sequences(frame)
    frame = (
        frame.sort_values(["user_id", "as_of_time", "example_id"])
        .drop_duplicates("user_id", keep="last")
        .reset_index(drop=True)
    )
    return frame.head(limit).reset_index(drop=True) if limit is not None else frame


def load_relevance(root: Path, split: str) -> pd.DataFrame:
    return (
        read_parquet(
            root / "sequence_examples" / split,
            ["user_id", "video_id"],
            None,
            ds.field("target_class") == "STRONG_POSITIVE",
        )
        .drop_duplicates(["user_id", "video_id"])
        .reset_index(drop=True)
    )


def load_train_item_ids(root: Path) -> set[int]:
    frame = read_parquet(root / "sequence_examples/train", ["video_id"])
    return set(frame["video_id"].dropna().astype("int64").unique().tolist())


def history_bucket(value: object) -> str:
    count = int(value or 0)
    if count == 0:
        return "empty"
    if count < 5:
        return "short_1_4"
    if count < 20:
        return "medium_5_19"
    return "long_20_plus"


def per_query_metrics(
    candidate_ids: np.ndarray,
    ranked_idx: np.ndarray,
    queries: pd.DataFrame,
    relevance_frame: pd.DataFrame,
    train_item_ids: set[int],
    top_ks: tuple[int, ...],
) -> pd.DataFrame:
    relevance = (
        relevance_frame.groupby("user_id")["video_id"]
        .apply(lambda values: set(int(value) for value in values.dropna()))
        .to_dict()
    )
    max_k = min(max(top_ks), ranked_idx.shape[1])
    ranked_ids = candidate_ids[ranked_idx[:, :max_k]]
    candidate_catalog = set(int(value) for value in candidate_ids)
    rows: list[dict[str, object]] = []
    for position, query in queries.reset_index(drop=True).iterrows():
        user_id = int(query["user_id"])
        target = int(query["video_id"])
        relevant = relevance.get(user_id, set())
        ranked = [int(value) for value in ranked_ids[position]]
        row: dict[str, object] = {
            "query_index": position,
            "user_id": user_id,
            "target_video_id": target,
            "target_in_candidate_catalog": target in candidate_catalog,
            "target_warm": target in train_item_ids,
            "history_length": int(query.get("recent_event_count", 0) or 0),
            "history_bucket": history_bucket(query.get("recent_event_count", 0)),
            "relevant_items": len(relevant),
        }
        target_positions = [index for index, item in enumerate(ranked) if item == target]
        row["exact_target_rank"] = target_positions[0] + 1 if target_positions else np.nan
        for k in top_ks:
            actual_k = min(k, len(ranked))
            top = ranked[:actual_k]
            flags = np.asarray([item in relevant for item in top], dtype=np.float32)
            hits = float(flags.sum())
            dcg = float(sum(flag / math.log2(index + 2) for index, flag in enumerate(flags)))
            idcg = float(sum(1.0 / math.log2(index + 2) for index in range(min(len(relevant), actual_k))))
            row[f"hitrate@{k}"] = float(hits > 0)
            row[f"recall@{k}"] = hits / max(len(relevant), 1)
            row[f"ndcg@{k}"] = dcg / idcg if idcg else 0.0
            row[f"exact_hitrate@{k}"] = float(target in top)
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(frame: pd.DataFrame, top_ks: tuple[int, ...]) -> dict[str, float | int]:
    metrics = [
        column
        for k in top_ks
        for column in (f"hitrate@{k}", f"recall@{k}", f"ndcg@{k}", f"exact_hitrate@{k}")
    ]
    output: dict[str, float | int] = {
        "evaluated_queries": int(len(frame)),
        "evaluated_users": int(frame["user_id"].nunique()),
        "target_catalog_coverage": float(frame["target_in_candidate_catalog"].mean()),
        "warm_target_rate": float(frame["target_warm"].mean()),
    }
    output.update({metric: float(frame[metric].mean()) for metric in metrics})
    return output


def bootstrap_intervals(
    frame: pd.DataFrame,
    metrics: list[str],
    samples: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(seed)
    values = frame[metrics].to_numpy("float64")
    estimates = values.mean(axis=0)
    if samples <= 0 or len(values) == 0:
        return {
            metric: {"mean": float(estimates[index]), "ci95_low": float("nan"), "ci95_high": float("nan")}
            for index, metric in enumerate(metrics)
        }
    draws = np.empty((samples, len(metrics)), dtype=np.float64)
    for sample in range(samples):
        indices = rng.integers(0, len(values), size=len(values))
        draws[sample] = values[indices].mean(axis=0)
    return {
        metric: {
            "mean": float(estimates[index]),
            "ci95_low": float(np.quantile(draws[:, index], 0.025)),
            "ci95_high": float(np.quantile(draws[:, index], 0.975)),
        }
        for index, metric in enumerate(metrics)
    }


def segment_summary(frame: pd.DataFrame, top_ks: tuple[int, ...]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    segments = {
        "all": pd.Series(True, index=frame.index),
        "warm_target": frame["target_warm"],
        "cold_target": ~frame["target_warm"],
    }
    for bucket in ["empty", "short_1_4", "medium_5_19", "long_20_plus"]:
        segments[f"history_{bucket}"] = frame["history_bucket"].eq(bucket)
    for name, mask in segments.items():
        part = frame.loc[mask]
        if part.empty:
            continue
        row: dict[str, object] = {"segment": name, "queries": len(part)}
        for k in top_ks:
            for metric in ("hitrate", "recall", "ndcg", "exact_hitrate"):
                row[f"{metric}@{k}"] = float(part[f"{metric}@{k}"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def popularity_ranking(candidate_pool: pd.DataFrame, count: int, queries: int) -> np.ndarray:
    popularity = pd.to_numeric(
        candidate_pool.get("item_hist_events", pd.Series(np.zeros(len(candidate_pool)))),
        errors="coerce",
    ).fillna(0).to_numpy("float64")
    order = np.argsort(-popularity)[:count]
    return np.tile(order[None, :], (queries, 1))


def main() -> None:
    args = parse_args()
    top_ks = parse_ints(args.top_ks)
    blend_weights = parse_floats(args.popularity_blend_grid)
    retrieval_k = max(top_ks)
    query_limit = None if args.max_queries == 0 else args.max_queries
    item_limit = None if args.max_candidate_items == 0 else args.max_candidate_items
    args.output_dir.mkdir(parents=True, exist_ok=True)

    required = [
        args.adaptive_root / "sequence_examples/train",
        args.adaptive_root / "sequence_examples" / args.split,
        args.base_gold_root / "transforms/numeric_stats.json",
        args.v4_dir / "best_model.pt",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing audit inputs:\n- " + "\n- ".join(missing))

    device = select_device(args.device)
    print("device:", device, flush=True)
    print("loading V4 and train-fitted preprocessing...", flush=True)
    artifacts = load_artifacts(args.adaptive_root, args.base_gold_root, args.v4_dir, device=device)
    queries = load_queries(args.adaptive_root, args.split, query_limit)
    relevance = load_relevance(args.adaptive_root, args.split)
    candidate_pool = prepare_item_pool(args.adaptive_root, args.split, item_limit)
    train_item_ids = load_train_item_ids(args.adaptive_root)
    print("queries/items:", len(queries), len(candidate_pool), flush=True)

    item_ids, item_embeddings = encode_items(
        artifacts.v4,
        candidate_pool,
        queries.iloc[0].to_dict(),
        artifacts,
        batch_size=args.item_batch_size,
    )
    popularity = pd.to_numeric(
        candidate_pool.get("item_hist_events", pd.Series(np.zeros(len(candidate_pool)))),
        errors="coerce",
    ).fillna(0).to_numpy("float32")
    popularity = np.log1p(np.maximum(popularity, 0.0))
    popularity = np.nan_to_num(
        (popularity - popularity.mean()) / (popularity.std() + 1e-6),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    variants: dict[str, pd.DataFrame] = {}
    popularity_idx = popularity_ranking(candidate_pool, retrieval_k, len(queries))
    variants["popularity"] = per_query_metrics(
        item_ids, popularity_idx, queries, relevance, train_item_ids, top_ks
    )
    for weight in blend_weights:
        name = "v4_raw" if weight == 0.0 else f"v4_popularity_{weight:g}"
        print("retrieving:", name, flush=True)
        _, ranked_idx, _ = retrieve_for_queries(
            artifacts,
            queries,
            item_ids,
            item_embeddings,
            retrieval_k,
            args.query_batch_size,
            item_score_bias=None if weight == 0.0 else weight * popularity,
        )
        variants[name] = per_query_metrics(
            item_ids, ranked_idx, queries, relevance, train_item_ids, top_ks
        )

    summary_rows = []
    segment_frames = []
    ci_metrics = [f"{metric}@{k}" for k in top_ks for metric in ("hitrate", "recall", "ndcg", "exact_hitrate")]
    confidence_intervals: dict[str, dict[str, dict[str, float]]] = {}
    for name, frame in variants.items():
        frame.insert(0, "variant", name)
        frame.to_parquet(args.output_dir / f"per_query_{name}.parquet", index=False)
        summary_rows.append({"variant": name, **summarize(frame, top_ks)})
        segments = segment_summary(frame, top_ks)
        segments.insert(0, "variant", name)
        segment_frames.append(segments)
        confidence_intervals[name] = bootstrap_intervals(
            frame, ci_metrics, args.bootstrap_samples, args.seed
        )

    summary_frame = pd.DataFrame(summary_rows)
    segment_frame = pd.concat(segment_frames, ignore_index=True)
    summary_frame.to_csv(args.output_dir / "summary.csv", index=False)
    segment_frame.to_csv(args.output_dir / "segments.csv", index=False)
    report = {
        "protocol": {
            "split": args.split,
            "query_unit": "latest_strong_positive_per_user",
            "queries": len(queries),
            "candidate_items": len(candidate_pool),
            "retrieval_k": retrieval_k,
            "top_ks": list(top_ks),
            "popularity_blend_grid": list(blend_weights),
            "bootstrap_samples": args.bootstrap_samples,
            "point_in_time_sanitized": True,
        },
        "summary": {row["variant"]: {k: v for k, v in row.items() if k != "variant"} for row in summary_rows},
        "bootstrap_ci95": confidence_intervals,
        "decision_gates": {
            "candidate_recall_gate": f"recall@{retrieval_k}",
            "ranking_primary_metric": "ndcg@10",
            "uncertainty_unit": "query/user bootstrap",
            "higher_model_allowed_only_after_protocol_passes": True,
        },
    }
    save_json(args.output_dir / "retrieval_audit.json", report)
    print(summary_frame.to_string(index=False), flush=True)
    print("wrote:", args.output_dir, flush=True)


if __name__ == "__main__":
    main()
