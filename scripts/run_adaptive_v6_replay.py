from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from recommender.models.adaptive.checkpoint import (
    INPUT_COLUMNS,
    LoadedAdaptiveArtifacts,
    encode_items,
    exact_next_metrics,
    load_artifacts,
    multi_positive_metrics,
    prepare_item_pool,
    read_parquet,
    rerank_with_v5,
    retrieve_for_queries,
    sanitize_point_in_time_sequences,
    save_json,
    select_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run V6-style online replay from local V4/V5 adaptive checkpoints.")
    parser.add_argument("--adaptive-root", type=Path, default=Path("data/gold/adaptive/v1"))
    parser.add_argument("--base-gold-root", type=Path, default=Path("data/gold/two_tower/v1_colab"))
    parser.add_argument("--v4-dir", type=Path, default=Path("data/artifacts/adaptive/v4_sequence_retrieval"))
    parser.add_argument("--v5-dir", type=Path, default=Path("data/artifacts/adaptive/v5_adaptive_ranker"))
    parser.add_argument("--v3-metrics", type=Path, default=Path("data/artifacts/two_tower/baseline_v3/metrics.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/artifacts/adaptive/v6_online_replay"))
    parser.add_argument("--split", default="validation", choices=["validation", "test"])
    parser.add_argument("--max-replay-queries", type=int, default=5000, help="Use 0 for all positive replay queries.")
    parser.add_argument("--max-replay-users", type=int, default=100, help="Deterministic user sample; use 0 for all users.")
    parser.add_argument("--max-candidate-items", type=int, default=300000, help="Use 0 for all candidate items.")
    parser.add_argument("--retrieval-candidates", type=int, default=1000)
    parser.add_argument("--top-ks", default="10,50,100")
    parser.add_argument("--item-batch-size", type=int, default=8192)
    parser.add_argument("--query-batch-size", type=int, default=64)
    parser.add_argument("--max-report-rows", type=int, default=5000)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def parse_int_tuple(raw: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in raw.split(",") if x.strip())


def validate_inputs(args: argparse.Namespace) -> None:
    required = [
        args.adaptive_root / "sequence_examples/train",
        args.adaptive_root / "sequence_examples" / args.split,
        args.base_gold_root / "transforms/numeric_stats.json",
        args.base_gold_root / "vocabularies/user_active_degree",
        args.base_gold_root / "vocabularies/video_type",
        args.base_gold_root / "vocabularies/upload_type",
        args.v4_dir / "best_model.pt",
        args.v5_dir / "best_model.pt",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required V6 replay inputs:\n- " + "\n- ".join(missing))


def load_replay_queries(
    adaptive_root: Path,
    split: str,
    max_queries: int | None,
    max_users: int | None,
) -> pd.DataFrame:
    frame = read_parquet(
        adaptive_root / "sequence_examples" / split,
        INPUT_COLUMNS,
        None,
        ds.field("target_class") == "STRONG_POSITIVE",
    ).reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"No STRONG_POSITIVE replay queries found in {adaptive_root / 'sequence_examples' / split}")
    frame["query_time_ms"] = (pd.to_datetime(frame["as_of_time"]).astype("int64") // 1_000_000).astype("int64")
    frame = frame.sort_values(["user_id", "query_time_ms", "example_id"]).reset_index(drop=True)

    user_ids = np.sort(frame["user_id"].dropna().unique())
    if max_users is not None and len(user_ids) > max_users:
        selected_positions = np.linspace(0, len(user_ids) - 1, num=max_users, dtype=int)
        user_ids = user_ids[selected_positions]
        frame = frame[frame["user_id"].isin(user_ids)]

    if max_queries is not None and len(frame) > max_queries:
        selected_users = np.sort(frame["user_id"].dropna().unique())
        base_quota, remainder = divmod(max_queries, len(selected_users))
        selected_parts: list[pd.DataFrame] = []
        for position, user_id in enumerate(selected_users):
            group = frame[frame["user_id"] == user_id]
            quota = base_quota + (1 if position < remainder else 0)
            quota = min(max(quota, 1), len(group))
            indices = np.linspace(0, len(group) - 1, num=quota, dtype=int)
            selected_parts.append(group.iloc[np.unique(indices)])
        frame = pd.concat(selected_parts, ignore_index=True)
        if len(frame) > max_queries:
            frame = frame.sort_values("query_time_ms").head(max_queries)

    frame = sanitize_point_in_time_sequences(frame.reset_index(drop=True))
    return frame.sort_values("query_time_ms").reset_index(drop=True)


def load_relevance_frame(adaptive_root: Path, split: str) -> pd.DataFrame:
    """Load compact user/item relevance without materializing sequence columns."""
    return read_parquet(
        adaptive_root / "sequence_examples" / split,
        ["user_id", "video_id"],
        None,
        ds.field("target_class") == "STRONG_POSITIVE",
    ).drop_duplicates(["user_id", "video_id"]).reset_index(drop=True)


def metric_from_ranked_rows(rows: list[dict[str, Any]], key: str, top_ks: tuple[int, ...]) -> dict[str, float]:
    output: dict[str, float] = {}
    for k in top_ks:
        hits: list[bool] = []
        ndcgs: list[float] = []
        for row in rows:
            ranked = np.asarray(row[key][:k], dtype=np.int64)
            pos = np.where(ranked == int(row["true_video_id"]))[0]
            hits.append(len(pos) > 0)
            ndcgs.append(0.0 if len(pos) == 0 else 1.0 / math.log2(int(pos[0]) + 2))
        output[f"hitrate@{k}"] = float(np.mean(hits)) if hits else 0.0
        output[f"ndcg@{k}"] = float(np.mean(ndcgs)) if ndcgs else 0.0
    return output


def segment(rows: list[dict[str, Any]], predicate) -> list[dict[str, Any]]:
    return [row for row in rows if predicate(row)]


def score_replay_rows(
    artifacts: LoadedAdaptiveArtifacts,
    query_frame: pd.DataFrame,
    item_ids: np.ndarray,
    item_emb: np.ndarray,
    retrieval_candidates: int,
    query_batch_size: int,
) -> list[dict[str, Any]]:
    user_emb, v4_idx, v4_scores = retrieve_for_queries(
        artifacts,
        query_frame,
        item_ids,
        item_emb,
        retrieval_candidates,
        query_batch_size,
    )
    v5_scores = rerank_with_v5(artifacts, user_emb, item_emb, v4_idx, v4_scores, query_batch_size)
    rows: list[dict[str, Any]] = []
    for row_idx, row in query_frame.reset_index(drop=True).iterrows():
        v4_order = np.argsort(-v4_scores[row_idx])
        v5_order = np.argsort(-v5_scores[row_idx])
        v4_ranked = item_ids[v4_idx[row_idx][v4_order]].tolist()
        v5_ranked = item_ids[v4_idx[row_idx][v5_order]].tolist()
        rows.append(
            {
                "example_id": row.get("example_id"),
                "user_id": int(row["user_id"]),
                "true_video_id": int(row["video_id"]),
                "query_time_ms": int(row["query_time_ms"]),
                "history_len": int(row.get("recent_event_count", 0) or 0),
                "item_hist_events": float(row.get("item_hist_events", 0) or 0),
                "shift_score": abs(float(row.get("session_vs_user_long_view_delta", 0) or 0))
                + abs(float(row.get("session_vs_user_watch_ratio_delta", 0) or 0)),
                "v4_ranked": v4_ranked,
                "v5_ranked": v5_ranked,
            }
        )
    return rows


def build_comparison(rows: list[dict[str, Any]], top_ks: tuple[int, ...]) -> dict[str, Any]:
    v4_metrics = metric_from_ranked_rows(rows, "v4_ranked", top_ks)
    v5_metrics = metric_from_ranked_rows(rows, "v5_ranked", top_ks)
    segments = {
        "short_history": segment(rows, lambda r: r["history_len"] < 5),
        "long_history": segment(rows, lambda r: r["history_len"] >= 20),
        "cold_item_proxy": segment(rows, lambda r: r["item_hist_events"] <= 0),
        "warm_item_proxy": segment(rows, lambda r: r["item_hist_events"] > 0),
        "recent_preference_shift_proxy": segment(rows, lambda r: r["shift_score"] >= 0.1),
        "stable_preference_proxy": segment(rows, lambda r: r["shift_score"] < 0.1),
    }
    return {
        "v4_adaptive_retrieval": v4_metrics,
        "v4_plus_v5_ranker": v5_metrics,
        "ranker_delta_vs_v4": {key: v5_metrics[key] - v4_metrics.get(key, 0.0) for key in v5_metrics},
        "evaluated_queries": len(rows),
        "segments": {
            name: {
                "queries": len(seg_rows),
                "v4": metric_from_ranked_rows(seg_rows, "v4_ranked", top_ks) if seg_rows else {},
                "v4_v5": metric_from_ranked_rows(seg_rows, "v5_ranked", top_ks) if seg_rows else {},
            }
            for name, seg_rows in segments.items()
        },
    }


def main() -> None:
    args = parse_args()
    validate_inputs(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    top_ks = parse_int_tuple(args.top_ks)
    max_queries = None if args.max_replay_queries == 0 else args.max_replay_queries
    max_users = None if args.max_replay_users == 0 else args.max_replay_users
    max_candidate_items = None if args.max_candidate_items == 0 else args.max_candidate_items
    device = select_device(args.device)
    print("device:", device)

    artifacts = load_artifacts(args.adaptive_root, args.base_gold_root, args.v4_dir, args.v5_dir, device=device)
    if artifacts.v5 is None:
        raise FileNotFoundError(f"V5 checkpoint not found in {args.v5_dir}")

    query_frame = load_replay_queries(args.adaptive_root, args.split, max_queries, max_users)
    candidate_pool = prepare_item_pool(args.adaptive_root, args.split, max_candidate_items)
    relevance_frame = load_relevance_frame(args.adaptive_root, args.split)
    print("replay queries / candidates:", query_frame.shape, candidate_pool.shape)

    item_ids, item_emb = encode_items(
        artifacts.v4,
        candidate_pool,
        query_frame.iloc[0].to_dict(),
        artifacts,
        batch_size=args.item_batch_size,
    )
    replay_rows = score_replay_rows(artifacts, query_frame, item_ids, item_emb, args.retrieval_candidates, args.query_batch_size)
    comparison = build_comparison(replay_rows, top_ks)

    # Also save user-level multi-positive metrics for the same V4 candidate order.
    # V5 multi-positive is handled through exact target reranking above; exact replay
    # is the strict production-style next-positive simulation.
    user_emb, v4_idx, _ = retrieve_for_queries(artifacts, query_frame, item_ids, item_emb, max(top_ks), args.query_batch_size)
    comparison["v4_multi_positive_reference"] = multi_positive_metrics(
        item_ids,
        v4_idx,
        query_frame["user_id"].to_numpy("int64"),
        relevance_frame,
        top_ks,
    )
    comparison["v4_exact_next_reference"] = exact_next_metrics(
        item_ids,
        v4_idx,
        query_frame["video_id"].to_numpy("int64"),
        top_ks,
    )
    if args.v3_metrics.exists():
        comparison["v3_reference"] = json.loads(args.v3_metrics.read_text())

    save_json(args.output_dir / "online_model_comparison.json", comparison)
    report_rows = [{k: v for k, v in row.items() if k not in {"v4_ranked", "v5_ranked"}} for row in replay_rows[: args.max_report_rows]]
    pd.DataFrame(report_rows).to_csv(args.output_dir / "online_replay_rows.csv", index=False)
    summary = {
        "split": args.split,
        "replay_queries": int(len(query_frame)),
        "users": int(query_frame["user_id"].nunique()),
        "sampling": "deterministic_even_users_and_timeline",
        "candidate_items": int(len(item_ids)),
        "retrieval_candidates": int(args.retrieval_candidates),
        "short_history_queries": int((query_frame["recent_event_count"] < 5).sum()),
        "long_history_queries": int((query_frame["recent_event_count"] >= 20).sum()),
    }
    save_json(args.output_dir / "online_replay_summary.json", summary)
    print(json.dumps(comparison, indent=2)[:6000])
    print("wrote:", args.output_dir)


if __name__ == "__main__":
    main()
