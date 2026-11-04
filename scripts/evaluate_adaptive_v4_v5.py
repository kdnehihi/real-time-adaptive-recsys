from __future__ import annotations

import argparse
import json
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
    top_indices,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate local V4/V5 adaptive checkpoints without retraining.")
    parser.add_argument("--adaptive-root", type=Path, default=Path("data/gold/adaptive/v1"))
    parser.add_argument("--base-gold-root", type=Path, default=Path("data/gold/two_tower/v1_colab"))
    parser.add_argument("--v4-dir", type=Path, default=Path("data/artifacts/adaptive/v4_sequence_retrieval"))
    parser.add_argument("--v5-dir", type=Path, default=Path("data/artifacts/adaptive/v5_adaptive_ranker"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/artifacts/adaptive/evaluation"))
    parser.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    parser.add_argument("--max-queries", type=int, default=0, help="Latest-user queries to evaluate; use 0 for all users.")
    parser.add_argument("--max-candidate-items", type=int, default=300000, help="Candidate items to score; use 0 for all.")
    parser.add_argument("--retrieval-candidates", type=int, default=1000)
    parser.add_argument("--top-ks", default="10,50,100")
    parser.add_argument("--popularity-blend-grid", default="0.0,0.02,0.05,0.08")
    parser.add_argument("--v5-candidate-blend-weight", type=float, default=0.0)
    parser.add_argument("--item-batch-size", type=int, default=8192)
    parser.add_argument("--query-batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def parse_int_tuple(raw: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in raw.split(",") if x.strip())


def parse_float_tuple(raw: str) -> tuple[float, ...]:
    return tuple(float(x.strip()) for x in raw.split(",") if x.strip())


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
        raise FileNotFoundError("Missing required evaluation inputs:\n- " + "\n- ".join(missing))


def load_query_frame(adaptive_root: Path, split: str, max_queries: int | None) -> pd.DataFrame:
    frame = read_parquet(
        adaptive_root / "sequence_examples" / split,
        INPUT_COLUMNS,
        None,
        ds.field("target_class") == "STRONG_POSITIVE",
    ).reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"No STRONG_POSITIVE rows found in {adaptive_root / 'sequence_examples' / split}")
    frame = sanitize_point_in_time_sequences(frame)
    frame = frame.sort_values(["user_id", "as_of_time", "example_id"]).drop_duplicates("user_id", keep="last")
    if max_queries is not None:
        frame = frame.head(max_queries)
    return frame.reset_index(drop=True)


def load_relevance_frame(adaptive_root: Path, split: str) -> pd.DataFrame:
    """Load every strong-positive item so user-level Recall/NDCG has full relevance."""
    return read_parquet(
        adaptive_root / "sequence_examples" / split,
        ["user_id", "video_id"],
        None,
        ds.field("target_class") == "STRONG_POSITIVE",
    ).drop_duplicates(["user_id", "video_id"]).reset_index(drop=True)


def compare_rankings(
    item_ids: np.ndarray,
    v4_idx: np.ndarray,
    v5_scores: np.ndarray,
    query_frame: pd.DataFrame,
    validation_positive_frame: pd.DataFrame,
    top_ks: tuple[int, ...],
) -> dict[str, dict[str, float]]:
    true_items = query_frame["video_id"].to_numpy("int64")
    user_ids = query_frame["user_id"].to_numpy("int64")
    v5_order = np.argsort(-v5_scores, axis=1)
    v5_ranked_idx = np.take_along_axis(v4_idx, v5_order, axis=1)
    return {
        "v4_exact_next": exact_next_metrics(item_ids, v4_idx, true_items, top_ks),
        "v4_multi_positive": multi_positive_metrics(item_ids, v4_idx, user_ids, validation_positive_frame, top_ks),
        "v5_exact_next": exact_next_metrics(item_ids, v5_ranked_idx, true_items, top_ks),
        "v5_multi_positive": multi_positive_metrics(item_ids, v5_ranked_idx, user_ids, validation_positive_frame, top_ks),
    }


def main() -> None:
    args = parse_args()
    validate_inputs(args)
    top_ks = parse_int_tuple(args.top_ks)
    blend_grid = parse_float_tuple(args.popularity_blend_grid)
    max_queries = None if args.max_queries == 0 else args.max_queries
    max_candidate_items = None if args.max_candidate_items == 0 else args.max_candidate_items
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = select_device(args.device)
    print("device:", device)
    print("loading artifacts...")
    artifacts = load_artifacts(args.adaptive_root, args.base_gold_root, args.v4_dir, args.v5_dir, device=device)

    print("loading queries/items...")
    query_frame = load_query_frame(args.adaptive_root, args.split, max_queries)
    candidate_pool = prepare_item_pool(args.adaptive_root, args.split, max_candidate_items)
    # Relevance for multi-positive metric. Keep full split when feasible; for local smoke runs
    # this can be limited externally by setting max-queries high enough.
    relevance_frame = load_relevance_frame(args.adaptive_root, args.split)

    print("encoding item pool:", candidate_pool.shape)
    item_ids, item_emb = encode_items(
        artifacts.v4,
        candidate_pool,
        query_frame.iloc[0].to_dict(),
        artifacts,
        batch_size=args.item_batch_size,
    )
    print("retrieving candidates:", query_frame.shape, item_emb.shape)
    user_emb, v4_idx, v4_scores = retrieve_for_queries(
        artifacts,
        query_frame,
        item_ids,
        item_emb,
        args.retrieval_candidates,
        args.query_batch_size,
    )

    raw_metrics = {
        "exact_next": exact_next_metrics(item_ids, v4_idx, query_frame["video_id"].to_numpy("int64"), top_ks),
        "multi_positive": multi_positive_metrics(item_ids, v4_idx, query_frame["user_id"].to_numpy("int64"), relevance_frame, top_ks),
    }

    blend_results = []
    popularity = pd.to_numeric(candidate_pool.get("item_hist_events", pd.Series(np.zeros(len(candidate_pool)))), errors="coerce").fillna(0).to_numpy("float32")
    popularity = np.log1p(np.maximum(popularity, 0.0))
    popularity = (popularity - popularity.mean()) / (popularity.std() + 1e-6)
    popularity = np.nan_to_num(popularity, nan=0.0, posinf=0.0, neginf=0.0)
    user_emb_for_blend = user_emb
    for weight in blend_grid:
        blended_idx_parts = []
        blended_score_parts = []
        for start in range(0, len(user_emb_for_blend), args.query_batch_size):
            scores = np.nan_to_num(user_emb_for_blend[start : start + args.query_batch_size] @ item_emb.T, nan=-1e4, posinf=1e4, neginf=-1e4)
            scores = scores + float(weight) * popularity[None, :]
            idx, vals = top_indices(scores, max(top_ks))
            blended_idx_parts.append(idx)
            blended_score_parts.append(vals)
        idx = np.vstack(blended_idx_parts)
        metrics = {
            "popularity_blend_weight": float(weight),
            "exact_next": exact_next_metrics(item_ids, idx, query_frame["video_id"].to_numpy("int64"), top_ks),
            "multi_positive": multi_positive_metrics(item_ids, idx, query_frame["user_id"].to_numpy("int64"), relevance_frame, top_ks),
        }
        blend_results.append(metrics)

    serving_idx = v4_idx
    serving_scores = v4_scores
    serving_user_emb = user_emb
    if args.v5_candidate_blend_weight != 0.0:
        serving_user_emb, serving_idx, serving_scores = retrieve_for_queries(
            artifacts,
            query_frame,
            item_ids,
            item_emb,
            args.retrieval_candidates,
            args.query_batch_size,
            item_score_bias=float(args.v5_candidate_blend_weight) * popularity,
        )
    print("reranking with V5...")
    v5_scores = rerank_with_v5(
        artifacts,
        serving_user_emb,
        item_emb,
        serving_idx,
        serving_scores,
        args.query_batch_size,
    )
    v5_metrics = compare_rankings(item_ids, serving_idx, v5_scores, query_frame, relevance_frame, top_ks)
    ranker_delta = {
        metric: float(v5_metrics["v5_multi_positive"].get(metric, 0.0) - v5_metrics["v4_multi_positive"].get(metric, 0.0))
        for metric in v5_metrics["v5_multi_positive"]
        if metric.startswith(("hitrate@", "recall@", "ndcg@"))
    }

    summary = {
        "protocol": {
            "split": args.split,
            "query_unit": "latest_strong_positive_per_user",
            "queries": int(len(query_frame)),
            "candidate_items": int(len(item_ids)),
            "retrieval_candidates": int(args.retrieval_candidates),
            "v5_candidate_blend_weight": float(args.v5_candidate_blend_weight),
            "top_ks": list(top_ks),
        },
        "v4_raw": raw_metrics,
        "v4_blend_grid": blend_results,
        "v5_same_candidates": v5_metrics,
        "ranker_delta_vs_v4_multi_positive": ranker_delta,
    }
    save_json(args.output_dir / "adaptive_v4_v5_evaluation.json", summary)
    print(json.dumps(summary, indent=2)[:6000])
    print("wrote:", args.output_dir / "adaptive_v4_v5_evaluation.json")


if __name__ == "__main__":
    main()
