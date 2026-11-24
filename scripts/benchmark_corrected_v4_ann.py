from __future__ import annotations

import argparse
import gc
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
)
from scripts.evaluate_adaptive_protocol_v2 import (
    build_safe_candidate_pool,
    load_event_history,
    load_protocol_queries,
    read_shard,
)


def parse_ints(raw: str) -> tuple[int, ...]:
    return tuple(sorted({int(value.strip()) for value in raw.split(",") if value.strip()}))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark ANN serving parity for fixed Corrected V4.2."
    )
    parser.add_argument("--adaptive-root", type=Path, required=True)
    parser.add_argument("--base-gold-root", type=Path, required=True)
    parser.add_argument("--corrected-root", type=Path, required=True)
    parser.add_argument("--corrected-v4-dir", type=Path, required=True)
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--author-weight", type=float, default=0.2)
    parser.add_argument("--popularity-weight", type=float, default=0.02)
    parser.add_argument("--recency-tau", type=float, default=3.0)
    parser.add_argument("--top-k", type=int, default=1000)
    parser.add_argument("--search-k", type=int, default=5000)
    parser.add_argument("--hnsw-m", type=int, default=32)
    parser.add_argument("--ef-construction", type=int, default=80)
    parser.add_argument("--ef-search-values", default="64,128,256")
    parser.add_argument("--calibration-user-fraction", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--query-batch-size", type=int, default=8)
    parser.add_argument("--item-batch-size", type=int, default=8192)
    parser.add_argument("--max-holdout-queries", type=int, default=0)
    parser.add_argument("--minimum-overlap-recall", type=float, default=0.95)
    parser.add_argument("--maximum-hitrate-drop", type=float, default=0.001)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def load_video_author_mapping(static_path: Path) -> dict[int, int]:
    """Load immutable item authors shard-by-shard without retaining Drive handles."""
    mapping: dict[int, int] = {}
    files = parquet_files(static_path)
    for index, path in enumerate(files, start=1):
        part = read_shard(path, list(STATIC_COLUMNS)).dropna(subset=["video_id", "author_id"])
        mapping.update(zip(part["video_id"].astype("int64"), part["author_id"].astype("int64")))
        if index % 16 == 0 or index == len(files):
            print(f"author metadata: {index}/{len(files)} shards; {len(mapping):,} videos", flush=True)
    return mapping


def normalized_popularity(candidate_pool: pd.DataFrame) -> np.ndarray:
    """Recreate the fixed V4.2 standardized log-popularity feature."""
    values = np.log1p(np.maximum(candidate_pool["item_hist_events"].to_numpy("float32"), 0.0))
    return np.nan_to_num((values - values.mean()) / (values.std() + 1e-6)).astype("float32")


def overlap_recall(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    """Return per-query fraction of exact Top-K indices recovered by ANN."""
    if reference.shape != candidate.shape:
        raise ValueError("Reference and candidate Top-K matrices must have the same shape")
    return np.asarray(
        [len(set(left.tolist()).intersection(right.tolist())) / len(left) for left, right in zip(reference, candidate)],
        dtype="float32",
    )


def target_hits(top_indices: np.ndarray, target_indices: np.ndarray) -> np.ndarray:
    """Return one exact-target hit flag per query."""
    return (top_indices == target_indices[:, None]).any(axis=1)


def select_ann_config(rows: list[dict[str, float | int]], min_recall: float, max_drop: float) -> dict[str, float | int] | None:
    """Choose the fastest configuration satisfying parity and quality tolerances."""
    eligible = [
        row for row in rows
        if float(row["mean_topk_overlap_recall"]) >= min_recall
        and float(row["target_hitrate_delta_vs_exact"]) >= -max_drop
        and int(row["queries_with_fewer_than_k"]) == 0
    ]
    if not eligible:
        return None
    return min(eligible, key=lambda row: (float(row["search_rerank_ms_per_query"]), -float(row["mean_topk_overlap_recall"])))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError("Install faiss-cpu before running the ANN benchmark") from exc

    device = select_device(args.device)
    legacy_queries, protocol = load_protocol_queries(
        args.adaptive_root, args.protocol_dir, "validation", None
    )
    corrected = corrected_queries(legacy_queries, args.corrected_root)
    holdout = ~deterministic_user_split(
        protocol["user_id"], args.calibration_user_fraction, args.seed
    )
    indices = np.flatnonzero(holdout)
    if args.max_holdout_queries > 0:
        rng = np.random.default_rng(args.seed)
        indices = np.sort(
            rng.choice(indices, size=min(args.max_holdout_queries, len(indices)), replace=False)
        )
    protocol = protocol.iloc[indices].reset_index(drop=True)
    corrected = corrected.iloc[indices].reset_index(drop=True)
    print("benchmark queries/users:", len(protocol), protocol["user_id"].nunique(), flush=True)

    candidate_pool = build_safe_candidate_pool(args.adaptive_root, "validation")
    video_to_author = load_video_author_mapping(args.base_gold_root / "item_features/static")
    history = load_event_history(
        args.adaptive_root,
        "validation",
        set(protocol["user_id"].astype("int64")),
        int(protocol["query_time_ms"].max()) + 1,
    )
    artifacts = load_artifacts(
        args.adaptive_root, args.base_gold_root, args.corrected_v4_dir, device=device
    )
    item_ids, item_embeddings = encode_items(
        artifacts.v4, candidate_pool, corrected.iloc[0].to_dict(), artifacts, args.item_batch_size
    )
    users = encode_users(
        artifacts.v4, corrected, artifacts, batch_size=max(args.query_batch_size, 32)
    )
    item_to_index = {int(value): offset for offset, value in enumerate(item_ids)}
    candidate_authors_np = np.asarray(
        [int(video_to_author.get(int(video), 0) or 0) for video in item_ids], dtype=np.int64
    )
    candidate_authors = torch.as_tensor(candidate_authors_np, dtype=torch.long, device=device)
    recent_authors = encode_raw_recent_authors(
        corrected, video_to_author, int(artifacts.v4_cfg["sequence_max_len"])
    )
    popularity = normalized_popularity(candidate_pool)
    eligible_from_ms = candidate_pool["eligible_from_ms"].to_numpy("int64")
    del artifacts
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("building HNSW index...", flush=True)
    index = faiss.IndexHNSWFlat(
        item_embeddings.shape[1], args.hnsw_m, faiss.METRIC_INNER_PRODUCT
    )
    index.hnsw.efConstruction = args.ef_construction
    build_start = time.perf_counter()
    index.add(np.ascontiguousarray(item_embeddings, dtype="float32"))
    build_seconds = time.perf_counter() - build_start
    index_path = args.output_dir / "corrected_v4_hnsw.index"
    faiss.write_index(index, str(index_path))
    print(f"HNSW built in {build_seconds:.1f}s; {index_path.stat().st_size / 1024**2:.1f} MiB", flush=True)

    ef_values = parse_ints(args.ef_search_values)
    exact_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    ann_parts: dict[int, list[np.ndarray]] = {ef: [] for ef in ef_values}
    ann_elapsed = {ef: 0.0 for ef in ef_values}
    short_rows = {ef: 0 for ef in ef_values}
    exact_elapsed = 0.0
    item_tensor = torch.as_tensor(item_embeddings, dtype=torch.float32, device=device)
    popularity_tensor = torch.as_tensor(popularity, dtype=torch.float32, device=device)

    for start in range(0, len(protocol), args.query_batch_size):
        end = min(start + args.query_batch_size, len(protocol))
        part = protocol.iloc[start:end]
        mask, targets, _ = build_query_mask(
            part, item_ids, eligible_from_ms, item_to_index, history
        )
        mask_tensor = torch.as_tensor(mask, dtype=torch.bool, device=device)
        user_tensor = torch.as_tensor(users[start:end], dtype=torch.float32, device=device)
        exact_start = time.perf_counter()
        scores = user_tensor @ item_tensor.T
        scores += args.author_weight * author_affinity_scores(
            recent_authors[start:end].to(device), candidate_authors, args.recency_tau
        )
        scores += args.popularity_weight * popularity_tensor
        scores = scores.masked_fill(mask_tensor, -torch.inf)
        exact = torch.topk(scores, k=args.top_k, dim=1).indices.cpu().numpy()
        exact_elapsed += time.perf_counter() - exact_start
        exact_parts.append(exact)
        target_parts.append(targets)

        query_vectors = np.ascontiguousarray(users[start:end], dtype="float32")
        for ef in ef_values:
            index.hnsw.efSearch = ef
            ann_start = time.perf_counter()
            _, raw = index.search(query_vectors, args.search_k)
            batch_top = np.full((len(part), args.top_k), -1, dtype=np.int64)
            for row in range(len(part)):
                candidates = raw[row]
                candidates = candidates[candidates >= 0]
                candidates = candidates[~mask[row, candidates]]
                if len(candidates):
                    unique = np.unique(candidates)
                    policy_scores = scores[row, torch.as_tensor(unique, device=device)].detach().cpu().numpy()
                    order = np.argsort(-policy_scores)[: args.top_k]
                    selected = unique[order]
                    batch_top[row, : len(selected)] = selected
                if (batch_top[row] < 0).any():
                    short_rows[ef] += 1
            ann_elapsed[ef] += time.perf_counter() - ann_start
            ann_parts[ef].append(batch_top)
        if end % (args.query_batch_size * 20) == 0 or end == len(protocol):
            print(f"benchmarked queries: {end}/{len(protocol)}", flush=True)

    exact_top = np.vstack(exact_parts)
    targets = np.concatenate(target_parts)
    exact_hit = target_hits(exact_top, targets)
    rows: list[dict[str, float | int]] = []
    for ef in ef_values:
        ann_top = np.vstack(ann_parts[ef])
        overlap = overlap_recall(exact_top, ann_top)
        ann_hit = target_hits(ann_top, targets)
        row: dict[str, float | int] = {
            "ef_search": ef,
            "search_k": args.search_k,
            "queries": len(protocol),
            "mean_topk_overlap_recall": float(overlap.mean()),
            "p05_topk_overlap_recall": float(np.quantile(overlap, 0.05)),
            "exact_target_hitrate": float(exact_hit.mean()),
            "ann_target_hitrate": float(ann_hit.mean()),
            "target_hitrate_delta_vs_exact": float(ann_hit.mean() - exact_hit.mean()),
            "search_rerank_ms_per_query": 1000.0 * ann_elapsed[ef] / len(protocol),
            "queries_with_fewer_than_k": short_rows[ef],
        }
        rows.append(row)
    selected = select_ann_config(
        rows, args.minimum_overlap_recall, args.maximum_hitrate_drop
    )
    table = pd.DataFrame(rows)
    table.to_csv(args.output_dir / "ann_benchmark.csv", index=False)
    report = {
        "protocol": "corrected_v4_2_ann_serving_benchmark_v1",
        "model_policy": {
            "author_weight": args.author_weight,
            "popularity_weight": args.popularity_weight,
            "recency_tau": args.recency_tau,
        },
        "queries": len(protocol),
        "users": int(protocol["user_id"].nunique()),
        "candidate_items": len(item_ids),
        "embedding_dim": int(item_embeddings.shape[1]),
        "top_k": args.top_k,
        "index": {
            "type": "HNSWFlat-IP",
            "m": args.hnsw_m,
            "ef_construction": args.ef_construction,
            "build_seconds": build_seconds,
            "bytes": index_path.stat().st_size,
        },
        "exact_ms_per_query": 1000.0 * exact_elapsed / len(protocol),
        "configs": rows,
        "acceptance": {
            "minimum_overlap_recall": args.minimum_overlap_recall,
            "maximum_hitrate_drop": args.maximum_hitrate_drop,
            "selected": selected,
            "passed": selected is not None,
        },
        "test_data_used": False,
    }
    save_json(args.output_dir / "corrected_v4_ann_benchmark.json", report)
    print(table.to_string(index=False), flush=True)
    print(json.dumps(report["acceptance"], indent=2), flush=True)


if __name__ == "__main__":
    main()
