from __future__ import annotations

import argparse
import gc
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
import torch

from recommender.models.adaptive.checkpoint import (
    encode_items,
    encode_users,
    load_artifacts,
    read_parquet,
    save_json,
    select_device,
)
from scripts.evaluate_adaptive_protocol_v2 import (
    build_safe_candidate_pool,
    clustered_bootstrap,
    load_event_history,
    load_protocol_queries,
    metric_columns,
    per_query_metrics,
    prior_seen_items,
    segment_summary,
    user_macro_summary,
)


CORRECTED_COLUMNS = (
    "current_time_ms",
    "recent_video_ids",
    "recent_target_classes",
    "recent_engagement_strengths",
    "recent_long_views",
    "recent_likes",
    "recent_hates",
    "recent_clicks",
    "recent_time_ms",
    "recent_event_count",
    "seconds_since_last_event",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare legacy and corrected V4 under one locked A2 protocol.")
    parser.add_argument("--adaptive-root", type=Path, required=True)
    parser.add_argument("--base-gold-root", type=Path, required=True)
    parser.add_argument("--corrected-root", type=Path, required=True)
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--legacy-v4-dir", type=Path, required=True)
    parser.add_argument("--corrected-v4-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-ks", default="10,50,100,500,1000")
    parser.add_argument("--popularity-blend-weight", type=float, default=0.02)
    parser.add_argument("--item-batch-size", type=int, default=8192)
    parser.add_argument("--query-batch-size", type=int, default=16)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def corrected_queries(
    legacy: pd.DataFrame,
    corrected_root: Path,
    split: str = "validation",
) -> pd.DataFrame:
    """Replace only sequence fields while preserving the locked query/features contract."""
    ids = legacy["example_id"].tolist()
    delta = read_parquet(
        corrected_root / "sequences" / f"split={split}",
        ["example_id", *CORRECTED_COLUMNS],
        filter_expr=ds.field("example_id").isin(ids),
    ).drop_duplicates("example_id", keep="last")
    if len(delta) != len(ids) or set(delta["example_id"]) != set(ids):
        raise RuntimeError("Corrected delta does not cover every locked A2 query")
    order = {value: index for index, value in enumerate(ids)}
    delta["_order"] = delta["example_id"].map(order)
    delta = delta.sort_values("_order").reset_index(drop=True)
    frame = legacy.copy(deep=True)
    if not np.array_equal(frame["example_id"].to_numpy(), delta["example_id"].to_numpy()):
        raise AssertionError("Corrected delta order mismatch")
    for column in CORRECTED_COLUMNS:
        frame[column] = delta[column].tolist()
    violations = sum(
        any(float(value) >= float(current) for value in (times if isinstance(times, (list, np.ndarray)) else []))
        for times, current in zip(frame["recent_time_ms"], frame["current_time_ms"])
    )
    if violations:
        raise AssertionError(f"Corrected A2 queries contain {violations} strict-prior violations")
    return frame


def retrieve_variants(
    embeddings_by_variant: dict[str, np.ndarray],
    item_embeddings: np.ndarray,
    item_ids: np.ndarray,
    candidate_pool: pd.DataFrame,
    protocol: pd.DataFrame,
    history,
    top_k: int,
    batch_size: int,
    popularity_weight: float,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Rank raw and popularity-blended scores with identical time/seen masks."""
    item_tensor = torch.as_tensor(item_embeddings, dtype=torch.float32, device=device)
    eligible = candidate_pool["eligible_from_ms"].to_numpy("int64")
    item_to_index = {int(value): index for index, value in enumerate(item_ids)}
    popularity = np.log1p(np.maximum(candidate_pool["item_hist_events"].to_numpy("float32"), 0))
    popularity = np.nan_to_num((popularity - popularity.mean()) / (popularity.std() + 1e-6)).astype("float32")
    popularity_tensor = torch.as_tensor(popularity, device=device)
    output: dict[str, list[np.ndarray]] = {}
    for name in embeddings_by_variant:
        output[name] = []
        output[f"{name}_popularity"] = []
    k = min(top_k, len(item_ids))
    for start in range(0, len(protocol), batch_size):
        end = min(start + batch_size, len(protocol))
        part = protocol.iloc[start:end]
        mask = eligible[None, :] > part["query_time_ms"].to_numpy("int64")[:, None]
        for offset, row in enumerate(part.itertuples(index=False)):
            target_index = item_to_index[int(row.video_id)]
            seen = prior_seen_items(history, int(row.user_id), int(row.query_time_ms))
            seen_indices = [item_to_index[item] for item in seen if item in item_to_index]
            if seen_indices:
                mask[offset, seen_indices] = True
            mask[offset, target_index] = False
        mask_tensor = torch.as_tensor(mask, dtype=torch.bool, device=device)
        for name, user_embeddings in embeddings_by_variant.items():
            users = torch.as_tensor(user_embeddings[start:end], dtype=torch.float32, device=device)
            raw = users @ item_tensor.T
            score_sets = {
                name: raw,
                f"{name}_popularity": raw + popularity_weight * popularity_tensor,
            }
            for variant, scores in score_sets.items():
                indices = torch.topk(scores.masked_fill(mask_tensor, -torch.inf), k=k, dim=1).indices
                output[variant].append(item_ids[indices.cpu().numpy()])
            del users, raw
        if end % (batch_size * 20) == 0 or end == len(protocol):
            print(f"retrieved queries: {end}/{len(protocol)}", flush=True)
    return {name: np.vstack(parts) for name, parts in output.items()}


def paired_bootstrap_delta(left: pd.DataFrame, right: pd.DataFrame, metrics, samples, seed):
    """Return right-minus-left user-clustered deltas on aligned exact queries."""
    left_user = left.groupby("user_id")[metrics].mean().sort_index()
    right_user = right.groupby("user_id")[metrics].mean().reindex(left_user.index)
    delta = right_user.to_numpy("float64") - left_user.to_numpy("float64")
    rng = np.random.default_rng(seed)
    draws = np.empty((samples, len(metrics)), dtype="float64")
    for index in range(samples):
        sample = rng.integers(0, len(delta), len(delta))
        draws[index] = delta[sample].mean(axis=0)
    return {
        metric: {
            "mean": float(delta[:, column].mean()),
            "ci95_low": float(np.quantile(draws[:, column], 0.025)),
            "ci95_high": float(np.quantile(draws[:, column], 0.975)),
        }
        for column, metric in enumerate(metrics)
    }


def annotate_target_repeats(protocol: pd.DataFrame, history) -> pd.DataFrame:
    """Add the full strict-prior repeat flag expected by shared metric code."""
    output = protocol.copy()
    output["target_repeated_full_history"] = [
        int(row.video_id) in prior_seen_items(history, int(row.user_id), int(row.query_time_ms))
        for row in output.itertuples(index=False)
    ]
    return output


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)
    top_ks = tuple(sorted({int(value) for value in args.top_ks.split(",")}))
    maximum = args.max_queries or None
    print("loading locked A2 queries...", flush=True)
    legacy_queries, protocol = load_protocol_queries(
        args.adaptive_root, args.protocol_dir, "validation", maximum
    )
    corrected = corrected_queries(legacy_queries, args.corrected_root)
    print("queries/users:", len(protocol), protocol["user_id"].nunique(), flush=True)
    candidate_pool = build_safe_candidate_pool(args.adaptive_root, "validation")
    history = load_event_history(
        args.adaptive_root,
        "validation",
        set(protocol["user_id"].astype("int64")),
        int(protocol["query_time_ms"].max()) + 1,
    )
    protocol = annotate_target_repeats(protocol, history)
    all_ranked: dict[str, np.ndarray] = {}
    cache_dir = args.output_dir / "ranked_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    for model_name, model_dir, query_map in (
        ("legacy_v4", args.legacy_v4_dir, {"legacy_v4_legacy_sequence": legacy_queries, "legacy_v4_corrected_sequence": corrected}),
        ("corrected_v4", args.corrected_v4_dir, {"corrected_v4_corrected_sequence": corrected}),
    ):
        expected_variants = [name for base in query_map for name in (base, f"{base}_popularity")]
        cached = {name: cache_dir / f"{name}.npy" for name in expected_variants}
        if all(path.exists() for path in cached.values()):
            print("loading cached rankings:", model_name, flush=True)
            all_ranked.update({name: np.load(path) for name, path in cached.items()})
            continue
        print("loading model:", model_name, flush=True)
        artifacts = load_artifacts(args.adaptive_root, args.base_gold_root, model_dir, device=device)
        item_ids, item_embeddings = encode_items(
            artifacts.v4,
            candidate_pool,
            corrected.iloc[0].to_dict(),
            artifacts,
            batch_size=args.item_batch_size,
        )
        query_embeddings = {
            name: encode_users(artifacts.v4, frame, artifacts, batch_size=512)
            for name, frame in query_map.items()
        }
        model_ranked = retrieve_variants(
                query_embeddings,
                item_embeddings,
                item_ids,
                candidate_pool,
                protocol,
                history,
                max(top_ks),
                args.query_batch_size,
                args.popularity_blend_weight,
                device,
        )
        for name, values in model_ranked.items():
            np.save(cached[name], values)
            print("cached rankings:", cached[name], flush=True)
        all_ranked.update(model_ranked)
        del artifacts, item_embeddings, query_embeddings
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    metrics = metric_columns(top_ks)
    frames: dict[str, pd.DataFrame] = {}
    summaries = []
    segments = []
    confidence = {}
    for variant, ranked in all_ranked.items():
        frame = per_query_metrics(ranked, protocol, top_ks)
        frame.insert(0, "variant", variant)
        frame.to_parquet(args.output_dir / f"per_query_{variant}.parquet", index=False)
        frames[variant] = frame
        summaries.append({"variant": variant, **user_macro_summary(frame, metrics)})
        segment = segment_summary(frame, metrics)
        segment.insert(0, "variant", variant)
        segments.append(segment)
        confidence[variant] = clustered_bootstrap(frame, metrics, args.bootstrap_samples, args.seed)
    summary = pd.DataFrame(summaries)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    pd.concat(segments, ignore_index=True).to_csv(args.output_dir / "segments.csv", index=False)

    selected_metrics = [name for name in metrics if name in {"hitrate@100", "hitrate@1000", "ndcg@1000", "mrr@1000"}]
    comparisons = {
        "legacy_input_shift": paired_bootstrap_delta(
            frames["legacy_v4_legacy_sequence_popularity"],
            frames["legacy_v4_corrected_sequence_popularity"],
            selected_metrics,
            args.bootstrap_samples,
            args.seed + 1,
        ),
        "corrected_retraining_delta": paired_bootstrap_delta(
            frames["legacy_v4_corrected_sequence_popularity"],
            frames["corrected_v4_corrected_sequence_popularity"],
            selected_metrics,
            args.bootstrap_samples,
            args.seed + 2,
        ),
    }
    report = {
        "protocol": "adaptive_eval_v2_a2_corrected_sequence_comparison",
        "queries": int(len(protocol)),
        "users": int(protocol["user_id"].nunique()),
        "candidate_items": int(len(candidate_pool)),
        "popularity_blend_weight": args.popularity_blend_weight,
        "summary": {row["variant"]: {key: value for key, value in row.items() if key != "variant"} for row in summaries},
        "clustered_user_bootstrap_ci95": confidence,
        "paired_deltas": comparisons,
    }
    save_json(args.output_dir / "corrected_v4_comparison.json", report)
    print(summary.to_string(index=False), flush=True)
    print(json.dumps(comparisons, indent=2), flush=True)


if __name__ == "__main__":
    main()
