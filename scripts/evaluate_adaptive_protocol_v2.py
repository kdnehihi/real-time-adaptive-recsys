from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import torch

from recommender.models.adaptive.checkpoint import (
    INPUT_COLUMNS,
    ITEM_CATEGORICAL,
    ITEM_ID_FEATURES,
    ITEM_NUMERIC,
    SEQUENCE_COLUMNS,
    encode_items,
    encode_users,
    load_artifacts,
    parquet_files,
    read_parquet,
    sanitize_point_in_time_sequences,
    save_json,
    select_device,
)

DAY_MS = 86_400_000
ITEM_DYNAMIC = (
    "upload_age_days_at_event",
    "item_hist_events",
    "item_hist_long_view_rate",
    "item_hist_like_rate",
    "item_hist_hate_rate",
    "item_hist_avg_watch_ratio",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate protocol-v2 exact next-item retrieval without retraining.")
    parser.add_argument("--adaptive-root", type=Path, required=True)
    parser.add_argument("--base-gold-root", type=Path, required=True)
    parser.add_argument("--v4-dir", type=Path, required=True)
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="validation", choices=["validation"])
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--top-ks", default="10,50,100,500,1000")
    parser.add_argument("--popularity-blend-weight", type=float, default=0.02)
    parser.add_argument("--item-batch-size", type=int, default=8192)
    parser.add_argument("--query-batch-size", type=int, default=16)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def parse_ints(raw: str) -> tuple[int, ...]:
    return tuple(sorted({int(value.strip()) for value in raw.split(",") if value.strip()}))


def session_history_bucket(value: object) -> str:
    """Bucket the session sequence using its actual Gold cap of ten events."""
    count = max(int(value or 0), 0)
    if count == 0:
        return "empty"
    if count <= 2:
        return "short_1_2"
    if count <= 5:
        return "medium_3_5"
    return "long_6_10"


def read_shard(path: Path, columns: list[str], retries: int = 5) -> pd.DataFrame:
    """Read one Drive-backed Parquet shard and close its handle deterministically."""
    for attempt in range(1, retries + 1):
        try:
            with path.open("rb") as stream:
                table = pq.ParquetFile(stream).read(columns=columns, use_threads=False)
            return table.to_pandas()
        except (OSError, IOError) as exc:
            gc.collect()
            if attempt == retries:
                raise OSError(f"Failed reading {path.name} after {retries} attempts") from exc
            delay = min(2 ** (attempt - 1), 8)
            print(f"Drive retry {attempt}/{retries}: {path.name}; sleeping {delay}s", flush=True)
            time.sleep(delay)
    raise AssertionError("unreachable")


def _reduce_latest(current: pd.DataFrame | None, incoming: pd.DataFrame) -> pd.DataFrame:
    """Keep the latest safe feature row and earliest estimated availability per item."""
    frame = incoming if current is None else pd.concat([current, incoming], ignore_index=True)
    earliest = frame.groupby("video_id", sort=False)["eligible_from_ms"].min()
    latest = (
        frame.sort_values(["current_time_ms", "video_id"])
        .drop_duplicates("video_id", keep="last")
        .reset_index(drop=True)
    )
    latest["eligible_from_ms"] = latest["video_id"].map(earliest).astype("int64")
    return latest


def load_item_snapshot(
    path: Path,
    candidate_ids: set[int] | None = None,
    progress_label: str = "items",
) -> pd.DataFrame:
    """Build a bounded latest-per-item snapshot without loading a full split at once."""
    columns = list(dict.fromkeys([*ITEM_ID_FEATURES, *ITEM_CATEGORICAL, *ITEM_NUMERIC, "current_time_ms"]))
    accumulator: pd.DataFrame | None = None
    files = parquet_files(path)
    for index, file_path in enumerate(files, start=1):
        part = read_shard(file_path, columns)
        part["video_id"] = pd.to_numeric(part["video_id"], errors="coerce")
        part = part.dropna(subset=["video_id", "current_time_ms"])
        part["video_id"] = part["video_id"].astype("int64")
        if candidate_ids is not None:
            part = part[part["video_id"].isin(candidate_ids)]
        if not part.empty:
            event_ms = pd.to_numeric(part["current_time_ms"], errors="coerce").fillna(0).astype("int64")
            age_days = pd.to_numeric(part["upload_age_days_at_event"], errors="coerce")
            estimated = (event_ms // DAY_MS) * DAY_MS - age_days.fillna(0).clip(lower=0).astype("int64") * DAY_MS
            part["eligible_from_ms"] = np.where(age_days.notna(), estimated, event_ms).astype("int64")
            part = _reduce_latest(None, part)
            accumulator = _reduce_latest(accumulator, part)
        if index % 16 == 0 or index == len(files):
            count = 0 if accumulator is None else len(accumulator)
            print(f"{progress_label}: {index}/{len(files)} shards; {count:,} items", flush=True)
    if accumulator is None or accumulator.empty:
        raise ValueError(f"No item rows loaded from {path}")
    return accumulator.reset_index(drop=True)


def build_safe_candidate_pool(
    adaptive_root: Path,
    split: str,
    train_source: Path | None = None,
) -> pd.DataFrame:
    """Use validation metadata, but only train-cutoff dynamic history features."""
    validation = load_item_snapshot(adaptive_root / "sequence_examples" / split, progress_label="catalog")
    candidate_ids = set(validation["video_id"].astype("int64"))
    train = load_item_snapshot(
        train_source or (adaptive_root / "sequence_examples/train"),
        candidate_ids=candidate_ids,
        progress_label="train snapshots",
    )
    train = train[["video_id", *ITEM_DYNAMIC]].rename(
        columns={column: f"train_{column}" for column in ITEM_DYNAMIC}
    )
    pool = validation.merge(train, on="video_id", how="left", validate="one_to_one")
    pool["target_warm"] = pool["train_item_hist_events"].notna()
    for column in ITEM_DYNAMIC:
        pool[column] = pd.to_numeric(pool[f"train_{column}"], errors="coerce").fillna(0.0)
        pool = pool.drop(columns=f"train_{column}")
    return pool.reset_index(drop=True)


def load_protocol_queries(
    adaptive_root: Path,
    protocol_dir: Path,
    split: str,
    maximum: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join fixed query IDs to full features, sanitize, and retain protocol metadata."""
    protocol = pd.read_parquet(protocol_dir / "queries.parquet")
    if maximum is not None:
        protocol = protocol.sort_values(["user_id", "as_of_time", "example_id"]).head(maximum)
    ids = protocol["example_id"].tolist()
    features = read_parquet(
        adaptive_root / "sequence_examples" / split,
        INPUT_COLUMNS,
        None,
        ds.field("example_id").isin(ids),
    )
    features = sanitize_point_in_time_sequences(features)
    features = features.drop_duplicates("example_id", keep="last")
    if set(features["example_id"]) != set(ids):
        missing = len(set(ids) - set(features["example_id"]))
        raise RuntimeError(f"Protocol query feature join missed {missing} example IDs")
    order = {value: index for index, value in enumerate(ids)}
    features["_protocol_order"] = features["example_id"].map(order)
    features = features.sort_values("_protocol_order").drop(columns="_protocol_order").reset_index(drop=True)
    protocol = protocol.set_index("example_id").loc[features["example_id"]].reset_index()
    if not np.array_equal(features["video_id"].to_numpy(), protocol["video_id"].to_numpy()):
        raise AssertionError("Protocol targets disagree with source features")
    protocol["session_history_bucket_v2"] = protocol["session_recent_event_count"].map(session_history_bucket)
    return features, protocol


def load_event_history(
    adaptive_root: Path,
    split: str,
    user_ids: set[int],
    max_time_ms: int,
    train_source: Path | None = None,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Load strict prior events from train plus the evaluation split for seen filtering."""
    parts: list[pd.DataFrame] = []
    columns = ["user_id", "video_id", "current_time_ms"]
    sources = (
        ("train", train_source or (adaptive_root / "sequence_examples/train")),
        (split, adaptive_root / "sequence_examples" / split),
    )
    for source_split, source_path in sources:
        files = parquet_files(source_path)
        for index, file_path in enumerate(files, start=1):
            part = read_shard(file_path, columns)
            part = part[part["user_id"].isin(user_ids)]
            part = part[pd.to_numeric(part["current_time_ms"], errors="coerce") < max_time_ms]
            parts.append(part)
            if index % 24 == 0 or index == len(files):
                print(f"seen history {source_split}: {index}/{len(files)} shards", flush=True)
    events = pd.concat(parts, ignore_index=True).dropna()
    events[["user_id", "video_id", "current_time_ms"]] = events[
        ["user_id", "video_id", "current_time_ms"]
    ].astype("int64")
    events = events.sort_values(["user_id", "current_time_ms", "video_id"])
    output: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for user_id, group in events.groupby("user_id", sort=False):
        output[int(user_id)] = (
            group["current_time_ms"].to_numpy("int64"),
            group["video_id"].to_numpy("int64"),
        )
    return output


def prior_seen_items(
    history: dict[int, tuple[np.ndarray, np.ndarray]],
    user_id: int,
    query_time_ms: int,
) -> set[int]:
    times, items = history.get(user_id, (np.asarray([], dtype="int64"), np.asarray([], dtype="int64")))
    stop = int(np.searchsorted(times, query_time_ms, side="left"))
    return set(int(value) for value in items[:stop])


def make_static_ablation(queries: pd.DataFrame) -> pd.DataFrame:
    """Remove only recent-event inputs while preserving long-term point-in-time state."""
    static = queries.copy(deep=True)
    array_columns = [column for column in SEQUENCE_COLUMNS if column.startswith("recent_") and column != "recent_event_count"]
    for column in array_columns:
        static[column] = [[] for _ in range(len(static))]
    static["recent_event_count"] = 0
    static["seconds_since_last_event"] = 0.0
    return static


def masked_exact_retrieval(
    adaptive_user_embeddings: np.ndarray,
    static_user_embeddings: np.ndarray,
    item_embeddings: np.ndarray,
    item_ids: np.ndarray,
    eligible_from_ms: np.ndarray,
    popularity: np.ndarray,
    protocol: pd.DataFrame,
    history: dict[int, tuple[np.ndarray, np.ndarray]],
    top_k: int,
    batch_size: int,
    blend_weight: float,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, int], np.ndarray]:
    """Apply identical time/seen masks before ranking every retrieval variant."""
    item_tensor = torch.as_tensor(item_embeddings, dtype=torch.float32, device=device)
    eligible_tensor = torch.as_tensor(eligible_from_ms, dtype=torch.int64, device=device)
    popularity_tensor = torch.as_tensor(popularity, dtype=torch.float32, device=device)
    item_to_index = {int(value): index for index, value in enumerate(item_ids)}
    variants = ("popularity", "v4_static_ablation", "v4_adaptive", "v4_adaptive_popularity")
    ranked_parts: dict[str, list[np.ndarray]] = {name: [] for name in variants}
    diagnostics = {"target_missing_catalog": 0, "target_time_override": 0, "repeat_target_override": 0}
    full_history_repeat = np.zeros(len(protocol), dtype=bool)
    k = min(top_k, len(item_ids))

    for start in range(0, len(protocol), batch_size):
        end = min(start + batch_size, len(protocol))
        part = protocol.iloc[start:end]
        mask = eligible_from_ms[None, :] > part["query_time_ms"].to_numpy("int64")[:, None]
        for row_offset, row in enumerate(part.itertuples(index=False)):
            target_index = item_to_index.get(int(row.video_id))
            if target_index is None:
                diagnostics["target_missing_catalog"] += 1
                continue
            if mask[row_offset, target_index]:
                diagnostics["target_time_override"] += 1
            seen = prior_seen_items(history, int(row.user_id), int(row.query_time_ms))
            seen_indices = [item_to_index[item] for item in seen if item in item_to_index]
            if seen_indices:
                mask[row_offset, seen_indices] = True
            if int(row.video_id) in seen:
                diagnostics["repeat_target_override"] += 1
                full_history_repeat[start + row_offset] = True
            mask[row_offset, target_index] = False
        mask_tensor = torch.as_tensor(mask, dtype=torch.bool, device=device)

        adaptive = torch.as_tensor(adaptive_user_embeddings[start:end], dtype=torch.float32, device=device) @ item_tensor.T
        static = torch.as_tensor(static_user_embeddings[start:end], dtype=torch.float32, device=device) @ item_tensor.T
        score_map = {
            "popularity": popularity_tensor.expand(end - start, -1),
            "v4_static_ablation": static,
            "v4_adaptive": adaptive,
            "v4_adaptive_popularity": adaptive + float(blend_weight) * popularity_tensor,
        }
        for name, scores in score_map.items():
            masked = scores.masked_fill(mask_tensor, -torch.inf)
            indices = torch.topk(masked, k=k, dim=1, largest=True, sorted=True).indices.cpu().numpy()
            ranked_parts[name].append(item_ids[indices])
        del adaptive, static, mask_tensor, score_map
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if end % (batch_size * 20) == 0 or end == len(protocol):
            print(f"retrieved queries: {end}/{len(protocol)}", flush=True)
    return {name: np.vstack(parts) for name, parts in ranked_parts.items()}, diagnostics, full_history_repeat


def per_query_metrics(ranked_ids: np.ndarray, protocol: pd.DataFrame, top_ks: tuple[int, ...]) -> pd.DataFrame:
    """Compute exact-target metrics for each query independently."""
    rows: list[dict[str, object]] = []
    for index, query in protocol.reset_index(drop=True).iterrows():
        target = int(query["video_id"])
        ranked = ranked_ids[index]
        positions = np.flatnonzero(ranked == target)
        rank = int(positions[0]) + 1 if len(positions) else None
        row: dict[str, object] = {
            "example_id": query["example_id"],
            "user_id": int(query["user_id"]),
            "target_video_id": target,
            "target_warm": bool(query["target_warm"]),
            "target_repeated": bool(query["target_repeated_full_history"]),
            "timeline_bucket": query["timeline_bucket"],
            "session_history_bucket": query["session_history_bucket_v2"],
            "exact_rank": np.nan if rank is None else rank,
        }
        for k in top_ks:
            hit = float(rank is not None and rank <= k)
            row[f"hitrate@{k}"] = hit
            row[f"ndcg@{k}"] = 0.0 if not hit else 1.0 / math.log2(rank + 1)
            row[f"mrr@{k}"] = 0.0 if not hit else 1.0 / rank
        rows.append(row)
    return pd.DataFrame(rows)


def metric_columns(top_ks: tuple[int, ...]) -> list[str]:
    return [f"{metric}@{k}" for k in top_ks for metric in ("hitrate", "ndcg", "mrr")]


def user_macro_summary(frame: pd.DataFrame, metrics: list[str]) -> dict[str, float | int]:
    per_user = frame.groupby("user_id")[metrics].mean()
    output: dict[str, float | int] = {
        "queries": int(len(frame)),
        "users": int(frame["user_id"].nunique()),
    }
    output.update({metric: float(per_user[metric].mean()) for metric in metrics})
    return output


def clustered_bootstrap(
    frame: pd.DataFrame,
    metrics: list[str],
    samples: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    """Bootstrap user-level metric means so repeated queries stay clustered."""
    per_user = frame.groupby("user_id")[metrics].mean().to_numpy("float64")
    estimate = per_user.mean(axis=0)
    rng = np.random.default_rng(seed)
    draws = np.empty((samples, len(metrics)), dtype=np.float64)
    for sample in range(samples):
        indices = rng.integers(0, len(per_user), size=len(per_user))
        draws[sample] = per_user[indices].mean(axis=0)
    return {
        metric: {
            "mean": float(estimate[index]),
            "ci95_low": float(np.quantile(draws[:, index], 0.025)),
            "ci95_high": float(np.quantile(draws[:, index], 0.975)),
        }
        for index, metric in enumerate(metrics)
    }


def segment_summary(frame: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    segment_columns = ("target_warm", "timeline_bucket", "session_history_bucket", "target_repeated")
    for column in segment_columns:
        for value, part in frame.groupby(column, dropna=False):
            rows.append({"segment_type": column, "segment": str(value), **user_macro_summary(part, metrics)})
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    top_ks = parse_ints(args.top_ks)
    maximum = None if args.max_queries == 0 else args.max_queries
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)
    print("device:", device, flush=True)

    required = [
        args.protocol_dir / "queries.parquet",
        args.adaptive_root / "sequence_examples/train",
        args.adaptive_root / "sequence_examples" / args.split,
        args.base_gold_root / "transforms/numeric_stats.json",
        args.v4_dir / "best_model.pt",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing A2 inputs:\n- " + "\n- ".join(missing))

    print("loading fixed protocol queries...", flush=True)
    query_features, protocol = load_protocol_queries(
        args.adaptive_root, args.protocol_dir, args.split, maximum
    )
    print("queries/users:", len(protocol), protocol["user_id"].nunique(), flush=True)

    print("building train-cutoff-safe candidate snapshot...", flush=True)
    candidate_pool = build_safe_candidate_pool(args.adaptive_root, args.split)
    print("candidate items:", len(candidate_pool), flush=True)

    print("loading V4 and train-fitted preprocessing...", flush=True)
    artifacts = load_artifacts(args.adaptive_root, args.base_gold_root, args.v4_dir, device=device)
    item_ids, item_embeddings = encode_items(
        artifacts.v4,
        candidate_pool,
        query_features.iloc[0].to_dict(),
        artifacts,
        batch_size=args.item_batch_size,
    )
    adaptive_users = encode_users(artifacts.v4, query_features, artifacts, batch_size=512)
    static_users = encode_users(
        artifacts.v4, make_static_ablation(query_features), artifacts, batch_size=512
    )

    popularity = np.log1p(np.maximum(candidate_pool["item_hist_events"].to_numpy("float32"), 0.0))
    popularity = np.nan_to_num(
        (popularity - popularity.mean()) / (popularity.std() + 1e-6),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype("float32")

    print("loading full strict-prior history for seen filtering...", flush=True)
    history = load_event_history(
        args.adaptive_root,
        args.split,
        set(protocol["user_id"].astype("int64")),
        int(protocol["query_time_ms"].max()) + 1,
    )
    ranked, diagnostics, full_history_repeat = masked_exact_retrieval(
        adaptive_users,
        static_users,
        item_embeddings,
        item_ids,
        candidate_pool["eligible_from_ms"].to_numpy("int64"),
        popularity,
        protocol,
        history,
        max(top_ks),
        args.query_batch_size,
        args.popularity_blend_weight,
        device,
    )
    protocol = protocol.copy()
    protocol["target_repeated_full_history"] = full_history_repeat

    metrics = metric_columns(top_ks)
    summaries: list[dict[str, object]] = []
    segments: list[pd.DataFrame] = []
    confidence: dict[str, dict[str, dict[str, float]]] = {}
    for variant, ranked_ids in ranked.items():
        per_query = per_query_metrics(ranked_ids, protocol, top_ks)
        per_query.insert(0, "variant", variant)
        per_query.to_parquet(args.output_dir / f"per_query_{variant}.parquet", index=False)
        summaries.append({"variant": variant, **user_macro_summary(per_query, metrics)})
        segment = segment_summary(per_query, metrics)
        segment.insert(0, "variant", variant)
        segments.append(segment)
        confidence[variant] = clustered_bootstrap(
            per_query, metrics, args.bootstrap_samples, args.seed
        )

    summary = pd.DataFrame(summaries)
    segment_frame = pd.concat(segments, ignore_index=True)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    segment_frame.to_csv(args.output_dir / "segments.csv", index=False)
    report = {
        "protocol_version": "adaptive_eval_v2_a2",
        "query_protocol_revision": str(protocol["builder_revision"].iloc[0]),
        "queries": int(len(protocol)),
        "users": int(protocol["user_id"].nunique()),
        "candidate_items": int(len(candidate_pool)),
        "candidate_policy": "validation observed catalog; upload-time eligible; train-cutoff item history; cold history zero",
        "seen_policy": "all strict-prior train+validation events excluded; repeated exact target restored",
        "relevance": "exact target at query time",
        "static_baseline": "same V4 with recent sequence removed; long-term state retained",
        "top_ks": list(top_ks),
        "popularity_blend_weight": float(args.popularity_blend_weight),
        "diagnostics": diagnostics,
        "summary": {row["variant"]: {k: v for k, v in row.items() if k != "variant"} for row in summaries},
        "clustered_user_bootstrap_ci95": confidence,
    }
    save_json(args.output_dir / "protocol_v2_evaluation.json", report)
    print(summary.to_string(index=False), flush=True)
    print("wrote:", args.output_dir, flush=True)


if __name__ == "__main__":
    main()
