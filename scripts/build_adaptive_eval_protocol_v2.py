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
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from recommender.models.adaptive.checkpoint import (
    as_sequence,
    parquet_files,
    read_parquet,
    sanitize_point_in_time_sequences,
    save_json,
)


HISTORY_BUCKET_ORDER = ["empty", "short_1_4", "medium_5_19", "long_20_plus"]
TIMELINE_BUCKET_ORDER = ["early", "middle", "late"]


def load_distinct_int64_column_sharded(
    path: Path,
    column: str,
    retries: int = 5,
) -> set[int]:
    """Read one column shard-by-shard to avoid exhausting Drive FUSE handles."""
    files = parquet_files(path)
    values: set[int] = set()
    for index, file_path in enumerate(files, start=1):
        for attempt in range(1, retries + 1):
            try:
                # A Python file context guarantees each remote FUSE handle closes
                # before the next shard is opened. Dataset.to_table may hold many
                # Drive files open concurrently and fail with transient stat errors.
                with file_path.open("rb") as stream:
                    table = pq.ParquetFile(stream).read(columns=[column], use_threads=False)
                shard_values = table.column(column).to_pandas().dropna().astype("int64").unique()
                values.update(int(value) for value in shard_values)
                del shard_values, table
                break
            except (OSError, IOError) as exc:
                gc.collect()
                if attempt == retries:
                    raise OSError(
                        f"Failed reading {column} from shard {file_path.name} after {retries} attempts"
                    ) from exc
                delay = min(2**(attempt - 1), 8)
                print(
                    f"Drive read retry {attempt}/{retries} for {file_path.name}; sleeping {delay}s",
                    flush=True,
                )
                time.sleep(delay)
        if index % 16 == 0 or index == len(files):
            print(f"train item shards: {index}/{len(files)}; distinct items: {len(values):,}", flush=True)
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build deterministic point-in-time evaluation queries for protocol v2."
    )
    parser.add_argument("--adaptive-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="validation", choices=["validation", "test"])
    parser.add_argument("--max-queries-per-user", type=int, default=12)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def history_bucket(value: object) -> str:
    count = int(value or 0)
    if count == 0:
        return "empty"
    if count < 5:
        return "short_1_4"
    if count < 20:
        return "medium_5_19"
    return "long_20_plus"


def timeline_bucket(value: float) -> str:
    if value < 1.0 / 3.0:
        return "early"
    if value < 2.0 / 3.0:
        return "middle"
    return "late"


def evenly_spaced_indices(length: int, count: int) -> list[int]:
    if length <= 0 or count <= 0:
        return []
    if length <= count:
        return list(range(length))
    return sorted(set(np.linspace(0, length - 1, num=count).round().astype(int).tolist()))


def select_user_queries(group: pd.DataFrame, maximum: int) -> list[object]:
    """Select session-history/timeline strata, then fill evenly over time."""
    ordered = group.sort_values(["as_of_time", "example_id"]).reset_index(drop=True)
    selected: list[object] = []

    def add(example_id: object) -> None:
        if example_id not in selected and len(selected) < maximum:
            selected.append(example_id)

    for bucket in HISTORY_BUCKET_ORDER:
        part = ordered[ordered["session_history_bucket"].eq(bucket)]
        if not part.empty:
            add(part.iloc[len(part) // 2]["example_id"])
    for bucket in TIMELINE_BUCKET_ORDER:
        part = ordered[ordered["timeline_bucket"].eq(bucket)]
        if not part.empty:
            add(part.iloc[len(part) // 2]["example_id"])

    remaining = ordered[~ordered["example_id"].isin(selected)].reset_index(drop=True)
    for index in evenly_spaced_indices(len(remaining), maximum - len(selected)):
        add(remaining.iloc[index]["example_id"])
    return selected


def build_index(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["as_of_time"] = pd.to_datetime(frame["as_of_time"], errors="coerce")
    frame = frame.dropna(subset=["example_id", "user_id", "video_id", "as_of_time"])
    frame["recent_event_count"] = (
        pd.to_numeric(frame["recent_event_count"], errors="coerce").fillna(0).clip(lower=0).astype("int64")
    )
    frame["session_recent_event_count"] = (
        pd.to_numeric(frame["session_recent_event_count"], errors="coerce")
        .fillna(0)
        .clip(lower=0)
        .astype("int64")
    )
    frame["session_seconds_since_last_event"] = pd.to_numeric(
        frame["session_seconds_since_last_event"], errors="coerce"
    )
    frame = frame.sort_values(["user_id", "as_of_time", "example_id"]).reset_index(drop=True)
    frame["positive_index"] = frame.groupby("user_id").cumcount()
    frame["positive_count"] = frame.groupby("user_id")["example_id"].transform("size")
    denominator = (frame["positive_count"] - 1).clip(lower=1)
    frame["timeline_fraction"] = frame["positive_index"] / denominator
    # recent_event_count is capped by the model sequence length (currently 30),
    # so it is useful model context but not a reliable lifecycle segment. Session
    # history varies naturally and is the primary adaptive-evaluation stratum.
    frame["global_history_bucket"] = frame["recent_event_count"].map(history_bucket)
    frame["session_history_bucket"] = frame["session_recent_event_count"].map(history_bucket)
    frame["timeline_bucket"] = frame["timeline_fraction"].map(timeline_bucket)
    return frame


def distribution(frame: pd.DataFrame, column: str) -> list[dict[str, object]]:
    counts = frame[column].value_counts(dropna=False)
    return [
        {
            column: str(value),
            "queries": int(count),
            "rate": float(count / max(len(frame), 1)),
        }
        for value, count in counts.items()
    ]


def main() -> None:
    args = parse_args()
    source = args.adaptive_root / "sequence_examples" / args.split
    train_source = args.adaptive_root / "sequence_examples/train"
    if not source.exists() or not train_source.exists():
        raise FileNotFoundError(f"Missing adaptive source: {source} or {train_source}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("loading positive query index...", flush=True)
    index = read_parquet(
        source,
        [
            "example_id",
            "user_id",
            "video_id",
            "as_of_time",
            "current_time_ms",
            "recent_event_count",
            "session_recent_event_count",
            "session_seconds_since_last_event",
        ],
        None,
        ds.field("target_class") == "STRONG_POSITIVE",
    )
    index = build_index(index)
    selected_ids: list[object] = []
    for _, group in index.groupby("user_id", sort=True):
        selected_ids.extend(select_user_queries(group, args.max_queries_per_user))
    selected = index[index["example_id"].isin(selected_ids)].copy()
    selected = selected.sort_values(["user_id", "as_of_time", "example_id"]).reset_index(drop=True)
    print("selected queries/users:", len(selected), selected["user_id"].nunique(), flush=True)

    print("loading selected recent histories for point-in-time validation...", flush=True)
    history = read_parquet(
        source,
        ["example_id", "user_id", "video_id", "as_of_time", "current_time_ms", "recent_video_ids", "recent_time_ms"],
        None,
        ds.field("example_id").isin(selected["example_id"].tolist()),
    )
    before_lengths = history["recent_time_ms"].map(lambda value: len(as_sequence(value))).to_numpy("int64")
    history = sanitize_point_in_time_sequences(history)
    after_lengths = history["recent_time_ms"].map(lambda value: len(as_sequence(value))).to_numpy("int64")
    rows_with_sanitized_events = int((before_lengths != after_lengths).sum())
    sanitized_events = int((before_lengths - after_lengths).sum())
    history = history[["example_id", "recent_video_ids", "recent_time_ms"]].rename(
        columns={"recent_video_ids": "recent_seen_video_ids"}
    )
    selected = selected.merge(history, on="example_id", how="left", validate="one_to_one")
    if selected["recent_seen_video_ids"].isna().any():
        raise RuntimeError("Some selected query histories were not loaded.")

    print("loading train item IDs for warm/cold labels...", flush=True)
    train_items = load_distinct_int64_column_sharded(train_source, "video_id")
    selected["target_warm"] = selected["video_id"].astype("int64").isin(train_items)
    selected["target_repeated_in_recent_history"] = [
        int(target) in set(int(value) for value in as_sequence(seen) if value is not None)
        for target, seen in zip(selected["video_id"], selected["recent_seen_video_ids"])
    ]
    selected["query_time_ms"] = (
        pd.to_datetime(selected["as_of_time"]).astype("int64") // 1_000_000
    ).astype("int64")
    selected["relevance_policy"] = "exact_target_at_query_time"
    selected["seen_filter_policy"] = "exclude_prior_items_except_exact_repeat_target"
    selected["protocol_version"] = "adaptive_eval_v2"
    selected["builder_revision"] = "2.1-session-history"

    output_columns = [
        "protocol_version",
        "builder_revision",
        "example_id",
        "user_id",
        "video_id",
        "as_of_time",
        "query_time_ms",
        "recent_event_count",
        "global_history_bucket",
        "session_recent_event_count",
        "session_history_bucket",
        "session_seconds_since_last_event",
        "positive_index",
        "positive_count",
        "timeline_fraction",
        "timeline_bucket",
        "target_warm",
        "target_repeated_in_recent_history",
        "recent_seen_video_ids",
        "relevance_policy",
        "seen_filter_policy",
    ]
    selected[output_columns].to_parquet(args.output_dir / "queries.parquet", index=False)
    selected[output_columns].drop(columns=["recent_seen_video_ids"]).to_csv(
        args.output_dir / "queries_preview.csv", index=False
    )

    per_user = selected.groupby("user_id").size()
    summary = {
        "protocol_version": "adaptive_eval_v2",
        "builder_revision": "2.1-session-history",
        "split": args.split,
        "source_positive_rows": int(len(index)),
        "queries": int(len(selected)),
        "users": int(selected["user_id"].nunique()),
        "max_queries_per_user": int(args.max_queries_per_user),
        "queries_per_user": {
            "min": int(per_user.min()),
            "median": float(per_user.median()),
            "max": int(per_user.max()),
        },
        "warm_target_rate": float(selected["target_warm"].mean()),
        "cold_target_rate": float((~selected["target_warm"]).mean()),
        "recent_repeat_target_rate": float(selected["target_repeated_in_recent_history"].mean()),
        "rows_with_sanitized_events": rows_with_sanitized_events,
        "sanitized_events": sanitized_events,
        "session_history_distribution": distribution(selected, "session_history_bucket"),
        "global_sequence_length_distribution": distribution(selected, "global_history_bucket"),
        "timeline_distribution": distribution(selected, "timeline_bucket"),
        "evaluation_rules": {
            "primary_relevance": "exact target video_id at query time",
            "optional_multi_positive": "future-window only; not materialized in step A1",
            "point_in_time": "all sequence timestamps are strictly earlier than query_time_ms",
            "seen_filter": "exclude prior items, except an exact repeated target remains eligible",
            "bootstrap_unit": "user",
            "candidate_eligibility": "must be enforced in step A2",
            "history_segmentation": "session_recent_event_count; recent_event_count is sequence-capped context only",
        },
        "artifact_paths": {
            "queries": str(args.output_dir / "queries.parquet"),
            "preview": str(args.output_dir / "queries_preview.csv"),
            "manifest": str(args.output_dir / "manifest.json"),
        },
    }
    save_json(args.output_dir / "manifest.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    print("wrote:", args.output_dir, flush=True)


if __name__ == "__main__":
    main()
