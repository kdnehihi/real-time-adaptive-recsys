from __future__ import annotations

import argparse
import gc
import json
import shutil
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from recommender.models.adaptive.checkpoint import as_sequence, parquet_files, read_parquet, save_json
from recommender.models.adaptive.online_state import InteractionEvent, OnlineUserStateReducer


RECENT_ARRAY_COLUMNS = (
    "recent_video_ids",
    "recent_target_classes",
    "recent_engagement_strengths",
    "recent_long_views",
    "recent_likes",
    "recent_hates",
    "recent_clicks",
    "recent_time_ms",
)
SESSION_ARRAY_COLUMNS = tuple(f"session_{name}" for name in RECENT_ARRAY_COLUMNS)
SCALAR_COLUMNS = (
    "recent_event_count",
    "session_recent_event_count",
    "seconds_since_last_event",
    "session_seconds_since_last_event",
    "recent_long_view_rate",
    "recent_like_rate",
    "recent_hate_rate",
    "recent_avg_engagement_strength",
)
PARITY_COLUMNS = (*RECENT_ARRAY_COLUMNS, *SESSION_ARRAY_COLUMNS, *SCALAR_COLUMNS)
CORE_PARITY_COLUMNS = (
    *RECENT_ARRAY_COLUMNS,
    *SESSION_ARRAY_COLUMNS,
    "recent_event_count",
    "session_recent_event_count",
)
EVENT_VALUE_COLUMNS = (
    "user_id",
    "video_id",
    "current_time_ms",
    "session_id",
    "target_class",
    "engagement_strength",
    "long_view",
    "is_like",
    "is_hate",
    "is_click",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild bounded adaptive state event by event and compare it with Gold snapshots."
    )
    parser.add_argument("--adaptive-root", type=Path, required=True)
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="validation", choices=("validation",))
    parser.add_argument("--max-sequence-len", type=int, default=30)
    parser.add_argument("--max-session-sequence-len", type=int, default=10)
    parser.add_argument("--calibration-user-fraction", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--max-users", type=int, default=0, help="Smoke-test limit; zero means all holdout users.")
    parser.add_argument("--work-dir", type=Path, default=Path("/content/recsys_online_state_work"))
    parser.add_argument("--keep-work-db", action="store_true")
    return parser.parse_args()


def parquet_schema(path: Path) -> set[str]:
    """Read only the first shard schema, avoiding a full remote dataset scan."""
    with parquet_files(path)[0].open("rb") as stream:
        return set(pq.ParquetFile(stream).schema_arrow.names)


def deterministic_user_split(user_ids: pd.Series, fraction: float, seed: int) -> np.ndarray:
    """Reproduce the user-level calibration/holdout split used by A2 and V4.2."""
    if not 0.0 < fraction < 1.0:
        raise ValueError("calibration-user-fraction must be between zero and one")
    values = user_ids.astype("int64").to_numpy()
    hashed = (values * 2_654_435_761 + int(seed) * 97_531) % 1_000_003
    return hashed < int(fraction * 1_000_003)


def read_shard(path: Path, columns: list[str], retries: int = 5) -> pd.DataFrame:
    """Read and close one Drive-backed shard, retrying transient FUSE failures."""
    for attempt in range(1, retries + 1):
        try:
            with path.open("rb") as stream:
                table = pq.ParquetFile(stream).read(columns=columns, use_threads=False)
            return table.to_pandas()
        except (OSError, IOError) as exc:
            gc.collect()
            if attempt == retries:
                raise OSError(f"Failed reading {path.name} after {retries} attempts") from exc
            time.sleep(min(2 ** (attempt - 1), 8))
    raise AssertionError("unreachable")


def sanitize_arrays_strict_prior(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Apply the same strict ``event_time < query_time`` rule to aligned arrays."""
    frame = frame.copy()
    time_column = f"{prefix}_time_ms"
    array_columns = RECENT_ARRAY_COLUMNS if prefix == "recent" else SESSION_ARRAY_COLUMNS
    for row_index, row in frame.iterrows():
        times = as_sequence(row.get(time_column))
        boundary = int(row["current_time_ms"])
        keep = [index for index, value in enumerate(times) if value is not None and int(value) < boundary]
        for column in array_columns:
            values = as_sequence(row.get(column))
            frame.at[row_index, column] = [values[index] for index in keep] if len(values) == len(times) else []
    frame[f"{prefix}_event_count"] = frame[time_column].map(lambda value: len(as_sequence(value)))
    return frame


def load_holdout_queries(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load A2 holdout IDs and the full Gold arrays needed for state parity."""
    protocol = pd.read_parquet(args.protocol_dir / "queries.parquet")
    holdout = ~deterministic_user_split(
        protocol["user_id"], args.calibration_user_fraction, args.seed
    )
    protocol = protocol.loc[holdout].copy()
    users = sorted(protocol["user_id"].astype("int64").unique())
    if args.max_users:
        users = users[: args.max_users]
        protocol = protocol[protocol["user_id"].isin(users)].copy()
    source = args.adaptive_root / "sequence_examples" / args.split
    available = parquet_schema(source)
    required = {
        "example_id",
        "user_id",
        "video_id",
        "session_id",
        "as_of_time",
        "current_time_ms",
        *RECENT_ARRAY_COLUMNS,
        *SESSION_ARRAY_COLUMNS,
        *SCALAR_COLUMNS,
    }
    missing = sorted(required - available)
    if missing:
        raise RuntimeError(f"Adaptive Gold is missing online-state parity columns: {missing}")
    ids = protocol["example_id"].tolist()
    features = read_parquet(source, sorted(required), filter_expr=ds.field("example_id").isin(ids))
    features = features.drop_duplicates("example_id", keep="last")
    order = {value: index for index, value in enumerate(ids)}
    features = features[features["example_id"].isin(order)].copy()
    features["_order"] = features["example_id"].map(order)
    features = features.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    protocol = protocol.set_index("example_id").loc[features["example_id"]].reset_index()
    features["current_time_ms"] = pd.to_numeric(
        features["current_time_ms"], errors="raise"
    ).astype("int64")
    features = sanitize_arrays_strict_prior(features, "recent")
    features = sanitize_arrays_strict_prior(features, "session_recent")
    return features, protocol


def initialize_event_db(path: Path) -> sqlite3.Connection:
    """Create a disk-backed event spool so Colab RAM does not scale with history."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute(
        """
        CREATE TABLE events (
            event_key TEXT NOT NULL,
            order_id TEXT NOT NULL,
            user_id INTEGER NOT NULL,
            video_id INTEGER NOT NULL,
            time_ms INTEGER NOT NULL,
            session_id TEXT NOT NULL,
            target_class TEXT NOT NULL,
            engagement_strength REAL NOT NULL,
            long_view INTEGER NOT NULL,
            is_like INTEGER NOT NULL,
            is_hate INTEGER NOT NULL,
            is_click INTEGER NOT NULL
        )
        """
    )
    return connection


def _safe_number(value: Any, cast: type, default: int | float = 0) -> int | float:
    """Normalize nullable pandas values before inserting them into SQLite."""
    try:
        if value is None or bool(pd.isna(value)):
            return default
        return cast(value)
    except (TypeError, ValueError):
        return default


def event_rows(frame: pd.DataFrame, order_column: str) -> Iterable[tuple[Any, ...]]:
    """Yield compact SQLite rows without retaining Python event objects."""
    for row in frame.to_dict("records"):
        event_key = str(row.get("example_id", row.get(order_column, "")))
        yield (
            event_key,
            str(row.get(order_column, event_key)),
            _safe_number(row.get("user_id"), int),
            _safe_number(row.get("video_id"), int),
            _safe_number(row.get("current_time_ms"), int),
            str(row.get("session_id", "")),
            str(row.get("target_class", "AMBIGUOUS_WEAK")),
            _safe_number(row.get("engagement_strength"), float),
            _safe_number(row.get("long_view"), int),
            _safe_number(row.get("is_like"), int),
            _safe_number(row.get("is_hate"), int),
            _safe_number(row.get("is_click"), int),
        )


def spool_events(
    connection: sqlite3.Connection,
    adaptive_root: Path,
    split: str,
    user_ids: set[int],
    max_time_ms: int,
) -> dict[str, Any]:
    """Scan train/evaluation shards once and spool only selected users to disk."""
    inserted = 0
    order_sources: set[str] = set()
    statement = "INSERT INTO events VALUES (?,?,?,?,?,?,?,?,?,?,?,?)"
    for source_split in ("train", split):
        source = adaptive_root / "sequence_examples" / source_split
        available = parquet_schema(source)
        order_column = next(
            (name for name in ("current_event_id", "event_id", "user_event_index", "example_id") if name in available),
            None,
        )
        if order_column is None:
            raise RuntimeError(f"No deterministic event tie-breaker in {source}")
        order_sources.add(order_column)
        required = set(EVENT_VALUE_COLUMNS) | {"example_id", order_column}
        missing = sorted(required - available)
        if missing:
            raise RuntimeError(f"Adaptive event source {source_split} is missing: {missing}")
        columns = sorted(required)
        files = parquet_files(source)
        for index, file_path in enumerate(files, start=1):
            part = read_shard(file_path, columns)
            part["user_id"] = pd.to_numeric(part["user_id"], errors="coerce")
            part["current_time_ms"] = pd.to_numeric(part["current_time_ms"], errors="coerce")
            part = part[
                part["user_id"].isin(user_ids)
                & part["current_time_ms"].notna()
                & (part["current_time_ms"] <= max_time_ms)
            ]
            if not part.empty:
                rows = list(event_rows(part, order_column))
                connection.executemany(statement, rows)
                inserted += len(rows)
            if index % 16 == 0 or index == len(files):
                connection.commit()
                print(
                    f"event spool {source_split}: {index}/{len(files)} shards; {inserted:,} rows",
                    flush=True,
                )
            del part
            gc.collect()
    print("building event-time index...", flush=True)
    connection.execute("CREATE INDEX event_order ON events(time_ms, order_id, user_id, event_key)")
    connection.commit()
    return {
        "events_spooled": int(inserted),
        "duplicate_event_keys": "not_precomputed; parity detects duplicate-state effects",
        "tie_breaker_columns": sorted(order_sources),
    }


def values_match(expected: Any, actual: Any) -> bool:
    """Compare Gold and online values with tolerance only for floating point data."""
    expected_sequence = as_sequence(expected)
    actual_sequence = as_sequence(actual)
    if expected_sequence or actual_sequence or isinstance(expected, (list, tuple, np.ndarray)):
        if len(expected_sequence) != len(actual_sequence):
            return False
        for left, right in zip(expected_sequence, actual_sequence):
            if isinstance(left, (float, np.floating)) or isinstance(right, (float, np.floating)):
                if not np.isclose(float(left), float(right), rtol=1e-6, atol=1e-6, equal_nan=True):
                    return False
            elif left != right:
                return False
        return True
    def missing_scalar(value: Any) -> bool:
        if value is None:
            return True
        try:
            marker = pd.isna(value)
            return bool(marker) if np.isscalar(marker) else False
        except (TypeError, ValueError):
            return False

    if missing_scalar(expected):
        return missing_scalar(actual)
    if isinstance(expected, (float, np.floating)) or isinstance(actual, (float, np.floating)):
        return bool(np.isclose(float(expected), float(actual), rtol=1e-6, atol=1e-6))
    return expected == actual


def replay_and_compare(
    connection: sqlite3.Connection,
    queries: pd.DataFrame,
    max_sequence_len: int,
    max_session_sequence_len: int,
) -> tuple[dict[str, Any], pd.DataFrame, list[dict[str, Any]]]:
    """Replay the event spool and compare every selected query before its event is appended."""
    reducer = OnlineUserStateReducer(max_sequence_len, max_session_sequence_len)
    by_time: dict[int, list[int]] = defaultdict(list)
    for index, value in enumerate(queries["current_time_ms"].astype("int64")):
        by_time[int(value)].append(index)
    field_matches = {column: 0 for column in PARITY_COLUMNS}
    rows: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    evaluated: set[int] = set()
    cursor = connection.execute(
        """SELECT event_key, order_id, user_id, video_id, time_ms, session_id,
                  target_class, engagement_strength, long_view, is_like, is_hate, is_click
           FROM events ORDER BY time_ms, order_id, user_id, event_key"""
    )
    pending_time: int | None = None
    pending: list[InteractionEvent] = []

    def evaluate_timestamp(timestamp: int, events: list[InteractionEvent]) -> None:
        for query_index in by_time.get(timestamp, []):
            expected = queries.iloc[query_index]
            actual = reducer.snapshot(
                int(expected["user_id"]), expected["session_id"], timestamp
            )
            failed: list[str] = []
            for column in PARITY_COLUMNS:
                if values_match(expected[column], actual[column]):
                    field_matches[column] += 1
                else:
                    failed.append(column)
            all_match = not failed
            core_failed = [column for column in failed if column in CORE_PARITY_COLUMNS]
            rows.append(
                {
                    "example_id": expected["example_id"],
                    "user_id": int(expected["user_id"]),
                    "query_time_ms": timestamp,
                    "all_fields_match": all_match,
                    "core_fields_match": not core_failed,
                    "failed_fields": ",".join(failed),
                }
            )
            evaluated.add(query_index)
            if failed and len(mismatches) < 100:
                mismatches.append(
                    {
                        "example_id": expected["example_id"],
                        "user_id": int(expected["user_id"]),
                        "query_time_ms": timestamp,
                        "failed_fields": failed,
                        "expected_recent_video_ids": as_sequence(expected["recent_video_ids"]),
                        "actual_recent_video_ids": actual["recent_video_ids"],
                        "expected_recent_time_ms": as_sequence(expected["recent_time_ms"]),
                        "actual_recent_time_ms": actual["recent_time_ms"],
                    }
                )
        reducer.apply_timestamp_batch(events)

    for record in cursor:
        event = InteractionEvent(
            event_id=str(record[1]),
            user_id=int(record[2]),
            video_id=int(record[3]),
            time_ms=int(record[4]),
            session_id=str(record[5]),
            target_class=str(record[6]),
            engagement_strength=float(record[7]),
            long_view=int(record[8]),
            is_like=int(record[9]),
            is_hate=int(record[10]),
            is_click=int(record[11]),
        )
        if pending_time is None:
            pending_time = event.time_ms
        if event.time_ms != pending_time:
            evaluate_timestamp(pending_time, pending)
            pending_time = event.time_ms
            pending = []
        pending.append(event)
    if pending_time is not None:
        evaluate_timestamp(pending_time, pending)

    unresolved = sorted(set(range(len(queries))) - evaluated)
    query_count = len(queries)
    result = pd.DataFrame(rows)
    report = {
        "queries_expected": query_count,
        "queries_evaluated": len(evaluated),
        "queries_unresolved": len(unresolved),
        "events_applied": reducer.events_applied,
        "active_users": reducer.active_users,
        "active_sessions": reducer.active_sessions,
        "all_fields_match_rate": float(result["all_fields_match"].mean()) if len(result) else 0.0,
        "core_fields_match_rate": float(result["core_fields_match"].mean()) if len(result) else 0.0,
        "field_match_rates": {
            column: field_matches[column] / max(query_count, 1) for column in PARITY_COLUMNS
        },
        "accepted": len(unresolved) == 0 and len(result) == query_count and bool(result["core_fields_match"].all()),
        "unresolved_example_ids": queries.iloc[unresolved]["example_id"].tolist()[:100],
    }
    return report, result, mismatches


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    print("loading fixed A2 holdout queries and Gold snapshots...", flush=True)
    queries, protocol = load_holdout_queries(args)
    users = set(queries["user_id"].astype("int64"))
    print(f"parity queries/users: {len(queries):,}/{len(users):,}", flush=True)

    db_path = args.work_dir / "selected_events.sqlite"
    connection = initialize_event_db(db_path)
    try:
        spool = spool_events(
            connection,
            args.adaptive_root,
            args.split,
            users,
            int(queries["current_time_ms"].max()),
        )
        replay_started = time.perf_counter()
        parity, per_query, mismatches = replay_and_compare(
            connection,
            queries,
            args.max_sequence_len,
            args.max_session_sequence_len,
        )
        replay_seconds = time.perf_counter() - replay_started
    finally:
        connection.close()

    per_query.to_csv(args.output_dir / "online_state_parity_per_query.csv", index=False)
    save_json(args.output_dir / "online_state_mismatch_examples.json", {"examples": mismatches})
    report = {
        "protocol": "v6_true_event_by_event_state_parity_a2_holdout",
        "state_semantics": "all queries at t read state from events strictly before t; events at t append afterward",
        "sequence_limits": {
            "global": args.max_sequence_len,
            "session": args.max_session_sequence_len,
        },
        "scope": {
            "reconstructed": "bounded recent and same-session event arrays plus available rates/counters",
            "not_reconstructed": "long-term Gold user/item aggregates; those require their own incremental reducers",
        },
        "queries": len(queries),
        "users": len(users),
        "event_spool": spool,
        "parity": parity,
        "timing": {
            "replay_seconds": replay_seconds,
            "state_events_per_second": parity["events_applied"] / max(replay_seconds, 1e-9),
            "total_seconds": time.perf_counter() - started,
            "note": "State-reducer throughput only; excludes V4.2 exact matrix retrieval.",
        },
        "artifacts": {
            "per_query": "online_state_parity_per_query.csv",
            "mismatch_examples": "online_state_mismatch_examples.json",
        },
    }
    save_json(args.output_dir / "online_state_parity_report.json", report)
    print(json.dumps(report, indent=2), flush=True)
    if not args.keep_work_db:
        db_path.unlink(missing_ok=True)
        shutil.rmtree(args.work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
