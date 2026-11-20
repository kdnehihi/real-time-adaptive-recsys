from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from pyspark import StorageLevel
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql import types as T

from recommender.data.adaptive_sequences import build_strict_prior_sequence_context
from recommender.models.adaptive.online_state import InteractionEvent, OnlineUserStateReducer
from recommender.spark import get_spark


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
CORE_COLUMNS = (*RECENT_ARRAY_COLUMNS, *SESSION_ARRAY_COLUMNS, "recent_event_count", "session_recent_event_count")
EVENT_COLUMNS = (
    "event_id",
    "user_id",
    "video_id",
    "time_ms",
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
        description="Audit corrected timestamp-batch sequences against an independent event reducer."
    )
    parser.add_argument("--adaptive-root", type=Path, required=True)
    parser.add_argument("--base-gold-root", type=Path, required=True)
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-sequence-len", type=int, default=30)
    parser.add_argument("--max-session-sequence-len", type=int, default=10)
    parser.add_argument("--calibration-user-fraction", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--max-users", type=int, default=0)
    parser.add_argument("--shuffle-partitions", type=int, default=32)
    return parser.parse_args()


def holdout_expression(fraction: float, seed: int) -> F.Column:
    """Reproduce the user-disjoint A2/V4.2 holdout partition in Spark."""
    if not 0.0 < fraction < 1.0:
        raise ValueError("calibration-user-fraction must be between zero and one")
    hashed = F.pmod(
        F.col("user_id").cast("long") * F.lit(2_654_435_761)
        + F.lit(int(seed) * 97_531),
        F.lit(1_000_003),
    )
    return hashed >= F.lit(int(fraction * 1_000_003))


def load_queries(args: argparse.Namespace) -> tuple[DataFrame, DataFrame]:
    """Load locked holdout IDs plus session/legacy state from adaptive Gold."""
    spark = get_spark()
    protocol = (
        spark.read.parquet(str(args.protocol_dir / "queries.parquet"))
        .where(holdout_expression(args.calibration_user_fraction, args.seed))
        .select("example_id", "user_id", "video_id", "query_time_ms")
    )
    if args.max_users:
        selected_users = protocol.select("user_id").distinct().orderBy("user_id").limit(args.max_users)
        protocol = protocol.join(F.broadcast(selected_users), "user_id", "inner")
    legacy_columns = [
        "example_id",
        "user_id",
        "video_id",
        "session_id",
        "current_time_ms",
        *RECENT_ARRAY_COLUMNS,
        *SESSION_ARRAY_COLUMNS,
        "recent_event_count",
        "session_recent_event_count",
    ]
    legacy = spark.read.parquet(str(args.adaptive_root / "sequence_examples/validation")).select(
        *legacy_columns
    )
    queries = (
        protocol.join(legacy, ["example_id", "user_id", "video_id"], "inner")
        .dropDuplicates(["example_id"])
        .persist()
    )
    users = queries.select("user_id").distinct().persist()
    return queries, users


def load_canonical_history(args: argparse.Namespace, users: DataFrame, max_time_ms: int) -> DataFrame:
    """Read the canonical history used by the original Gold builder, not joined examples."""
    history_path = args.base_gold_root / "history/events"
    if not history_path.exists():
        raise FileNotFoundError(f"Canonical history is required: {history_path}")
    history = (
        get_spark()
        .read.parquet(str(history_path))
        .select(*EVENT_COLUMNS)
        .join(F.broadcast(users), "user_id", "inner")
        .where(F.col("time_ms") <= F.lit(max_time_ms))
        .dropDuplicates(["event_id"])
        .persist(StorageLevel.DISK_ONLY)
    )
    return history


def corrected_batch_queries(
    history: DataFrame,
    queries: DataFrame,
    maximum_events: int,
    maximum_session_events: int,
) -> DataFrame:
    """Join query rows to corrected batch sequences built independently in Spark SQL."""
    global_context, session_context = build_strict_prior_sequence_context(
        history, maximum_events, maximum_session_events
    )
    query_keys = queries.select(
        "example_id", "user_id", "video_id", "session_id", "current_time_ms"
    )
    global_columns = [*RECENT_ARRAY_COLUMNS, "recent_event_count"]
    session_columns = [*SESSION_ARRAY_COLUMNS, "session_recent_event_count"]
    return (
        query_keys.alias("q")
        .join(
            global_context.alias("g"),
            (F.col("q.user_id") == F.col("g.user_id"))
            & (F.col("q.current_time_ms") == F.col("g.time_ms")),
            "inner",
        )
        .join(
            session_context.alias("s"),
            (F.col("q.user_id") == F.col("s.user_id"))
            & (F.col("q.session_id") == F.col("s.session_id"))
            & (F.col("q.current_time_ms") == F.col("s.time_ms")),
            "inner",
        )
        .select(
            "q.example_id",
            "q.user_id",
            "q.video_id",
            "q.session_id",
            "q.current_time_ms",
            *[F.col(f"g.{name}").alias(name) for name in global_columns],
            *[F.col(f"s.{name}").alias(name) for name in session_columns],
        )
        .persist()
    )


def replay_output_schema() -> T.StructType:
    """Declare Arrow-compatible output for the independent per-user reducer."""
    return T.StructType(
        [
            T.StructField("example_id", T.StringType(), False),
            T.StructField("user_id", T.LongType(), False),
            T.StructField("video_id", T.IntegerType(), False),
            T.StructField("session_id", T.StringType(), False),
            T.StructField("current_time_ms", T.LongType(), False),
            T.StructField("recent_video_ids", T.ArrayType(T.IntegerType()), False),
            T.StructField("recent_target_classes", T.ArrayType(T.StringType()), False),
            T.StructField("recent_engagement_strengths", T.ArrayType(T.DoubleType()), False),
            T.StructField("recent_long_views", T.ArrayType(T.IntegerType()), False),
            T.StructField("recent_likes", T.ArrayType(T.IntegerType()), False),
            T.StructField("recent_hates", T.ArrayType(T.IntegerType()), False),
            T.StructField("recent_clicks", T.ArrayType(T.IntegerType()), False),
            T.StructField("recent_time_ms", T.ArrayType(T.LongType()), False),
            T.StructField("session_recent_video_ids", T.ArrayType(T.IntegerType()), False),
            T.StructField("session_recent_target_classes", T.ArrayType(T.StringType()), False),
            T.StructField("session_recent_engagement_strengths", T.ArrayType(T.DoubleType()), False),
            T.StructField("session_recent_long_views", T.ArrayType(T.IntegerType()), False),
            T.StructField("session_recent_likes", T.ArrayType(T.IntegerType()), False),
            T.StructField("session_recent_hates", T.ArrayType(T.IntegerType()), False),
            T.StructField("session_recent_clicks", T.ArrayType(T.IntegerType()), False),
            T.StructField("session_recent_time_ms", T.ArrayType(T.LongType()), False),
            T.StructField("recent_event_count", T.IntegerType(), False),
            T.StructField("session_recent_event_count", T.IntegerType(), False),
        ]
    )


def replay_user_partition(
    frame: pd.DataFrame,
    maximum_events: int,
    maximum_session_events: int,
) -> pd.DataFrame:
    """Replay one user's timestamp batches and emit state before current-time events."""
    reducer = OnlineUserStateReducer(maximum_events, maximum_session_events)
    output: list[dict[str, Any]] = []
    frame = frame.sort_values(["time_ms", "record_type", "event_id"], na_position="last")
    for timestamp, current in frame.groupby("time_ms", sort=True):
        query_rows = current[current["record_type"] == "query"]
        for row in query_rows.to_dict("records"):
            state = reducer.snapshot(int(row["user_id"]), row["session_id"], int(timestamp))
            output.append(
                {
                    "example_id": str(row["example_id"]),
                    "user_id": int(row["user_id"]),
                    "video_id": int(row["video_id"]),
                    "session_id": str(row["session_id"]),
                    "current_time_ms": int(timestamp),
                    **{name: state[name] for name in CORE_COLUMNS},
                }
            )
        events = [
            InteractionEvent.from_mapping(row)
            for row in current[current["record_type"] == "event"].to_dict("records")
        ]
        reducer.apply_timestamp_batch(events)
    return pd.DataFrame(output, columns=[field.name for field in replay_output_schema()])


def corrected_replay_queries(
    history: DataFrame,
    queries: DataFrame,
    maximum_events: int,
    maximum_session_events: int,
) -> DataFrame:
    """Run an independent event reducer per user using grouped Arrow batches."""
    events = history.select(
        F.lit("event").alias("record_type"),
        *EVENT_COLUMNS,
        F.lit(None).cast("string").alias("example_id"),
    )
    query_rows = queries.select(
        F.lit("query").alias("record_type"),
        F.lit(None).cast("string").alias("event_id"),
        "user_id",
        "video_id",
        F.col("current_time_ms").alias("time_ms"),
        "session_id",
        F.lit(None).cast("string").alias("target_class"),
        F.lit(None).cast("double").alias("engagement_strength"),
        *[F.lit(None).cast("int").alias(name) for name in ("long_view", "is_like", "is_hate", "is_click")],
        "example_id",
    )
    timeline = events.unionByName(query_rows)

    def replay(frame: pd.DataFrame) -> pd.DataFrame:
        return replay_user_partition(frame, maximum_events, maximum_session_events)

    return timeline.groupBy("user_id").applyInPandas(replay, replay_output_schema()).persist()


def compare_contexts(batch: DataFrame, replay: DataFrame) -> tuple[DataFrame, dict[str, Any]]:
    """Compare every model-facing core array/count and return query-level diagnostics."""
    joined = batch.alias("b").join(replay.alias("r"), "example_id", "inner")
    checks = [F.col(f"b.{name}").eqNullSafe(F.col(f"r.{name}")).alias(f"match_{name}") for name in CORE_COLUMNS]
    compared = joined.select(
        "example_id",
        F.col("b.user_id").alias("user_id"),
        F.col("b.current_time_ms").alias("query_time_ms"),
        F.size(F.col("b.recent_video_ids")).alias("batch_recent_count"),
        F.size(F.col("r.recent_video_ids")).alias("replay_recent_count"),
        *checks,
    )
    match_columns = [f"match_{name}" for name in CORE_COLUMNS]
    all_match = F.aggregate(
        F.array(*[F.col(name) for name in match_columns]),
        F.lit(True),
        lambda total, value: total & value,
    )
    compared = compared.withColumn("core_fields_match", all_match).persist()
    row = compared.agg(
        F.count("*").alias("queries"),
        F.avg(F.col("core_fields_match").cast("double")).alias("core_rate"),
        *[F.avg(F.col(name).cast("double")).alias(name) for name in match_columns],
    ).first()
    report = {
        "queries_compared": int(row["queries"]),
        "core_fields_match_rate": float(row["core_rate"] or 0.0),
        "field_match_rates": {
            name.removeprefix("match_"): float(row[name] or 0.0) for name in match_columns
        },
    }
    report["accepted"] = report["core_fields_match_rate"] == 1.0
    return compared, report


def legacy_delta(queries: DataFrame, corrected: DataFrame) -> dict[str, Any]:
    """Quantify the legacy post-window sanitation/no-backfill mismatch."""
    joined = queries.alias("l").join(corrected.alias("c"), "example_id", "inner")
    legacy_count = F.size("l.recent_video_ids")
    corrected_count = F.size("c.recent_video_ids")
    suffix_start = corrected_count - legacy_count + F.lit(1)
    legacy_is_suffix = (legacy_count <= corrected_count) & F.slice(
        F.col("c.recent_video_ids"), suffix_start, legacy_count
    ).eqNullSafe(F.col("l.recent_video_ids"))
    row = joined.agg(
        F.count("*").alias("queries"),
        F.avg((legacy_count == corrected_count).cast("double")).alias("count_match"),
        F.avg(legacy_is_suffix.cast("double")).alias("suffix_rate"),
        F.avg(legacy_count.cast("double")).alias("legacy_mean"),
        F.avg(corrected_count.cast("double")).alias("corrected_mean"),
    ).first()
    return {
        "queries": int(row["queries"]),
        "recent_count_match_rate": float(row["count_match"] or 0.0),
        "legacy_is_corrected_suffix_rate": float(row["suffix_rate"] or 0.0),
        "legacy_mean_recent_count": float(row["legacy_mean"] or 0.0),
        "corrected_mean_recent_count": float(row["corrected_mean"] or 0.0),
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)
    os.environ.setdefault("SPARK_MASTER", "local[*]")
    spark = get_spark("adaptive-corrected-sequence-audit", reset=True)
    spark.sparkContext.setLogLevel("WARN")
    spark.conf.set("spark.sql.shuffle.partitions", str(args.shuffle_partitions))
    spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", "200000")
    started = time.perf_counter()

    print("loading locked holdout queries...", flush=True)
    queries, users = load_queries(args)
    query_count = queries.count()
    user_count = users.count()
    max_time_ms = int(queries.agg(F.max("current_time_ms")).first()[0])
    print(f"queries/users: {query_count:,}/{user_count:,}", flush=True)

    print("loading canonical base-Gold event history...", flush=True)
    history = load_canonical_history(args, users, max_time_ms)
    event_count = history.count()
    print(f"canonical events: {event_count:,}", flush=True)

    print("building corrected timestamp-batch contexts...", flush=True)
    batch = corrected_batch_queries(
        history, queries, args.max_sequence_len, args.max_session_sequence_len
    )
    batch_count = batch.count()
    print(f"batch snapshots: {batch_count:,}", flush=True)

    print("running independent event-by-event reducer...", flush=True)
    replay = corrected_replay_queries(
        history, queries, args.max_sequence_len, args.max_session_sequence_len
    )
    replay_count = replay.count()
    print(f"replay snapshots: {replay_count:,}", flush=True)

    compared, parity = compare_contexts(batch, replay)
    legacy = legacy_delta(queries, batch)
    compared.toPandas().to_csv(args.output_dir / "corrected_sequence_parity_per_query.csv", index=False)
    mismatch = (
        batch.alias("b")
        .join(replay.alias("r"), "example_id", "inner")
        .join(compared.where(~F.col("core_fields_match")).select("example_id"), "example_id", "inner")
        .select(
            "example_id",
            F.col("b.user_id").alias("user_id"),
            F.col("b.current_time_ms").alias("query_time_ms"),
            F.col("b.recent_video_ids").alias("batch_recent_video_ids"),
            F.col("r.recent_video_ids").alias("replay_recent_video_ids"),
            F.col("b.session_recent_video_ids").alias("batch_session_video_ids"),
            F.col("r.session_recent_video_ids").alias("replay_session_video_ids"),
        )
        .limit(100)
        .toPandas()
        .to_dict("records")
    )
    (args.output_dir / "corrected_sequence_mismatch_examples.json").write_text(
        json.dumps({"examples": mismatch}, indent=2, default=str)
    )
    report = {
        "protocol": "corrected_timestamp_batch_sequence_audit_v1",
        "semantics": "filter complete current timestamp batch before applying 30/10 event limits",
        "queries": query_count,
        "users": user_count,
        "canonical_events": event_count,
        "batch_snapshots": batch_count,
        "replay_snapshots": replay_count,
        "corrected_batch_vs_event_reducer": parity,
        "legacy_vs_corrected": legacy,
        "elapsed_seconds": time.perf_counter() - started,
        "next_gate": "rebuild adaptive Gold and retrain V4 only if corrected parity is accepted",
    }
    (args.output_dir / "corrected_sequence_audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True)
    )
    print(json.dumps(report, indent=2), flush=True)
    spark.stop()


if __name__ == "__main__":
    main()
