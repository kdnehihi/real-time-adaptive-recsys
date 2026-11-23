from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from pyspark import StorageLevel
from pyspark.sql import functions as F
from pyspark.sql import types as T

from recommender.data.adaptive_sequences import (
    EVENT_VALUE_COLUMNS,
    build_strict_prior_global_context,
)
from recommender.spark import get_spark


CORRECTED_COLUMNS = (
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
    "recent_long_view_rate",
    "recent_like_rate",
    "recent_hate_rate",
    "recent_avg_engagement_strength",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build corrected strict-prior sequences only for locked test queries."
    )
    parser.add_argument("--base-gold-root", type=Path, required=True)
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-sequence-len", type=int, default=30)
    parser.add_argument("--shuffle-partitions", type=int, default=64)
    parser.add_argument("--output-partitions", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def protocol_frame(path: Path) -> pd.DataFrame:
    """Load primitive query keys without exposing nanosecond timestamps to Spark."""
    columns = ["example_id", "user_id", "query_time_ms"]
    frame = pd.read_parquet(path, columns=columns).dropna(subset=columns).copy()
    frame["example_id"] = frame["example_id"].astype(str)
    frame["user_id"] = pd.to_numeric(frame["user_id"], errors="raise").astype("int64")
    frame["query_time_ms"] = pd.to_numeric(
        frame["query_time_ms"], errors="raise"
    ).astype("int64")
    return frame


def resolve_exact_query_clocks(queries, history):
    """Resolve millisecond query clocks using the immutable Gold example key.

    Locked protocol timestamps are second-precision and must not be used for an
    equality join to millisecond event contexts.  Two-Tower Gold defines an
    example ID as SHA-256(``example||event_id||time_ms``), so reproducing that
    key from canonical history preserves the locked query population while
    recovering each query's exact event time.
    """
    canonical_example_id = F.sha2(
        F.concat_ws(
            "||",
            F.lit("example"),
            F.col("event_id").cast("string"),
            F.col("time_ms").cast("string"),
        ),
        256,
    )
    return (
        queries.alias("q")
        .join(
            history.select(
                canonical_example_id.alias("example_id"),
                "user_id",
                F.col("time_ms").alias("current_time_ms"),
            ).alias("e"),
            (F.col("q.example_id") == F.col("e.example_id"))
            & (F.col("q.user_id") == F.col("e.user_id")),
            "inner",
        )
        .select(
            "q.example_id",
            "q.user_id",
            "q.query_time_ms",
            "e.current_time_ms",
        )
        .dropDuplicates(["example_id"])
    )


def main() -> None:
    args = parse_args()
    history_path = args.base_gold_root / "history/events"
    protocol_path = args.protocol_dir / "queries.parquet"
    if not history_path.exists() or not protocol_path.exists():
        raise FileNotFoundError(f"Missing history/protocol: {history_path}, {protocol_path}")
    if args.output_dir.exists() and args.overwrite:
        shutil.rmtree(args.output_dir)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output exists; pass --overwrite: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("SPARK_MASTER", "local[2]")
    spark = get_spark("corrected-locked-test-sequences", reset=True)
    spark.sparkContext.setLogLevel("WARN")
    spark.conf.set("spark.sql.shuffle.partitions", str(args.shuffle_partitions))
    started = time.perf_counter()

    protocol = protocol_frame(protocol_path)
    schema = T.StructType(
        [
            T.StructField("example_id", T.StringType(), False),
            T.StructField("user_id", T.LongType(), False),
            T.StructField("query_time_ms", T.LongType(), False),
        ]
    )
    queries = spark.createDataFrame(protocol, schema=schema).persist()
    users = queries.select("user_id").distinct()
    # Protocol v2 intentionally stores a compact query clock derived from
    # ``as_of_time``.  That timestamp has second precision, while canonical
    # event history retains milliseconds.  It is therefore suitable for the
    # locked retrieval protocol but not for an equality join to event-time
    # sequence contexts.  Read the final protocol second as well (+999 ms),
    # then recover each query's exact clock from its immutable event ID.
    maximum_protocol_time = int(protocol["query_time_ms"].max())
    print(f"locked queries/users: {len(protocol):,}/{protocol['user_id'].nunique():,}", flush=True)

    required_history = ["event_id", "user_id", *EVENT_VALUE_COLUMNS]
    history = (
        spark.read.parquet(str(history_path))
        .select(*required_history)
        .join(F.broadcast(users), "user_id", "inner")
        .where(F.col("time_ms") <= F.lit(maximum_protocol_time + 999))
        .dropDuplicates(["event_id"])
        .persist(StorageLevel.DISK_ONLY)
    )
    history_count = history.count()
    print(f"canonical events for selected users: {history_count:,}", flush=True)
    query_clocks = resolve_exact_query_clocks(queries, history).persist()
    resolved_clock_count = query_clocks.count()
    if resolved_clock_count != len(protocol):
        raise AssertionError(
            "Could not resolve an exact canonical event clock for every locked query: "
            f"resolved={resolved_clock_count:,}, expected={len(protocol):,}"
        )
    clock_row = query_clocks.agg(
        F.min(F.col("current_time_ms") - F.col("query_time_ms")).alias("min_delta_ms"),
        F.max(F.col("current_time_ms") - F.col("query_time_ms")).alias("max_delta_ms"),
    ).first()
    if int(clock_row["min_delta_ms"]) < 0 or int(clock_row["max_delta_ms"]) > 999:
        raise AssertionError(
            "Canonical event clock is outside the protocol timestamp's source second: "
            f"min_delta_ms={clock_row['min_delta_ms']}, "
            f"max_delta_ms={clock_row['max_delta_ms']}"
        )
    print(
        "resolved exact query clocks: "
        f"{resolved_clock_count:,}; millisecond delta range="
        f"[{clock_row['min_delta_ms']}, {clock_row['max_delta_ms']}]",
        flush=True,
    )
    context = build_strict_prior_global_context(history, args.max_sequence_len)
    corrected = (
        query_clocks.alias("q")
        .join(
            context.alias("c"),
            (F.col("q.user_id") == F.col("c.user_id"))
            & (F.col("q.current_time_ms") == F.col("c.time_ms")),
            "inner",
        )
        .select(
            "q.example_id",
            "q.user_id",
            "q.current_time_ms",
            *[F.col(f"c.{name}").alias(name) for name in CORRECTED_COLUMNS],
        )
    )
    output_path = args.output_dir / "sequences/split=test"
    print("writing corrected locked-query delta:", output_path, flush=True)
    (
        corrected.repartition(args.output_partitions, "user_id")
        .write.mode("overwrite")
        .option("compression", "zstd")
        .parquet(str(output_path))
    )
    result = spark.read.parquet(str(output_path))
    row = result.agg(
        F.count("*").alias("rows"),
        F.countDistinct("example_id").alias("distinct_examples"),
        F.sum(
            F.exists("recent_time_ms", lambda value: value >= F.col("current_time_ms")).cast("long")
        ).alias("time_violations"),
        F.sum(
            (F.size("recent_time_ms") != F.col("recent_event_count")).cast("long")
        ).alias("alignment_violations"),
        F.max("recent_event_count").alias("max_recent_events"),
    ).first()
    quality = {name: int(row[name] or 0) for name in row.asDict()}
    quality["expected_rows"] = int(len(protocol))
    quality["passed"] = bool(
        quality["rows"] == len(protocol)
        and quality["distinct_examples"] == len(protocol)
        and quality["time_violations"] == 0
        and quality["alignment_violations"] == 0
        and quality["max_recent_events"] <= args.max_sequence_len
    )
    if not quality["passed"]:
        raise AssertionError(f"Corrected test sequence quality failed: {quality}")
    manifest = {
        "version": "corrected_locked_test_queries_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "adaptive_eval_v2_test",
        "semantics": "complete current timestamp batch excluded before sequence cap",
        "max_sequence_len": args.max_sequence_len,
        "canonical_events": history_count,
        "query_clock": {
            "source": "canonical Gold SHA256(example||event_id||time_ms)",
            "resolved_queries": resolved_clock_count,
            "min_delta_from_protocol_ms": int(clock_row["min_delta_ms"]),
            "max_delta_from_protocol_ms": int(clock_row["max_delta_ms"]),
        },
        "quality": quality,
        "elapsed_seconds": time.perf_counter() - started,
        "artifact_path": "sequences/split=test",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(json.dumps(manifest, indent=2), flush=True)
    spark.stop()


if __name__ == "__main__":
    main()
