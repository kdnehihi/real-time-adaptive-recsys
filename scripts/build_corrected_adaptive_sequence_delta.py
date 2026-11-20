from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pyspark import StorageLevel
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from recommender.data.adaptive_sequences import build_strict_prior_sequence_context
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
SEQUENCE_SCALARS = (
    "recent_event_count",
    "seconds_since_last_event",
    "recent_long_view_rate",
    "recent_like_rate",
    "recent_hate_rate",
    "recent_avg_engagement_strength",
    "session_recent_event_count",
    "session_seconds_since_last_event",
)
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
    parser = argparse.ArgumentParser(description="Build corrected strict-prior sequence delta Parquet.")
    parser.add_argument("--adaptive-root", type=Path, required=True)
    parser.add_argument("--base-gold-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--max-sequence-len", type=int, default=30)
    parser.add_argument("--max-session-sequence-len", type=int, default=10)
    parser.add_argument("--max-users", type=int, default=0)
    parser.add_argument("--shuffle-partitions", type=int, default=64)
    parser.add_argument("--output-partitions", type=int, default=96)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--rebuild-contexts", action="store_true")
    return parser.parse_args()


def example_id_expression() -> F.Column:
    """Reproduce the base Gold example identifier from canonical event fields."""
    return F.sha2(
        F.concat_ws(
            "||", F.lit("example"), F.col("event_id"), F.col("time_ms").cast("string")
        ),
        256,
    )


def load_split_keys(args: argparse.Namespace) -> DataFrame:
    """Load only immutable example/user keys from existing train/validation splits."""
    spark = get_spark()
    parts = []
    for split in ("train", "validation"):
        source = args.adaptive_root / "sequence_examples" / split
        if not source.exists():
            raise FileNotFoundError(source)
        parts.append(
            spark.read.parquet(str(source))
            .select("example_id", "user_id")
            .dropDuplicates(["example_id"])
            .withColumn("split", F.lit(split))
        )
    keys = parts[0].unionByName(parts[1])
    if args.max_users:
        users = keys.select("user_id").distinct().orderBy("user_id").limit(args.max_users)
        keys = keys.join(F.broadcast(users), "user_id", "inner")
    return keys.persist(StorageLevel.DISK_ONLY)


def load_history(args: argparse.Namespace, keys: DataFrame) -> DataFrame:
    """Load canonical events and bound smoke runs to their selected users."""
    source = args.base_gold_root / "history/events"
    if not source.exists():
        raise FileNotFoundError(source)
    history = get_spark().read.parquet(str(source)).select(*EVENT_COLUMNS)
    if args.max_users:
        users = keys.select("user_id").distinct()
        history = history.join(F.broadcast(users), "user_id", "inner")
    return history.dropDuplicates(["event_id"]).persist(StorageLevel.DISK_ONLY)


def materialize_contexts(args: argparse.Namespace, history: DataFrame) -> tuple[DataFrame, DataFrame]:
    """Build and checkpoint corrected contexts to break expensive window lineage."""
    global_path = args.work_dir / "global_context"
    session_path = args.work_dir / "session_context"
    if args.rebuild_contexts:
        shutil.rmtree(global_path, ignore_errors=True)
        shutil.rmtree(session_path, ignore_errors=True)
    if not (global_path / "_SUCCESS").exists() or not (session_path / "_SUCCESS").exists():
        print("building corrected timestamp-batch contexts...", flush=True)
        global_context, session_context = build_strict_prior_sequence_context(
            history, args.max_sequence_len, args.max_session_sequence_len
        )
        if not (global_path / "_SUCCESS").exists():
            global_context.write.mode("overwrite").parquet(str(global_path))
            print("materialized global context", flush=True)
        if not (session_path / "_SUCCESS").exists():
            session_context.write.mode("overwrite").parquet(str(session_path))
            print("materialized session context", flush=True)
    spark = get_spark()
    return spark.read.parquet(str(global_path)), spark.read.parquet(str(session_path))


def build_delta(
    keys: DataFrame,
    history: DataFrame,
    global_context: DataFrame,
    session_context: DataFrame,
) -> DataFrame:
    """Join corrected contexts to canonical events and retain only reusable sequence columns."""
    events = history.withColumn("example_id", example_id_expression()).select(
        "example_id", "user_id", "time_ms", "session_id"
    )
    selected_events = keys.join(events, ["example_id", "user_id"], "inner")
    global_columns = [*RECENT_ARRAY_COLUMNS, *SEQUENCE_SCALARS[:6]]
    session_columns = [*SESSION_ARRAY_COLUMNS, *SEQUENCE_SCALARS[6:]]
    return (
        selected_events.alias("e")
        .join(
            global_context.alias("g"),
            (F.col("e.user_id") == F.col("g.user_id"))
            & (F.col("e.time_ms") == F.col("g.time_ms")),
            "inner",
        )
        .join(
            session_context.alias("s"),
            (F.col("e.user_id") == F.col("s.user_id"))
            & (F.col("e.session_id") == F.col("s.session_id"))
            & (F.col("e.time_ms") == F.col("s.time_ms")),
            "inner",
        )
        .select(
            "e.split",
            "e.example_id",
            "e.user_id",
            F.col("e.time_ms").alias("current_time_ms"),
            *[F.col(f"g.{name}").alias(name) for name in global_columns],
            *[F.col(f"s.{name}").alias(name) for name in session_columns],
        )
    )


def quality_report(delta: DataFrame, keys: DataFrame, args: argparse.Namespace) -> dict[str, Any]:
    """Validate coverage, alignment, strict-prior time, caps, and uniqueness."""
    alignment_checks = [
        F.size(F.col(name)) == F.col("recent_event_count") for name in RECENT_ARRAY_COLUMNS
    ] + [
        F.size(F.col(name)) == F.col("session_recent_event_count")
        for name in SESSION_ARRAY_COLUMNS
    ]
    aligned = F.aggregate(
        F.array(*alignment_checks), F.lit(True), lambda total, value: total & value
    )
    strict_global = ~F.exists(
        "recent_time_ms", lambda value: value >= F.col("current_time_ms")
    )
    strict_session = ~F.exists(
        "session_recent_time_ms", lambda value: value >= F.col("current_time_ms")
    )
    rows = {
        row["split"]: {
            "expected": int(row["expected"]),
            "actual": int(row["actual"]),
        }
        for row in (
            keys.groupBy("split")
            .agg(F.count("*").alias("expected"))
            .join(delta.groupBy("split").agg(F.count("*").alias("actual")), "split", "full")
            .fillna(0)
            .collect()
        )
    }
    checks = delta.agg(
        F.count("*").alias("rows"),
        F.countDistinct("example_id").alias("distinct_examples"),
        F.sum((~aligned).cast("long")).alias("alignment_violations"),
        F.sum((~strict_global).cast("long")).alias("global_time_violations"),
        F.sum((~strict_session).cast("long")).alias("session_time_violations"),
        F.sum((F.col("recent_event_count") > args.max_sequence_len).cast("long")).alias("global_cap_violations"),
        F.sum((F.col("session_recent_event_count") > args.max_session_sequence_len).cast("long")).alias("session_cap_violations"),
    ).first()
    report = {
        "split_rows": rows,
        "rows": int(checks["rows"]),
        "distinct_examples": int(checks["distinct_examples"]),
        "alignment_violations": int(checks["alignment_violations"] or 0),
        "global_time_violations": int(checks["global_time_violations"] or 0),
        "session_time_violations": int(checks["session_time_violations"] or 0),
        "global_cap_violations": int(checks["global_cap_violations"] or 0),
        "session_cap_violations": int(checks["session_cap_violations"] or 0),
    }
    report["passed"] = (
        report["rows"] == report["distinct_examples"]
        and all(value["expected"] == value["actual"] for value in rows.values())
        and all(
            report[name] == 0
            for name in (
                "alignment_violations",
                "global_time_violations",
                "session_time_violations",
                "global_cap_violations",
                "session_cap_violations",
            )
        )
    )
    return report


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and args.overwrite:
        shutil.rmtree(args.output_dir)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output exists; pass --overwrite: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("SPARK_MASTER", "local[2]")
    spark = get_spark("build-corrected-adaptive-sequence-delta", reset=True)
    spark.sparkContext.setLogLevel("WARN")
    spark.conf.set("spark.sql.shuffle.partitions", str(args.shuffle_partitions))
    started = time.perf_counter()

    print("loading split keys...", flush=True)
    keys = load_split_keys(args)
    key_count = keys.count()
    user_count = keys.select("user_id").distinct().count()
    print(f"keys/users: {key_count:,}/{user_count:,}", flush=True)

    print("loading canonical history...", flush=True)
    history = load_history(args, keys)
    history_count = history.count()
    print(f"canonical events: {history_count:,}", flush=True)

    global_context, session_context = materialize_contexts(args, history)
    delta = build_delta(keys, history, global_context, session_context)
    output_path = args.output_dir / "sequences"
    print(f"writing corrected sequence delta: {output_path}", flush=True)
    (
        delta.repartition(args.output_partitions, "split", "user_id")
        .write.mode("overwrite")
        .partitionBy("split")
        .option("compression", "zstd")
        .parquet(str(output_path))
    )
    materialized = spark.read.parquet(str(output_path)).persist(StorageLevel.DISK_ONLY)
    quality = quality_report(materialized, keys, args)
    if not quality["passed"]:
        raise AssertionError(f"Corrected sequence delta quality checks failed: {quality}")
    manifest = {
        "version": "corrected_adaptive_sequences_v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_history": str(args.base_gold_root / "history/events"),
        "source_split_keys": str(args.adaptive_root / "sequence_examples/{train,validation}"),
        "semantics": "exclude complete current timestamp batch before applying sequence caps",
        "max_sequence_len": args.max_sequence_len,
        "max_session_sequence_len": args.max_session_sequence_len,
        "max_users": args.max_users,
        "users": user_count,
        "canonical_events": history_count,
        "quality": quality,
        "elapsed_seconds": time.perf_counter() - started,
        "artifact_path": "sequences/split={train,validation}",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(json.dumps(manifest, indent=2), flush=True)
    spark.stop()


if __name__ == "__main__":
    main()
