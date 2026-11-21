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

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from recommender.models.adaptive.checkpoint import (
    ITEM_CATEGORICAL,
    ITEM_NUMERIC,
    USER_CATEGORICAL,
    USER_NUMERIC,
)
from recommender.spark import get_spark


GLOBAL_ARRAYS = (
    "recent_time_ms",
    "recent_video_ids",
    "recent_target_classes",
    "recent_engagement_strengths",
    "recent_long_views",
    "recent_likes",
    "recent_hates",
    "recent_clicks",
)
SESSION_ARRAYS = tuple(f"session_{name}" for name in GLOBAL_ARRAYS)
ARRAY_TYPES = {
    "recent_time_ms": "array<long>",
    "recent_video_ids": "array<int>",
    "recent_target_classes": "array<string>",
    "recent_engagement_strengths": "array<double>",
    "recent_long_views": "array<int>",
    "recent_likes": "array<int>",
    "recent_hates": "array<int>",
    "recent_clicks": "array<int>",
    "session_recent_time_ms": "array<long>",
    "session_recent_video_ids": "array<int>",
    "session_recent_target_classes": "array<string>",
    "session_recent_engagement_strengths": "array<double>",
    "session_recent_long_views": "array<int>",
    "session_recent_likes": "array<int>",
    "session_recent_hates": "array<int>",
    "session_recent_clicks": "array<int>",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild only the missing adaptive test split from complete Two-Tower Gold."
    )
    parser.add_argument("--base-gold-root", type=Path, required=True)
    parser.add_argument("--sequence-context", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-partitions", type=int, default=96)
    parser.add_argument("--shuffle-partitions", type=int, default=96)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def select_available(frame: DataFrame, names: list[str]) -> list[str]:
    """Return requested columns that exist in the current Gold version."""
    existing = set(frame.columns)
    return [name for name in names if name in existing]


def empty_sequence_defaults(frame: DataFrame) -> DataFrame:
    """Represent first-event histories as typed empty arrays rather than nulls."""
    output = frame
    for name, data_type in ARRAY_TYPES.items():
        if name in output.columns:
            output = output.withColumn(
                name, F.coalesce(F.col(name), F.expr(f"cast(array() as {data_type})"))
            )
    return output


def enforce_strict_prior(frame: DataFrame) -> DataFrame:
    """Remove same/future timestamp events from every aligned sequence field."""
    output = empty_sequence_defaults(frame).withColumn(
        "as_of_time_ms", (F.unix_timestamp("as_of_time") * F.lit(1000)).cast("long")
    )
    recent_zip = "arrays_zip(" + ",".join(GLOBAL_ARRAYS) + ")"
    session_zip = "arrays_zip(" + ",".join(SESSION_ARRAYS) + ")"
    output = (
        output.withColumn(
            "_recent_safe",
            F.expr(f"filter({recent_zip}, x -> x.recent_time_ms < as_of_time_ms)"),
        )
        .withColumn(
            "_session_safe",
            F.expr(
                f"filter({session_zip}, x -> x.session_recent_time_ms < as_of_time_ms)"
            ),
        )
    )
    for name in GLOBAL_ARRAYS:
        output = output.withColumn(name, F.expr(f"transform(_recent_safe, x -> x.{name})"))
    for name in SESSION_ARRAYS:
        output = output.withColumn(name, F.expr(f"transform(_session_safe, x -> x.{name})"))
    return (
        output.withColumn("recent_event_count", F.size("recent_time_ms"))
        .withColumn("session_recent_event_count", F.size("session_recent_time_ms"))
        .withColumn("current_time_ms", F.col("as_of_time_ms"))
        .drop("_recent_safe", "_session_safe")
    )


def build_test_examples(base_root: Path, sequence_context_path: Path) -> DataFrame:
    """Join immutable test labels, user/item snapshots, and legacy sequence context."""
    spark = get_spark()
    targets = spark.read.parquet(str(base_root / "targets/test"))
    users = spark.read.parquet(str(base_root / "user_state/test"))
    items = spark.read.parquet(str(base_root / "item_features/point_in_time/test"))
    context = spark.read.parquet(str(sequence_context_path))

    user_columns = ["example_id", "user_id", "session_id", *USER_CATEGORICAL, *USER_NUMERIC]
    item_columns = ["example_id", "video_id", *ITEM_CATEGORICAL, *ITEM_NUMERIC]
    users = users.select(*select_available(users, user_columns))
    items = items.select(*select_available(items, item_columns))
    base = (
        targets.join(users, ["example_id", "user_id"], "left")
        .join(items, ["example_id", "video_id"], "left")
        .withColumn("split", F.lit("test"))
        .withColumn("retrieval_label", F.col("is_positive").cast("int"))
        .withColumn("ranker_label_strength", F.col("engagement_strength").cast("double"))
    )
    joined = (
        base.alias("b")
        .join(
            context.alias("s"),
            (F.col("b.user_id") == F.col("s.seq_user_id"))
            & (F.col("b.video_id") == F.col("s.current_video_id"))
            & (F.col("b.as_of_time") == F.col("s.current_event_ts")),
            "left",
        )
        .drop("seq_user_id", "current_video_id", "current_event_ts")
    )
    return enforce_strict_prior(joined)


def quality_report(frame: DataFrame, expected_rows: int) -> dict[str, object]:
    """Validate coverage, uniqueness, strict-prior arrays, and required feature joins."""
    row = frame.agg(
        F.count("*").alias("rows"),
        F.countDistinct("example_id").alias("distinct_examples"),
        F.countDistinct("user_id").alias("users"),
        F.countDistinct("video_id").alias("items"),
        F.sum(F.col("current_event_id").isNull().cast("long")).alias("missing_sequence_context"),
        F.sum(F.col("user_active_degree").isNull().cast("long")).alias("missing_user_features"),
        F.sum(F.col("video_type").isNull().cast("long")).alias("missing_item_features"),
        F.sum(
            F.exists("recent_time_ms", lambda value: value >= F.col("current_time_ms")).cast("long")
        ).alias("global_time_violations"),
        F.sum(
            F.exists(
                "session_recent_time_ms", lambda value: value >= F.col("current_time_ms")
            ).cast("long")
        ).alias("session_time_violations"),
        F.sum(F.col("is_positive").cast("long")).alias("positive_examples"),
    ).first()
    report = {name: int(row[name] or 0) for name in row.asDict()}
    report["expected_rows"] = int(expected_rows)
    report["passed"] = bool(
        report["rows"] == expected_rows
        and report["distinct_examples"] == expected_rows
        and report["missing_sequence_context"] == 0
        and report["missing_user_features"] == 0
        and report["missing_item_features"] == 0
        and report["global_time_violations"] == 0
        and report["session_time_violations"] == 0
        and report["positive_examples"] > 0
    )
    return report


def main() -> None:
    args = parse_args()
    required = [
        args.base_gold_root / "targets/test",
        args.base_gold_root / "user_state/test",
        args.base_gold_root / "item_features/point_in_time/test",
        args.sequence_context,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing test inputs:\n- " + "\n- ".join(missing))
    if args.output_dir.exists() and args.overwrite:
        shutil.rmtree(args.output_dir)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output exists; pass --overwrite: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("SPARK_MASTER", "local[2]")
    spark = get_spark("rebuild-adaptive-test", reset=True)
    spark.sparkContext.setLogLevel("WARN")
    spark.conf.set("spark.sql.shuffle.partitions", str(args.shuffle_partitions))
    started = time.perf_counter()
    expected_rows = spark.read.parquet(str(args.base_gold_root / "targets/test")).count()
    print(f"expected test examples: {expected_rows:,}", flush=True)
    examples = build_test_examples(args.base_gold_root, args.sequence_context)
    output_path = args.output_dir / "sequence_examples/test"
    print("writing adaptive test:", output_path, flush=True)
    (
        examples.repartition(args.output_partitions, "user_id")
        .write.mode("overwrite")
        .option("compression", "zstd")
        .parquet(str(output_path))
    )
    materialized = spark.read.parquet(str(output_path))
    quality = quality_report(materialized, expected_rows)
    if not quality["passed"]:
        raise AssertionError(f"Adaptive test quality gates failed: {quality}")
    manifest = {
        "version": "adaptive_test_v1_rebuilt",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_gold": str(args.base_gold_root),
        "source_sequence_context": str(args.sequence_context),
        "semantics": "legacy adaptive sequence contract with strict-prior sanitization",
        "quality": quality,
        "elapsed_seconds": time.perf_counter() - started,
        "artifact_path": "sequence_examples/test",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(json.dumps(manifest, indent=2), flush=True)
    spark.stop()


if __name__ == "__main__":
    main()
