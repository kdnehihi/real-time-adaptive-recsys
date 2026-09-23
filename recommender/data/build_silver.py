from __future__ import annotations

import argparse
from functools import reduce
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from recommender.spark import get_spark


USER_ID = "user_id"
ITEM_ID = "video_id"
INTERACTION_COLS = [
    "is_click",
    "is_like",
    "is_follow",
    "is_comment",
    "is_forward",
    "is_hate",
    "long_view",
    "is_profile_enter",
]
ONEHOT_COLS = [f"onehot_feat{i}" for i in range(18)]


def read_bronze_tables(spark: SparkSession, bronze_dir: Path) -> dict[str, DataFrame]:
    if not bronze_dir.exists():
        raise FileNotFoundError(f"Bronze directory does not exist: {bronze_dir}")

    tables = {}
    for path in sorted(p for p in bronze_dir.iterdir() if p.is_dir()):
        tables[path.name] = spark.read.parquet(str(path))
    if not tables:
        raise FileNotFoundError(f"No bronze Parquet tables found in {bronze_dir}")
    return tables


def build_silver_interactions(tables: dict[str, DataFrame]) -> DataFrame:
    log_tables = [
        df.withColumn("source_table", F.lit(name))
        for name, df in sorted(tables.items())
        if name.startswith("log_")
    ]
    if not log_tables:
        raise ValueError("No log_* tables found in bronze input")

    logs = reduce(lambda left, right: left.unionByName(right, allowMissingColumns=True), log_tables)

    positive_expr = reduce(
        lambda left, right: left | right,
        [
            F.coalesce(F.col("is_click"), F.lit(0)) == 1,
            F.coalesce(F.col("is_like"), F.lit(0)) == 1,
            F.coalesce(F.col("is_follow"), F.lit(0)) == 1,
            F.coalesce(F.col("is_comment"), F.lit(0)) == 1,
            F.coalesce(F.col("is_forward"), F.lit(0)) == 1,
            F.coalesce(F.col("long_view"), F.lit(0)) == 1,
        ],
    )

    return (
        logs.withColumn(
            "event_id",
            F.sha2(
                F.concat_ws(
                    "||",
                    F.col(USER_ID).cast("string"),
                    F.col(ITEM_ID).cast("string"),
                    F.col("time_ms").cast("string"),
                    F.col("source_table"),
                    F.monotonically_increasing_id().cast("string"),
                ),
                256,
            ),
        )
        .withColumn("event_date", F.to_date(F.col("date").cast("string"), "yyyyMMdd"))
        .withColumn("event_ts", F.to_timestamp(F.from_unixtime((F.col("time_ms") / 1000).cast("long"))))
        .withColumn("hourmin_str", F.lpad(F.col("hourmin").cast("string"), 4, "0"))
        .withColumn("event_hour", F.substring("hourmin_str", 1, 2).cast("int"))
        .withColumn("event_minute", F.substring("hourmin_str", 3, 2).cast("int"))
        .withColumn("play_time_sec", F.col("play_time_ms") / F.lit(1000.0))
        .withColumn("duration_sec", F.col("duration_ms") / F.lit(1000.0))
        .withColumn(
            "watch_ratio",
            F.when(F.col("duration_ms") > 0, F.col("play_time_ms") / F.col("duration_ms")).otherwise(None),
        )
        .withColumn("is_valid_play", (F.coalesce(F.col("play_time_ms"), F.lit(0)) > 0).cast("int"))
        .withColumn("is_positive", positive_expr.cast("int"))
        .withColumn("is_negative", (F.coalesce(F.col("is_hate"), F.lit(0)) == 1).cast("int"))
        .select(
            "event_id",
            USER_ID,
            ITEM_ID,
            "event_ts",
            "event_date",
            "event_hour",
            "event_minute",
            "date",
            "hourmin",
            "time_ms",
            *INTERACTION_COLS,
            "play_time_ms",
            "duration_ms",
            "play_time_sec",
            "duration_sec",
            "watch_ratio",
            "profile_stay_time",
            "comment_stay_time",
            "is_valid_play",
            "is_positive",
            "is_negative",
            "is_rand",
            "tab",
            "source_table",
        )
    )


def build_silver_users(tables: dict[str, DataFrame]) -> DataFrame:
    user_names = sorted(name for name in tables if name.startswith("user_features"))
    if not user_names:
        raise ValueError("No user_features* table found in bronze input")

    users = reduce(lambda left, right: left.unionByName(right, allowMissingColumns=True), [tables[n] for n in user_names])
    users = users.dropDuplicates([USER_ID])

    for col in ONEHOT_COLS:
        if col in users.columns:
            users = users.withColumn(f"{col}_is_missing", F.col(col).isNull().cast("int"))
            users = users.withColumn(col, F.coalesce(F.col(col), F.lit(0.0)))

    return users


def _safe_ratio(numerator: str, denominator: str) -> F.Column:
    return F.when(F.col(denominator) > 0, F.col(numerator) / F.col(denominator)).otherwise(None)


def build_silver_videos_basic(tables: dict[str, DataFrame]) -> DataFrame:
    basic_names = sorted(name for name in tables if name.startswith("video_features_basic"))
    if not basic_names:
        raise ValueError("No video_features_basic* table found in bronze input")

    basic = reduce(lambda left, right: left.unionByName(right, allowMissingColumns=True), [tables[n] for n in basic_names])

    missing_indicator_cols = [
        "upload_dt",
        "visible_status",
        "video_duration",
        "server_width",
        "server_height",
        "music_type",
        "tag",
    ]
    for col in missing_indicator_cols:
        if col in basic.columns:
            basic = basic.withColumn(f"{col}_is_missing", F.col(col).isNull().cast("int"))

    basic = (
        basic.withColumn("upload_date", F.to_date("upload_dt"))
        .withColumn("video_duration_sec", F.col("video_duration") / F.lit(1000.0))
        .withColumn(
            "aspect_ratio",
            F.when(F.col("server_height") > 0, F.col("server_width") / F.col("server_height")).otherwise(None),
        )
    )

    return basic


def build_silver_videos_statistics(tables: dict[str, DataFrame]) -> DataFrame:
    statistic_names = sorted(name for name in tables if name.startswith("video_features_statistic"))
    if not statistic_names:
        raise ValueError("No video_features_statistic* table found in bronze input")

    stats = reduce(lambda left, right: left.unionByName(right, allowMissingColumns=True), [tables[n] for n in statistic_names])

    derived_ratio_cols = {
        "play_per_show": ("play_cnt", "show_cnt"),
        "valid_play_rate": ("valid_play_cnt", "play_cnt"),
        "long_play_rate": ("long_time_play_cnt", "play_cnt"),
        "like_rate": ("like_cnt", "play_cnt"),
        "comment_rate": ("comment_cnt", "play_cnt"),
        "share_rate": ("share_cnt", "play_cnt"),
        "follow_rate": ("follow_cnt", "play_cnt"),
    }
    for output_col, (numerator, denominator) in derived_ratio_cols.items():
        if numerator in stats.columns and denominator in stats.columns:
            stats = stats.withColumn(output_col, _safe_ratio(numerator, denominator))

    return stats


def write_silver_tables(
    bronze_dir: Path,
    silver_dir: Path,
    overwrite: bool = False,
    tables_to_build: list[str] | None = None,
) -> None:
    spark = get_spark("kuairand-build-silver", reset=True)
    tables = read_bronze_tables(spark, bronze_dir)

    mode = "overwrite" if overwrite else "errorifexists"
    silver_dir.mkdir(parents=True, exist_ok=True)

    builders = {
        "interactions": build_silver_interactions,
        "users": build_silver_users,
        "videos_basic": build_silver_videos_basic,
        "videos_statistics": build_silver_videos_statistics,
    }
    requested = tables_to_build or list(builders)
    unknown = sorted(set(requested) - set(builders))
    if unknown:
        raise ValueError(f"Unknown silver table(s): {unknown}. Available: {sorted(builders)}")

    for name in requested:
        df = builders[name](tables)
        target = silver_dir / name
        df.write.mode(mode).parquet(str(target))
        print(f"Wrote {target}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build cleaned KuaiRand silver tables from bronze Parquet.")
    parser.add_argument("--bronze-dir", type=Path, default=Path("data/bronze/kuairand"))
    parser.add_argument("--silver-dir", type=Path, default=Path("data/silver/kuairand"))
    parser.add_argument(
        "--tables",
        nargs="+",
        choices=["interactions", "users", "videos_basic", "videos_statistics"],
        help="Optional subset of silver tables to build.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_silver_tables(args.bronze_dir, args.silver_dir, overwrite=args.overwrite, tables_to_build=args.tables)


if __name__ == "__main__":
    main()
