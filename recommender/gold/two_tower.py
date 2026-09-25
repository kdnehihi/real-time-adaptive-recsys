from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from pyspark import StorageLevel
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from recommender.gold.als_baseline import git_commit, split_events, split_summary, write_json, write_parquet
from recommender.spark import get_spark


DATASET_VERSION = "two_tower_v1"
FEATURE_CATALOG_VERSION = "two_tower_feature_catalog_v1"
TARGET_DEFINITION_VERSION = "direct_feedback_v1"
HISTORY_DEFINITION_VERSION = "user_chronological_30m_session_v1"
UNKNOWN_TOKEN = "__UNKNOWN__"

USER_ID = "user_id"
ITEM_ID = "video_id"

DIRECT_FEEDBACK_COLS = [
    "long_view",
    "is_like",
    "is_comment",
    "is_forward",
    "is_follow",
    "is_hate",
    "is_click",
    "is_profile_enter",
]

USER_STATIC_NUMERIC = [
    "is_lowactive_period",
    "is_live_streamer",
    "is_video_author",
    "follow_user_num",
    "fans_user_num",
    "friend_user_num",
    "register_days",
]
USER_STATIC_CATEGORICAL = ["user_active_degree"]
USER_ANON_NUMERIC = [f"onehot_feat{i}" for i in range(18)]

USER_HISTORY_NUMERIC = [
    "user_hist_events",
    "user_hist_long_view_rate",
    "user_hist_like_rate",
    "user_hist_comment_rate",
    "user_hist_forward_rate",
    "user_hist_follow_rate",
    "user_hist_hate_rate",
    "user_hist_avg_watch_ratio",
    "user_hist_avg_play_time_sec",
    "session_event_index",
    "session_elapsed_sec",
    "session_prior_long_view_rate",
    "session_prior_like_rate",
    "session_prior_hate_rate",
    "session_prior_avg_watch_ratio",
    "session_vs_user_long_view_delta",
    "session_vs_user_watch_ratio_delta",
]
CONTEXT_FEATURES = ["event_hour", "event_dayofweek", "tab", "is_rand"]

ITEM_STATIC_NUMERIC = [
    "video_duration_sec",
    "aspect_ratio",
    "visible_status",
    "music_type",
    "upload_age_days_at_event",
]
ITEM_STATIC_CATEGORICAL = ["video_type", "upload_type"]
ITEM_HISTORY_NUMERIC = [
    "item_hist_events",
    "item_hist_long_view_rate",
    "item_hist_like_rate",
    "item_hist_hate_rate",
    "item_hist_avg_watch_ratio",
]

TARGET_COLS = [
    "target_class",
    "is_positive",
    "is_observed_negative",
    "is_ambiguous",
    "engagement_strength",
    "watch_ratio_clipped",
    *DIRECT_FEEDBACK_COLS,
    "play_time_ms",
    "duration_ms",
    "watch_ratio",
]


def storage_level_from_env() -> StorageLevel:
    raw = os.getenv("TWO_TOWER_STORAGE_LEVEL", "DISK_ONLY").upper()
    choices = {
        "DISK_ONLY": StorageLevel.DISK_ONLY,
        "MEMORY_ONLY": StorageLevel.MEMORY_ONLY,
        "MEMORY_AND_DISK": StorageLevel.MEMORY_AND_DISK,
    }
    if raw not in choices:
        raise ValueError(f"Unknown TWO_TOWER_STORAGE_LEVEL={raw}. Use one of {sorted(choices)}")
    return choices[raw]


def read_silver_tables(spark: SparkSession, silver_dir: Path) -> dict[str, DataFrame]:
    required = ["interactions", "users", "videos_basic"]
    tables = {}
    for name in required:
        path = silver_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Missing silver table: {path}")
        tables[name] = spark.read.parquet(str(path))
    return tables


def _require_columns(df: DataFrame, required: Iterable[str], table_name: str) -> None:
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"{table_name} is missing required columns: {missing}")


def validate_silver_schema(tables: dict[str, DataFrame]) -> None:
    _require_columns(
        tables["interactions"],
        [
            "event_id",
            "user_id",
            "video_id",
            "event_ts",
            "time_ms",
            "event_hour",
            "watch_ratio",
            "play_time_ms",
            "duration_ms",
            "source_table",
            *DIRECT_FEEDBACK_COLS,
            "tab",
            "is_rand",
        ],
        "silver interactions",
    )
    _require_columns(tables["users"], ["user_id", *USER_STATIC_CATEGORICAL], "silver users")
    _require_columns(
        tables["videos_basic"],
        ["video_id", "video_type", "upload_type", "upload_date", "video_duration_sec", "aspect_ratio"],
        "silver videos_basic",
    )


def add_targets(events: DataFrame, weak_watch_ratio_threshold: float = 0.20) -> DataFrame:
    positive_expr = (
        (F.coalesce(F.col("long_view"), F.lit(0)) == 1)
        | (F.coalesce(F.col("is_like"), F.lit(0)) == 1)
        | (F.coalesce(F.col("is_comment"), F.lit(0)) == 1)
        | (F.coalesce(F.col("is_forward"), F.lit(0)) == 1)
        | (F.coalesce(F.col("is_follow"), F.lit(0)) == 1)
    )
    watch_ratio_clipped = F.least(F.greatest(F.coalesce(F.col("watch_ratio"), F.lit(0.0)), F.lit(0.0)), F.lit(3.0))
    observed_negative_expr = (
        (~positive_expr)
        & (
            (F.coalesce(F.col("is_hate"), F.lit(0)) == 1)
            | (
                (F.coalesce(F.col("is_click"), F.lit(0)) == 1)
                & (watch_ratio_clipped <= F.lit(weak_watch_ratio_threshold))
            )
        )
    )
    engagement_strength = (
        F.lit(0.5) * watch_ratio_clipped
        + F.lit(1.0) * F.coalesce(F.col("long_view"), F.lit(0)).cast("double")
        + F.lit(1.5) * F.coalesce(F.col("is_like"), F.lit(0)).cast("double")
        + F.lit(1.5) * F.coalesce(F.col("is_comment"), F.lit(0)).cast("double")
        + F.lit(1.5) * F.coalesce(F.col("is_forward"), F.lit(0)).cast("double")
        + F.lit(2.0) * F.coalesce(F.col("is_follow"), F.lit(0)).cast("double")
        - F.lit(1.0) * F.coalesce(F.col("is_hate"), F.lit(0)).cast("double")
    )
    return (
        events.withColumn("watch_ratio_clipped", watch_ratio_clipped)
        .withColumn("is_positive", positive_expr.cast("int"))
        .withColumn("is_observed_negative", observed_negative_expr.cast("int"))
        .withColumn(
            "target_class",
            F.when(positive_expr, F.lit("STRONG_POSITIVE"))
            .when(observed_negative_expr, F.lit("OBSERVED_NEGATIVE"))
            .otherwise(F.lit("AMBIGUOUS_WEAK")),
        )
        .withColumn("is_ambiguous", (F.col("target_class") == "AMBIGUOUS_WEAK").cast("int"))
        .withColumn("engagement_strength", engagement_strength)
    )


def add_point_in_time_features(events: DataFrame) -> DataFrame:
    user_order = Window.partitionBy("user_id").orderBy("time_ms", "event_id")
    user_prev = user_order.rowsBetween(Window.unboundedPreceding, -1)
    user_cur = user_order.rowsBetween(Window.unboundedPreceding, Window.currentRow)
    item_order = Window.partitionBy("video_id").orderBy("time_ms", "event_id")
    item_prev = item_order.rowsBetween(Window.unboundedPreceding, -1)

    with_session_seed = (
        events.withColumn("prev_event_ts", F.lag("event_ts").over(user_order))
        .withColumn("prev_video_id", F.lag("video_id").over(user_order))
        .withColumn("user_event_index", F.row_number().over(user_order))
        .withColumn("gap_sec", F.col("event_ts").cast("long") - F.col("prev_event_ts").cast("long"))
        .withColumn(
            "new_session_flag",
            F.when(F.col("prev_event_ts").isNull() | (F.col("gap_sec") > 30 * 60), F.lit(1)).otherwise(F.lit(0)),
        )
        .withColumn("session_seq", F.sum("new_session_flag").over(user_cur))
        .withColumn("session_id", F.concat_ws("_", F.col("user_id").cast("string"), F.col("session_seq").cast("string")))
    )
    session_order = Window.partitionBy("user_id", "session_seq").orderBy("time_ms", "event_id")
    session_prev = session_order.rowsBetween(Window.unboundedPreceding, -1)
    session_cur = session_order.rowsBetween(Window.unboundedPreceding, Window.currentRow)

    with_history = (
        with_session_seed.withColumn("session_event_index", F.row_number().over(session_order))
        .withColumn("session_start_ts", F.first("event_ts").over(session_cur))
        .withColumn("session_elapsed_sec", F.col("event_ts").cast("long") - F.col("session_start_ts").cast("long"))
        .withColumn("user_hist_events", F.count(F.lit(1)).over(user_prev))
        .withColumn("user_hist_long_view_rate", F.avg(F.coalesce("long_view", F.lit(0)).cast("double")).over(user_prev))
        .withColumn("user_hist_like_rate", F.avg(F.coalesce("is_like", F.lit(0)).cast("double")).over(user_prev))
        .withColumn("user_hist_comment_rate", F.avg(F.coalesce("is_comment", F.lit(0)).cast("double")).over(user_prev))
        .withColumn("user_hist_forward_rate", F.avg(F.coalesce("is_forward", F.lit(0)).cast("double")).over(user_prev))
        .withColumn("user_hist_follow_rate", F.avg(F.coalesce("is_follow", F.lit(0)).cast("double")).over(user_prev))
        .withColumn("user_hist_hate_rate", F.avg(F.coalesce("is_hate", F.lit(0)).cast("double")).over(user_prev))
        .withColumn("user_hist_avg_watch_ratio", F.avg("watch_ratio_clipped").over(user_prev))
        .withColumn("user_hist_avg_play_time_sec", F.avg("play_time_sec").over(user_prev))
        .withColumn("session_prior_long_view_rate", F.avg(F.coalesce("long_view", F.lit(0)).cast("double")).over(session_prev))
        .withColumn("session_prior_like_rate", F.avg(F.coalesce("is_like", F.lit(0)).cast("double")).over(session_prev))
        .withColumn("session_prior_hate_rate", F.avg(F.coalesce("is_hate", F.lit(0)).cast("double")).over(session_prev))
        .withColumn("session_prior_avg_watch_ratio", F.avg("watch_ratio_clipped").over(session_prev))
        .withColumn("item_hist_events", F.count(F.lit(1)).over(item_prev))
        .withColumn("item_hist_long_view_rate", F.avg(F.coalesce("long_view", F.lit(0)).cast("double")).over(item_prev))
        .withColumn("item_hist_like_rate", F.avg(F.coalesce("is_like", F.lit(0)).cast("double")).over(item_prev))
        .withColumn("item_hist_hate_rate", F.avg(F.coalesce("is_hate", F.lit(0)).cast("double")).over(item_prev))
        .withColumn("item_hist_avg_watch_ratio", F.avg("watch_ratio_clipped").over(item_prev))
        .withColumn(
            "session_vs_user_long_view_delta",
            F.col("session_prior_long_view_rate") - F.col("user_hist_long_view_rate"),
        )
        .withColumn(
            "session_vs_user_watch_ratio_delta",
            F.col("session_prior_avg_watch_ratio") - F.col("user_hist_avg_watch_ratio"),
        )
        .withColumn("history_end_user_event_index", F.col("user_event_index") - F.lit(1))
        .withColumn("history_end_session_event_index", F.col("session_event_index") - F.lit(1))
    )
    return with_history


def build_examples(
    interactions: DataFrame,
    users: DataFrame,
    videos: DataFrame,
    train_quantile: float,
    validation_quantile: float,
    weak_watch_ratio_threshold: float,
) -> tuple[DataFrame, dict]:
    events, split_meta = split_events(interactions, train_quantile, validation_quantile)
    events = add_targets(events, weak_watch_ratio_threshold)
    events = add_point_in_time_features(events)

    users_keep = ["user_id", *USER_STATIC_CATEGORICAL, *USER_STATIC_NUMERIC, *USER_ANON_NUMERIC]
    users_keep = [c for c in users_keep if c in users.columns]
    videos_keep = [
        "video_id",
        "author_id",
        *ITEM_STATIC_CATEGORICAL,
        "upload_date",
        "video_duration_is_missing",
        "server_width_is_missing",
        "server_height_is_missing",
        "music_type_is_missing",
        "tag_is_missing",
        *[c for c in ITEM_STATIC_NUMERIC if c != "upload_age_days_at_event"],
    ]
    videos_keep = [c for c in videos_keep if c in videos.columns]

    joined = events.join(F.broadcast(users.select(*users_keep)), on="user_id", how="left").join(
        videos.select(*videos_keep), on="video_id", how="left"
    )
    joined = joined.withColumn("event_dayofweek", F.dayofweek("event_ts")).withColumn(
        "upload_age_days_at_event", F.datediff(F.to_date("event_ts"), F.col("upload_date")).cast("double")
    )
    joined = joined.withColumn(
        "example_id",
        F.sha2(F.concat_ws("||", F.lit("example"), F.col("event_id"), F.col("time_ms").cast("string")), 256),
    ).withColumn(
        "context_id",
        F.sha2(
            F.concat_ws(
                "||",
                F.lit("context"),
                F.col("user_id").cast("string"),
                F.col("history_end_user_event_index").cast("string"),
                F.col("session_id"),
                F.col("history_end_session_event_index").cast("string"),
            ),
            256,
        ),
    )
    joined = joined.withColumn("as_of_time", F.col("event_ts"))
    return joined, split_meta


def split_counts(examples: DataFrame) -> list[dict]:
    rows = (
        examples.groupBy("split")
        .agg(
            F.count("*").alias("examples"),
            F.countDistinct("user_id").alias("users"),
            F.countDistinct("video_id").alias("items"),
            F.sum((F.col("target_class") == "STRONG_POSITIVE").cast("long")).alias("strong_positive"),
            F.sum((F.col("target_class") == "OBSERVED_NEGATIVE").cast("long")).alias("observed_negative"),
            F.sum((F.col("target_class") == "AMBIGUOUS_WEAK").cast("long")).alias("ambiguous_weak"),
        )
        .orderBy("split")
        .collect()
    )
    return [
        {
            "split": r["split"],
            "examples": int(r["examples"]),
            "users": int(r["users"]),
            "items": int(r["items"]),
            "target_counts": {
                "STRONG_POSITIVE": int(r["strong_positive"] or 0),
                "OBSERVED_NEGATIVE": int(r["observed_negative"] or 0),
                "AMBIGUOUS_WEAK": int(r["ambiguous_weak"] or 0),
            },
        }
        for r in rows
    ]


def add_cold_start_flags(examples: DataFrame) -> DataFrame:
    train_users = examples.where(F.col("split") == "train").select("user_id").distinct().withColumn("seen_user_in_train", F.lit(1))
    train_items = examples.where(F.col("split") == "train").select("video_id").distinct().withColumn("seen_item_in_train", F.lit(1))
    return (
        examples.join(F.broadcast(train_users), on="user_id", how="left")
        .join(train_items, on="video_id", how="left")
        .withColumn("is_warm_user", F.coalesce(F.col("seen_user_in_train"), F.lit(0)).cast("boolean"))
        .withColumn("is_warm_item", F.coalesce(F.col("seen_item_in_train"), F.lit(0)).cast("boolean"))
        .withColumn("has_user_history", (F.col("history_end_user_event_index") > 0).cast("boolean"))
        .withColumn("has_item_metadata", F.col("video_type").isNotNull())
        .drop("seen_user_in_train", "seen_item_in_train")
    )


def cold_start_stats(examples: DataFrame) -> dict[str, dict]:
    rows = (
        examples.groupBy("split")
        .agg(
            F.count("*").alias("examples"),
            F.avg(F.col("is_warm_user").cast("double")).alias("warm_user_rate"),
            F.avg(F.col("is_warm_item").cast("double")).alias("warm_item_rate"),
            F.avg(F.col("has_user_history").cast("double")).alias("has_user_history_rate"),
            F.avg(F.col("has_item_metadata").cast("double")).alias("has_item_metadata_rate"),
        )
        .collect()
    )
    stats = {split: {"examples": 0} for split in ["train", "validation", "test"]}
    for row in rows:
        stats[row["split"]] = {
            "examples": int(row["examples"] or 0),
            "warm_user_rate": float(row["warm_user_rate"] or 0.0),
            "warm_item_rate": float(row["warm_item_rate"] or 0.0),
            "cold_user_rate": float(1.0 - (row["warm_user_rate"] or 0.0)),
            "cold_item_rate": float(1.0 - (row["warm_item_rate"] or 0.0)),
            "has_user_history_rate": float(row["has_user_history_rate"] or 0.0),
            "has_item_metadata_rate": float(row["has_item_metadata_rate"] or 0.0),
        }
    return stats


def build_vocabulary(df: DataFrame, column: str) -> DataFrame:
    window = Window.orderBy(column)
    return (
        df.where(F.col(column).isNotNull())
        .select(F.col(column).cast("string").alias(column))
        .distinct()
        .withColumn(f"{column}_idx", F.row_number().over(window))
    )


def vocabulary_stats(train_examples: DataFrame, categorical_cols: list[str]) -> dict:
    existing = [c for c in categorical_cols if c in train_examples.columns]
    if not existing:
        return {}

    row = train_examples.agg(*[F.countDistinct(F.col(c)).alias(c) for c in existing]).first()
    return {
        col: {"fit_split": "train", "cardinality_excluding_oov": int(row[col] or 0)}
        for col in existing
    }


def numerical_transform_stats(train_examples: DataFrame, numeric_cols: list[str]) -> dict:
    existing = [c for c in numeric_cols if c in train_examples.columns]
    if not existing:
        return {}

    agg_exprs = []
    for col in existing:
        value = F.col(col).cast("double")
        agg_exprs.extend(
            [
                F.count(value).alias(f"{col}__non_null"),
                F.mean(value).alias(f"{col}__mean"),
                F.stddev(value).alias(f"{col}__stddev"),
                F.min(value).alias(f"{col}__min"),
                F.percentile_approx(value, [0.01, 0.5, 0.99], 1000).alias(f"{col}__quantiles"),
                F.max(value).alias(f"{col}__max"),
            ]
        )

    row = train_examples.agg(*agg_exprs).first()
    stats = {}
    for col in existing:
        qs = row[f"{col}__quantiles"] or [None, None, None]
        stats[col] = {
            "fit_split": "train",
            "non_null": int(row[f"{col}__non_null"] or 0),
            "mean": float(row[f"{col}__mean"]) if row[f"{col}__mean"] is not None else None,
            "stddev": float(row[f"{col}__stddev"]) if row[f"{col}__stddev"] is not None else None,
            "min": float(row[f"{col}__min"]) if row[f"{col}__min"] is not None else None,
            "p01": float(qs[0]) if qs[0] is not None else None,
            "median": float(qs[1]) if qs[1] is not None else None,
            "p99": float(qs[2]) if qs[2] is not None else None,
            "max": float(row[f"{col}__max"]) if row[f"{col}__max"] is not None else None,
        }
    return stats


def build_feature_catalog(total_rows: int | None = None) -> list[dict]:
    rows: list[dict] = []

    def add(name: str, source: str, description: str, assignment: str, state: str, pit: str, leakage: str, transform: str, reason: str) -> None:
        rows.append(
            {
                "feature_name": name,
                "source": source,
                "semantic_description": description,
                "tower_assignment": assignment,
                "static_vs_dynamic": state,
                "point_in_time_availability": pit,
                "leakage_risk": leakage,
                "missingness_or_coverage": "computed in transforms/profile outputs" if total_rows is None else "see transform statistics",
                "cardinality": "see vocabularies for categorical features",
                "transformation_needed": transform,
                "serving_time_feasibility": "feasible if maintained in online/session feature store",
                "reason": reason,
            }
        )

    for c in USER_STATIC_NUMERIC + USER_ANON_NUMERIC:
        add(c, "silver.users", "static user profile/anonymous attribute", "USER_TOWER", "static snapshot", "known before request", "LOW/MEDIUM", "impute + normalize from train stats", "selected as user-side prior where available")
    for c in USER_STATIC_CATEGORICAL:
        add(c, "silver.users", "categorical user activity segment", "USER_TOWER", "static snapshot", "known before request", "LOW/MEDIUM", "train-only vocabulary with OOV", "kept for cold-start user prior")
    for c in USER_HISTORY_NUMERIC:
        add(c, "silver.interactions", "strictly prior user/session behavior aggregate", "USER_TOWER", "dynamic point-in-time", "computed with rows before current event only", "LOW", "impute missing cold-start values + normalize", "selected for adaptive short-term and long-term preference")
    for c in CONTEXT_FEATURES:
        add(c, "silver.interactions", "request/logging context known at impression time", "USER_TOWER", "request context", "known at request/event time", "LOW/MEDIUM", "categorical or numeric encoding later", "selected as context available to user tower")
    for c in ITEM_STATIC_NUMERIC:
        add(c, "silver.videos_basic", "item metadata independent of current user", "ITEM_TOWER", "static/slowly changing", "known before recommendation if metadata exists", "LOW", "impute + normalize from train stats", "selected for item cold-start and embedding precompute")
    for c in ITEM_STATIC_CATEGORICAL:
        add(c, "silver.videos_basic", "categorical item metadata", "ITEM_TOWER", "static/slowly changing", "known before recommendation if metadata exists", "LOW", "train-only vocabulary with OOV", "selected for item content signal")
    for c in ITEM_HISTORY_NUMERIC:
        add(c, "silver.interactions", "prior item engagement aggregate independent of current user", "ITEM_TOWER", "dynamic point-in-time", "computed with item rows before current event only", "LOW", "impute cold-item values + normalize", "selected as precomputable item popularity/quality signal")
    for c in ["watch_ratio", "play_time_ms", "duration_ms", *DIRECT_FEEDBACK_COLS]:
        add(c, "silver.interactions", "current-event outcome/direct feedback", "TARGET_ONLY", "post-event", "available only after current interaction", "HIGH if used as feature", "preserve raw", "target construction only")
    for c in ["show_cnt", "play_per_show", "like_rate", "comment_rate", "share_rate", "follow_rate"]:
        add(c, "silver.videos_statistics", "global aggregate snapshot with unclear as-of timestamp", "DROP", "unknown snapshot", "not guaranteed point-in-time", "HIGH/UNKNOWN", "none in v1", "dropped until snapshot semantics are verified or rebuilt historically")
    add("tag", "silver.videos_basic", "raw multi-value text/tag field", "RESERVED_FOR_RANKER", "static metadata", "known if metadata exists", "LOW", "parse/tokenize later", "reserved because v1 avoids text/multivalue processing")
    add("user_item_similarity", "future feature store", "cross feature depending on current user and item", "RESERVED_FOR_RANKER", "dynamic cross", "requires both towers", "N/A", "none", "not valid for retrieval tower decomposition")
    return rows


def quality_checks(examples: DataFrame) -> dict:
    duplicate_examples = examples.groupBy("example_id").count().where(F.col("count") > 1).count()
    row = examples.agg(
        F.sum(
            (
                (F.col("history_end_user_event_index") >= F.col("user_event_index"))
                | (F.col("history_end_session_event_index") >= F.col("session_event_index"))
            ).cast("long")
        ).alias("bad_history"),
        F.sum((F.col("user_hist_events") != F.col("history_end_user_event_index")).cast("long")).alias(
            "bad_user_history_count"
        ),
        F.sum(
            ((F.col("history_end_user_event_index") == 0) & (F.col("user_hist_events") != 0)).cast("long")
        ).alias("current_item_in_first_history"),
        F.sum(F.col("time_ms").isNull().cast("long")).alias("null_time_rows"),
    ).first()
    bad_history = int(row["bad_history"] or 0)
    bad_user_history_count = int(row["bad_user_history_count"] or 0)
    current_item_in_first_history = int(row["current_item_in_first_history"] or 0)
    null_time_rows = int(row["null_time_rows"] or 0)
    return {
        "duplicate_example_ids": int(duplicate_examples),
        "bad_history_index_rows": bad_history,
        "bad_user_history_count_rows": bad_user_history_count,
        "first_history_contains_rows": current_item_in_first_history,
        "null_time_rows": null_time_rows,
        "passed": duplicate_examples == 0 and bad_history == 0 and bad_user_history_count == 0 and current_item_in_first_history == 0 and null_time_rows == 0,
    }


def write_gold_dataset(args: argparse.Namespace) -> dict:
    spark = get_spark("kuairand-build-two-tower-gold", reset=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tables = read_silver_tables(spark, args.silver_dir)
    validate_silver_schema(tables)

    examples, split_meta = build_examples(
        tables["interactions"],
        tables["users"],
        tables["videos_basic"],
        args.train_quantile,
        args.validation_quantile,
        args.weak_watch_ratio_threshold,
    )
    storage_level = storage_level_from_env()
    examples = add_cold_start_flags(examples).persist(storage_level)

    categorical_cols = [c for c in USER_STATIC_CATEGORICAL + ITEM_STATIC_CATEGORICAL if c in examples.columns]
    numeric_cols = [
        c
        for c in USER_STATIC_NUMERIC
        + USER_ANON_NUMERIC
        + USER_HISTORY_NUMERIC
        + CONTEXT_FEATURES
        + ITEM_STATIC_NUMERIC
        + ITEM_HISTORY_NUMERIC
        if c in examples.columns
    ]
    train_examples = examples.where(F.col("split") == "train").persist(storage_level)

    vocab_stats = vocabulary_stats(train_examples, categorical_cols)
    transform_stats = numerical_transform_stats(train_examples, numeric_cols)
    feature_catalog = build_feature_catalog()
    checks = quality_checks(examples)
    if not checks["passed"]:
        raise AssertionError(f"Two-tower gold quality checks failed: {checks}")

    base_cols = [
        "example_id",
        "context_id",
        "event_id",
        "user_id",
        "video_id",
        "session_id",
        "user_event_index",
        "session_event_index",
        "history_end_user_event_index",
        "history_end_session_event_index",
        "as_of_time",
        "time_ms",
        "split",
        "is_warm_user",
        "is_warm_item",
        "has_user_history",
        "has_item_metadata",
    ]
    user_cols = [c for c in USER_STATIC_CATEGORICAL + USER_STATIC_NUMERIC + USER_ANON_NUMERIC + USER_HISTORY_NUMERIC + CONTEXT_FEATURES if c in examples.columns]
    item_cols = [c for c in ITEM_STATIC_CATEGORICAL + ITEM_STATIC_NUMERIC + ITEM_HISTORY_NUMERIC if c in examples.columns]
    target_cols = [c for c in TARGET_COLS if c in examples.columns]
    split_output_cols = base_cols

    for split in ["train", "validation", "test"]:
        write_parquet(examples.where(F.col("split") == split).select(*split_output_cols), args.output_dir / split, args.overwrite)
        write_parquet(
            examples.where(F.col("split") == split).select("example_id", "context_id", "user_id", "video_id", "as_of_time", *target_cols),
            args.output_dir / "targets" / split,
            args.overwrite,
        )
        write_parquet(
            examples.where(F.col("split") == split).select("example_id", "context_id", "user_id", "session_id", *user_cols),
            args.output_dir / "user_state" / split,
            args.overwrite,
        )
        write_parquet(
            examples.where(F.col("split") == split).select("example_id", "video_id", "as_of_time", *item_cols),
            args.output_dir / "item_features" / "point_in_time" / split,
            args.overwrite,
        )

    item_feature_cols = ["video_id", "author_id", *ITEM_STATIC_CATEGORICAL, "upload_date", *[c for c in ITEM_STATIC_NUMERIC if c != "upload_age_days_at_event"]]
    item_feature_cols = [c for c in item_feature_cols if c in tables["videos_basic"].columns]
    write_parquet(
        tables["videos_basic"].select(*item_feature_cols).dropDuplicates(["video_id"]),
        args.output_dir / "item_features" / "static",
        args.overwrite,
    )

    history_cols = [
        "event_id",
        "user_id",
        "video_id",
        "event_ts",
        "time_ms",
        "session_id",
        "user_event_index",
        "session_event_index",
        "target_class",
        "engagement_strength",
        *[c for c in DIRECT_FEEDBACK_COLS if c in examples.columns],
    ]
    write_parquet(examples.select(*history_cols), args.output_dir / "history" / "events", args.overwrite)

    vocab_dir = args.output_dir / "vocabularies"
    for col in categorical_cols:
        write_parquet(build_vocabulary(train_examples, col), vocab_dir / col, args.overwrite)
    transforms_dir = args.output_dir / "transforms"
    transforms_dir.mkdir(parents=True, exist_ok=True)
    write_json({"version": "numeric_transforms_v1", "fit_split": "train", "features": transform_stats}, transforms_dir / "numeric_stats.json")
    write_json({"version": "categorical_vocabs_v1", "unknown_token": UNKNOWN_TOKEN, "features": vocab_stats}, vocab_dir / "manifest.json")
    write_json({"version": FEATURE_CATALOG_VERSION, "features": feature_catalog}, args.output_dir / "feature_catalog.json")

    examples_summary = split_counts(examples)
    manifest = {
        "gold_dataset_version": args.version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_silver_path": str(args.silver_dir),
        "output_path": str(args.output_dir),
        "code_commit": git_commit(),
        "temporal_split": split_meta,
        "source_event_summary": split_summary(examples.select("split", "user_id", "video_id", "event_ts")),
        "example_summary": examples_summary,
        "feature_catalog_version": FEATURE_CATALOG_VERSION,
        "target_definition": {
            "version": TARGET_DEFINITION_VERSION,
            "strong_positive": "long_view/like/comment/forward/follow on current observed item",
            "observed_negative": f"not strong positive and (is_hate=1 or clicked watch_ratio <= {args.weak_watch_ratio_threshold})",
            "ambiguous": "observed event without strong positive or reliable observed negative evidence",
            "unknown_policy": "unobserved user-item pairs are not materialized and are never assumed negative",
        },
        "history_definition": {
            "version": HISTORY_DEFINITION_VERSION,
            "session_gap_minutes": 30,
            "history_reference": "Each example stores history_end_user_event_index and history_end_session_event_index; history/events stores chronological events. The current event is excluded by construction.",
        },
        "feature_groups": {
            "user_tower": user_cols,
            "item_tower": item_cols,
            "target_only": target_cols,
            "reserved_for_ranker": ["tag", "user_item_similarity", "future ALS/user-item cross scores"],
            "dropped": ["videos_statistics global counters/rates with unclear snapshot time", "current-event watch/play outcomes as features"],
        },
        "cold_start": cold_start_stats(examples),
        "vocabularies": {"version": "categorical_vocabs_v1", "fit_split": "train", "features": vocab_stats},
        "numerical_transforms": {"version": "numeric_transforms_v1", "fit_split": "train", "features": list(transform_stats)},
        "quality_checks": checks,
        "artifact_paths": {
            "train": str(args.output_dir / "train"),
            "validation": str(args.output_dir / "validation"),
            "test": str(args.output_dir / "test"),
            "user_state": str(args.output_dir / "user_state"),
            "item_features_static": str(args.output_dir / "item_features" / "static"),
            "item_features_point_in_time": str(args.output_dir / "item_features" / "point_in_time"),
            "history": str(args.output_dir / "history" / "events"),
            "targets": str(args.output_dir / "targets"),
            "vocabularies": str(args.output_dir / "vocabularies"),
            "transforms": str(args.output_dir / "transforms"),
            "feature_catalog": str(args.output_dir / "feature_catalog.json"),
            "manifest": str(args.output_dir / "manifest.json"),
        },
    }
    write_json(manifest, args.output_dir / "manifest.json")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    spark.stop()
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build point-in-time KuaiRand Gold data for a future Two-Tower retrieval model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--silver-dir", type=Path, default=Path("data/silver/kuairand"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/gold/two_tower/v1"))
    parser.add_argument("--version", default=DATASET_VERSION)
    parser.add_argument("--train-quantile", type=float, default=0.70)
    parser.add_argument("--validation-quantile", type=float, default=0.85)
    parser.add_argument("--weak-watch-ratio-threshold", type=float, default=0.20)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    write_gold_dataset(parse_args())


if __name__ == "__main__":
    main()
