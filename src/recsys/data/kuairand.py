from __future__ import annotations

from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
)


LOG_SCHEMA = StructType(
    [
        StructField("user_id", IntegerType(), True),
        StructField("video_id", IntegerType(), True),
        StructField("date", IntegerType(), True),
        StructField("hourmin", IntegerType(), True),
        StructField("time_ms", LongType(), True),
        StructField("is_click", IntegerType(), True),
        StructField("is_like", IntegerType(), True),
        StructField("is_follow", IntegerType(), True),
        StructField("is_comment", IntegerType(), True),
        StructField("is_forward", IntegerType(), True),
        StructField("is_hate", IntegerType(), True),
        StructField("long_view", IntegerType(), True),
        StructField("play_time_ms", IntegerType(), True),
        StructField("duration_ms", IntegerType(), True),
        StructField("profile_stay_time", IntegerType(), True),
        StructField("comment_stay_time", IntegerType(), True),
        StructField("is_profile_enter", IntegerType(), True),
        StructField("is_rand", IntegerType(), True),
        StructField("tab", IntegerType(), True),
    ]
)

USER_FEATURE_SCHEMA = StructType(
    [
        StructField("user_id", IntegerType(), True),
        StructField("user_active_degree", StringType(), True),
        StructField("is_lowactive_period", IntegerType(), True),
        StructField("is_live_streamer", IntegerType(), True),
        StructField("is_video_author", IntegerType(), True),
        StructField("follow_user_num", IntegerType(), True),
        StructField("follow_user_num_range", StringType(), True),
        StructField("fans_user_num", IntegerType(), True),
        StructField("fans_user_num_range", StringType(), True),
        StructField("friend_user_num", IntegerType(), True),
        StructField("friend_user_num_range", StringType(), True),
        StructField("register_days", IntegerType(), True),
        StructField("register_days_range", StringType(), True),
        *[StructField(f"onehot_feat{i}", DoubleType(), True) for i in range(18)],
    ]
)

VIDEO_BASIC_SCHEMA = StructType(
    [
        StructField("video_id", IntegerType(), True),
        StructField("author_id", LongType(), True),
        StructField("video_type", StringType(), True),
        StructField("upload_dt", StringType(), True),
        StructField("upload_type", StringType(), True),
        StructField("visible_status", DoubleType(), True),
        StructField("video_duration", DoubleType(), True),
        StructField("server_width", DoubleType(), True),
        StructField("server_height", DoubleType(), True),
        StructField("music_id", LongType(), True),
        StructField("music_type", DoubleType(), True),
        StructField("tag", StringType(), True),
    ]
)

VIDEO_STATISTIC_COLUMNS = [
    "video_id",
    "counts",
    "show_cnt",
    "show_user_num",
    "play_cnt",
    "play_user_num",
    "play_duration",
    "complete_play_cnt",
    "complete_play_user_num",
    "valid_play_cnt",
    "valid_play_user_num",
    "long_time_play_cnt",
    "long_time_play_user_num",
    "short_time_play_cnt",
    "short_time_play_user_num",
    "play_progress",
    "comment_stay_duration",
    "like_cnt",
    "like_user_num",
    "click_like_cnt",
    "double_click_cnt",
    "cancel_like_cnt",
    "cancel_like_user_num",
    "comment_cnt",
    "comment_user_num",
    "direct_comment_cnt",
    "reply_comment_cnt",
    "delete_comment_cnt",
    "delete_comment_user_num",
    "comment_like_cnt",
    "comment_like_user_num",
    "follow_cnt",
    "follow_user_num",
    "cancel_follow_cnt",
    "cancel_follow_user_num",
    "share_cnt",
    "share_user_num",
    "download_cnt",
    "download_user_num",
    "report_cnt",
    "report_user_num",
    "reduce_similar_cnt",
    "reduce_similar_user_num",
    "collect_cnt",
    "collect_user_num",
    "cancel_collect_cnt",
    "cancel_collect_user_num",
    "direct_comment_user_num",
    "reply_comment_user_num",
    "share_all_cnt",
    "share_all_user_num",
    "outsite_share_all_cnt",
]

VIDEO_STATISTIC_SCHEMA = StructType(
    [StructField("video_id", IntegerType(), True)]
    + [StructField(column, DoubleType(), True) for column in VIDEO_STATISTIC_COLUMNS[1:]]
)


def schema_for_path(path: Path) -> StructType | None:
    name = path.stem
    if name.startswith("log_"):
        return LOG_SCHEMA
    if name.startswith("user_features"):
        return USER_FEATURE_SCHEMA
    if name.startswith("video_features_basic"):
        return VIDEO_BASIC_SCHEMA
    if name.startswith("video_features_statistic"):
        return VIDEO_STATISTIC_SCHEMA
    return None


def read_csv(spark: SparkSession, path: Path, infer_schema: bool = False) -> DataFrame:
    reader = spark.read.option("header", True).option("mode", "PERMISSIVE")
    schema = schema_for_path(path)
    if schema is not None:
        reader = reader.schema(schema)
    else:
        reader = reader.option("inferSchema", infer_schema)
    return reader.csv(str(path))
