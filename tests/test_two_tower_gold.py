from pyspark.sql import functions as F

from recommender.gold.two_tower import (
    add_point_in_time_features,
    add_targets,
    build_vocabulary,
    cold_start_stats,
    numerical_transform_stats,
    vocabulary_stats,
)


def test_point_in_time_user_and_item_history_exclude_current_event(spark):
    rows = [
        ("e1", 1, 10, "2022-01-01 00:00:00", 1000, 1, 0, 0, 0, 0, 0, 1, 1.0, 1000.0),
        ("e2", 1, 20, "2022-01-01 00:10:00", 2000, 0, 1, 0, 0, 0, 0, 1, 0.1, 100.0),
        ("e3", 2, 10, "2022-01-01 00:20:00", 3000, 0, 0, 0, 0, 0, 1, 1, 0.0, 0.0),
    ]
    df = spark.createDataFrame(
        rows,
        "event_id string, user_id int, video_id int, event_ts string, time_ms long, "
        "long_view int, is_like int, is_comment int, is_forward int, is_follow int, is_hate int, is_click int, "
        "watch_ratio double, play_time_sec double",
    ).withColumn("event_ts", F.to_timestamp("event_ts"))

    out = add_point_in_time_features(add_targets(df)).orderBy("time_ms").collect()

    assert out[0]["user_hist_events"] == 0
    assert out[0]["item_hist_events"] == 0
    assert out[1]["user_hist_events"] == 1
    assert out[1]["user_hist_long_view_rate"] == 1.0
    assert out[1]["session_event_index"] == 2
    assert out[1]["history_end_session_event_index"] == 1
    assert out[2]["item_hist_events"] == 1
    assert out[2]["item_hist_long_view_rate"] == 1.0
    assert out[2]["user_hist_events"] == 0


def test_session_gap_starts_new_session_and_current_item_not_in_history(spark):
    rows = [
        ("e1", 1, 10, "2022-01-01 00:00:00", 1000, 1, 0, 0, 0, 0, 0, 1, 1.0, 1000.0),
        ("e2", 1, 10, "2022-01-01 00:31:00", 2000, 0, 0, 0, 0, 0, 0, 1, 0.0, 0.0),
    ]
    df = spark.createDataFrame(
        rows,
        "event_id string, user_id int, video_id int, event_ts string, time_ms long, "
        "long_view int, is_like int, is_comment int, is_forward int, is_follow int, is_hate int, is_click int, "
        "watch_ratio double, play_time_sec double",
    ).withColumn("event_ts", F.to_timestamp("event_ts"))

    second = add_point_in_time_features(add_targets(df)).orderBy("time_ms").collect()[1]

    assert second["session_seq"] == 2
    assert second["session_event_index"] == 1
    assert second["history_end_session_event_index"] == 0
    assert second["user_hist_events"] == 1


def test_vocabulary_is_fit_from_supplied_train_only_frame(spark):
    train = spark.createDataFrame([("a",), ("b",), ("a",)], "category string")
    vocab = {r["category"]: r["category_idx"] for r in build_vocabulary(train, "category").collect()}

    assert vocab == {"a": 1, "b": 2}
    assert "c" not in vocab


def test_batched_metadata_stats_match_train_frame(spark):
    train = spark.createDataFrame(
        [
            ("train", "a", "x", 1.0, True, True, True, True),
            ("train", "b", "x", 3.0, True, False, True, True),
            ("validation", "c", "y", 100.0, False, True, True, True),
        ],
        "split string, cat string, other_cat string, value double, is_warm_user boolean, "
        "is_warm_item boolean, has_user_history boolean, has_item_metadata boolean",
    )
    train_only = train.where("split = 'train'")

    vocab_meta = vocabulary_stats(train_only, ["cat", "other_cat"])
    numeric_meta = numerical_transform_stats(train_only, ["value"])
    cold_meta = cold_start_stats(train)

    assert vocab_meta["cat"]["cardinality_excluding_oov"] == 2
    assert vocab_meta["other_cat"]["cardinality_excluding_oov"] == 1
    assert numeric_meta["value"]["non_null"] == 2
    assert numeric_meta["value"]["mean"] == 2.0
    assert numeric_meta["value"]["min"] == 1.0
    assert numeric_meta["value"]["max"] == 3.0
    assert cold_meta["train"]["examples"] == 2
    assert cold_meta["validation"]["examples"] == 1
