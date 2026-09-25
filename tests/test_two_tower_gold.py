from pyspark.sql import functions as F

from recommender.gold.two_tower import add_point_in_time_features, add_targets, build_vocabulary


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
