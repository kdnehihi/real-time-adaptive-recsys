from argparse import Namespace

import pandas as pd
from pyspark.sql import functions as F

from recommender.data.adaptive_sequences import build_strict_prior_sequence_context
from scripts.audit_corrected_adaptive_sequences import (
    compare_contexts,
    corrected_batch_queries,
    corrected_replay_queries,
    install_worker_runtime,
    legacy_delta,
    read_protocol_index,
)
from scripts.build_corrected_adaptive_sequence_delta import (
    build_delta,
    example_id_expression,
    materialize_contexts,
    quality_report,
)


def make_event(index: int, time_ms: int, video_id: int, session_id: str = "7_1") -> dict:
    """Create one compact history event for strict-prior Spark tests."""
    return {
        "event_id": f"event-{index:04d}",
        "user_id": 7,
        "video_id": video_id,
        "time_ms": time_ms,
        "session_id": session_id,
        "target_class": "STRONG_POSITIVE",
        "engagement_strength": 1.0,
        "long_view": 1,
        "is_like": 0,
        "is_hate": 0,
        "is_click": 1,
    }


def test_timestamp_batch_filter_happens_before_sequence_limit(spark):
    older = [make_event(index, index * 1000, index) for index in range(1, 36)]
    current_batch = [
        make_event(100, 100_000, 100),
        make_event(101, 100_000, 101),
        make_event(102, 100_000, 102),
    ]
    following = [make_event(103, 101_000, 103)]
    history = spark.createDataFrame(older + current_batch + following)

    global_context, session_context = build_strict_prior_sequence_context(history, 30, 10)
    at_current = global_context.where("time_ms = 100000").first().asDict()
    at_following = global_context.where("time_ms = 101000").first().asDict()
    session_current = session_context.where("time_ms = 100000").first().asDict()

    # Current-time events are excluded before limiting, so older events backfill to 30.
    assert at_current["recent_video_ids"] == list(range(6, 36))
    assert at_current["recent_event_count"] == 30
    assert max(at_current["recent_time_ms"]) < 100_000

    # The next timestamp sees the entire prior timestamp batch in event_id order.
    assert at_following["recent_video_ids"][-3:] == [100, 101, 102]
    assert at_following["recent_event_count"] == 30

    # Session state follows the same rule with its independent cap.
    assert session_current["session_recent_video_ids"] == list(range(26, 36))
    assert session_current["session_recent_event_count"] == 10


def test_same_timestamp_events_have_one_shared_prior_context(spark):
    rows = [
        make_event(1, 1000, 10),
        make_event(2, 2000, 20),
        make_event(3, 2000, 30),
    ]
    history = spark.createDataFrame(rows)
    global_context, session_context = build_strict_prior_sequence_context(history, 30, 10)

    global_row = global_context.where("time_ms = 2000").first().asDict()
    session_row = session_context.where("time_ms = 2000").first().asDict()
    assert global_context.where("time_ms = 2000").count() == 1
    assert session_context.where("time_ms = 2000").count() == 1
    assert global_row["recent_video_ids"] == [10]
    assert session_row["session_recent_video_ids"] == [10]


def test_batch_builder_matches_independent_event_reducer(spark):
    older = [make_event(index, index * 1000, index) for index in range(1, 36)]
    current = [make_event(100, 100_000, 100), make_event(101, 100_000, 101)]
    history = spark.createDataFrame(older + current)
    queries = spark.createDataFrame(
        [
            {
                "example_id": "query-100",
                "user_id": 7,
                "video_id": 100,
                "session_id": "7_1",
                "current_time_ms": 100_000,
            }
        ]
    )

    batch = corrected_batch_queries(history, queries, 30, 10)
    replay = corrected_replay_queries(history, queries, 30, 10)
    compared, report = compare_contexts(batch, replay)

    assert compared.count() == 1
    assert report["queries_compared"] == 1
    assert report["accepted"] is True

    legacy = batch.select(
        "example_id", F.slice("recent_video_ids", -26, 26).alias("recent_video_ids")
    )
    delta = legacy_delta(legacy, batch)
    assert delta["recent_count_match_rate"] == 0.0
    assert delta["legacy_is_corrected_suffix_rate"] == 1.0


def test_pandas_protocol_loader_hides_nanosecond_timestamp_from_spark(spark, tmp_path):
    path = tmp_path / "queries.parquet"
    pd.DataFrame(
        {
            "example_id": ["example-1"],
            "user_id": [7],
            "video_id": [99],
            "query_time_ms": [1_650_000_000_123],
            "as_of_time": [pd.Timestamp("2022-04-15 12:00:00.123456789")],
        }
    ).to_parquet(path, index=False)

    loaded = read_protocol_index(spark, path).first().asDict()
    assert loaded == {
        "example_id": "example-1",
        "user_id": 7,
        "video_id": 99,
        "query_time_ms": 1_650_000_000_123,
    }


def test_worker_runtime_zip_is_added_to_python_workers(spark, tmp_path):
    runtime = install_worker_runtime(spark, tmp_path / "worker-runtime.zip")

    def inspect_worker(_: int) -> tuple[bool, str]:
        import sys
        from recommender.models.adaptive.online_state import OnlineUserStateReducer

        return any(runtime.name in entry for entry in sys.path), OnlineUserStateReducer.__name__

    visible, class_name = spark.sparkContext.parallelize([1], 1).map(inspect_worker).first()
    assert visible is True
    assert class_name == "OnlineUserStateReducer"


def test_corrected_delta_has_full_key_coverage_and_strict_prior_arrays(spark, tmp_path):
    history = spark.createDataFrame(
        [
            make_event(1, 1000, 10),
            make_event(2, 2000, 20),
            make_event(3, 2000, 30),
            make_event(4, 3000, 40),
        ]
    )
    keyed = history.withColumn("example_id", example_id_expression())
    keys = keyed.select("example_id", "user_id").withColumn("split", F.lit("train"))
    args = Namespace(
        work_dir=tmp_path / "work",
        overwrite=True,
        rebuild_contexts=False,
        max_sequence_len=30,
        max_session_sequence_len=10,
    )
    global_context, session_context = materialize_contexts(args, history)
    delta = build_delta(keys, history, global_context, session_context)
    report = quality_report(delta, keys, args)

    assert report["passed"] is True
    assert report["rows"] == 4
    at_2000 = delta.where("current_time_ms = 2000").select("recent_video_ids").first()[0]
    assert at_2000 == [10]
