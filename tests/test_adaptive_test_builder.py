from datetime import datetime, timezone

from scripts.build_adaptive_test_artifacts import enforce_strict_prior


def test_test_builder_filters_same_timestamp_and_keeps_arrays_aligned(spark):
    frame = spark.createDataFrame(
        [
            (
                datetime.fromtimestamp(2, timezone.utc),
                [1000, 2000, 3000],
                [10, 20, 30],
                ["A", "B", "C"],
                [0.1, 0.2, 0.3],
                [0, 1, 1],
                [0, 0, 1],
                [0, 0, 0],
                [1, 1, 1],
                [1000, 2000],
                [10, 20],
                ["A", "B"],
                [0.1, 0.2],
                [0, 1],
                [0, 0],
                [0, 0],
                [1, 1],
            )
        ],
        """as_of_time timestamp,
           recent_time_ms array<long>, recent_video_ids array<int>,
           recent_target_classes array<string>, recent_engagement_strengths array<double>,
           recent_long_views array<int>, recent_likes array<int>, recent_hates array<int>,
           recent_clicks array<int>, session_recent_time_ms array<long>,
           session_recent_video_ids array<int>, session_recent_target_classes array<string>,
           session_recent_engagement_strengths array<double>, session_recent_long_views array<int>,
           session_recent_likes array<int>, session_recent_hates array<int>,
           session_recent_clicks array<int>""",
    )
    row = enforce_strict_prior(frame).first().asDict()
    assert row["recent_time_ms"] == [1000]
    assert row["recent_video_ids"] == [10]
    assert row["session_recent_time_ms"] == [1000]
    assert row["session_recent_video_ids"] == [10]
    assert row["recent_event_count"] == 1
    assert row["session_recent_event_count"] == 1
    assert row["current_time_ms"] == 2000
