import pandas as pd
import pytest

from recommender.models.adaptive.online_state import InteractionEvent, OnlineUserStateReducer
from scripts.run_adaptive_v6_true_online_state import (
    replay_and_compare,
    sanitize_arrays_strict_prior,
    values_match,
)


def event(
    event_id: str,
    time_ms: int,
    video_id: int,
    *,
    user_id: int = 7,
    session_id: str = "s1",
    target_class: str = "STRONG_POSITIVE",
    strength: float = 1.0,
) -> InteractionEvent:
    return InteractionEvent(
        event_id=event_id,
        user_id=user_id,
        video_id=video_id,
        time_ms=time_ms,
        session_id=session_id,
        target_class=target_class,
        engagement_strength=strength,
        long_view=1,
        is_like=0,
        is_hate=0,
        is_click=1,
    )


def test_query_reads_state_before_same_timestamp_batch_is_appended():
    reducer = OnlineUserStateReducer(max_sequence_len=3, max_session_sequence_len=2)
    reducer.apply_timestamp_batch([event("a", 1000, 10)])

    before = reducer.snapshot(7, "s1", 2000)
    assert before["recent_video_ids"] == [10]

    reducer.apply_timestamp_batch([event("c", 2000, 30), event("b", 2000, 20)])
    after = reducer.snapshot(7, "s1", 3000)
    assert after["recent_video_ids"] == [10, 20, 30]
    assert after["session_recent_video_ids"] == [20, 30]
    assert after["recent_time_ms"] == [1000, 2000, 2000]


def test_reducer_rejects_non_timestamp_batches_and_backward_time():
    reducer = OnlineUserStateReducer()
    with pytest.raises(ValueError, match="one timestamp"):
        reducer.apply_timestamp_batch([event("a", 1000, 10), event("b", 2000, 20)])
    reducer.apply_timestamp_batch([event("b", 2000, 20)])
    with pytest.raises(ValueError, match="backwards"):
        reducer.apply_timestamp_batch([event("a", 1000, 10)])


def test_snapshot_matches_gold_style_arrays_rates_and_empty_state():
    reducer = OnlineUserStateReducer()
    empty = reducer.snapshot(7, "s1", 1000)
    assert empty["recent_event_count"] == 0
    assert empty["recent_long_view_rate"] == 0.0
    assert empty["seconds_since_last_event"] is None

    reducer.apply_timestamp_batch(
        [
            event("a", 1000, 10, strength=0.5),
            event("b", 1000, 20, strength=1.5, session_id="s2"),
        ]
    )
    state = reducer.snapshot(7, "s1", 2500)
    assert state["recent_video_ids"] == [10, 20]
    assert state["session_recent_video_ids"] == [10]
    assert state["recent_avg_engagement_strength"] == 1.0
    assert state["seconds_since_last_event"] == 1.5


def test_value_comparison_handles_arrays_and_float_tolerance():
    assert values_match([1.0, 2.0], [1.0 + 1e-8, 2.0])
    assert not values_match([1, 2], [2, 1])
    assert values_match(float("nan"), None)
    assert not values_match(1.0, None)


def test_strict_prior_sanitation_keeps_aligned_array_positions():
    frame = pd.DataFrame(
        {
            "current_time_ms": [2000],
            "recent_time_ms": [[1000, 2000]],
            "recent_video_ids": [[10, 20]],
            "recent_target_classes": [["A", "B"]],
            "recent_engagement_strengths": [[0.5, 1.5]],
            "recent_long_views": [[0, 1]],
            "recent_likes": [[0, 1]],
            "recent_hates": [[0, 0]],
            "recent_clicks": [[1, 1]],
        }
    )
    clean = sanitize_arrays_strict_prior(frame, "recent")
    assert clean.loc[0, "recent_time_ms"] == [1000]
    assert clean.loc[0, "recent_video_ids"] == [10]
    assert clean.loc[0, "recent_target_classes"] == ["A"]
    assert clean.loc[0, "recent_event_count"] == 1


def test_sqlite_replay_parity_uses_query_before_current_event(tmp_path):
    import sqlite3

    db = sqlite3.connect(tmp_path / "events.sqlite")
    db.execute(
        """CREATE TABLE events (
        event_key TEXT, order_id TEXT, user_id INTEGER, video_id INTEGER,
        time_ms INTEGER, session_id TEXT, target_class TEXT,
        engagement_strength REAL, long_view INTEGER, is_like INTEGER,
        is_hate INTEGER, is_click INTEGER)"""
    )
    db.executemany(
        "INSERT INTO events VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        [
            ("e1", "a", 7, 10, 1000, "s1", "STRONG_POSITIVE", 1.0, 1, 0, 0, 1),
            ("e2", "b", 7, 20, 2000, "s1", "STRONG_POSITIVE", 1.0, 1, 0, 0, 1),
        ],
    )
    query = pd.DataFrame(
        {
            "example_id": ["e2"],
            "user_id": [7],
            "session_id": ["s1"],
            "current_time_ms": [2000],
            "recent_video_ids": [[10]],
            "recent_target_classes": [["STRONG_POSITIVE"]],
            "recent_engagement_strengths": [[1.0]],
            "recent_long_views": [[1]],
            "recent_likes": [[0]],
            "recent_hates": [[0]],
            "recent_clicks": [[1]],
            "recent_time_ms": [[1000]],
            "session_recent_video_ids": [[10]],
            "session_recent_target_classes": [["STRONG_POSITIVE"]],
            "session_recent_engagement_strengths": [[1.0]],
            "session_recent_long_views": [[1]],
            "session_recent_likes": [[0]],
            "session_recent_hates": [[0]],
            "session_recent_clicks": [[1]],
            "session_recent_time_ms": [[1000]],
            "recent_event_count": [1],
            "session_recent_event_count": [1],
            "seconds_since_last_event": [1.0],
            "session_seconds_since_last_event": [1.0],
            "recent_long_view_rate": [1.0],
            "recent_like_rate": [0.0],
            "recent_hate_rate": [0.0],
            "recent_avg_engagement_strength": [1.0],
        }
    )
    report, rows, mismatches = replay_and_compare(db, query, 30, 10)
    db.close()
    assert report["accepted"] is True
    assert report["all_fields_match_rate"] == 1.0
    assert rows["core_fields_match"].all()
    assert mismatches == []
