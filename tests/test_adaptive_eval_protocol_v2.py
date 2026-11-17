import pandas as pd

from scripts.build_adaptive_eval_protocol_v2 import (
    build_index,
    history_bucket,
    select_user_queries,
)


def test_history_buckets_cover_early_and_long_states():
    assert history_bucket(0) == "empty"
    assert history_bucket(1) == "short_1_4"
    assert history_bucket(4) == "short_1_4"
    assert history_bucket(5) == "medium_5_19"
    assert history_bucket(19) == "medium_5_19"
    assert history_bucket(20) == "long_20_plus"


def test_query_selection_is_deterministic_and_stratified():
    frame = pd.DataFrame(
        {
            "example_id": [f"e{index}" for index in range(12)],
            "user_id": [1] * 12,
            "video_id": list(range(100, 112)),
            "as_of_time": pd.date_range("2022-01-01", periods=12, freq="h"),
            "current_time_ms": list(range(12)),
            "recent_event_count": [0, 1, 2, 4, 5, 8, 12, 19, 20, 25, 30, 40],
        }
    )
    indexed = build_index(frame)
    first = select_user_queries(indexed, maximum=8)
    second = select_user_queries(indexed, maximum=8)
    selected = indexed[indexed["example_id"].isin(first)]

    assert first == second
    assert len(first) == 8
    assert set(selected["history_bucket"]) == {
        "empty",
        "short_1_4",
        "medium_5_19",
        "long_20_plus",
    }
    assert set(selected["timeline_bucket"]) == {"early", "middle", "late"}
