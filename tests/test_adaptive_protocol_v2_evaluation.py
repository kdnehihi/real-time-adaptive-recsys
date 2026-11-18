import numpy as np
import pandas as pd

from scripts.evaluate_adaptive_protocol_v2 import (
    build_safe_candidate_pool,
    clustered_bootstrap,
    masked_exact_retrieval,
    per_query_metrics,
    prior_seen_items,
    session_history_bucket,
    user_macro_summary,
)


def test_session_buckets_match_ten_event_cap():
    assert session_history_bucket(0) == "empty"
    assert session_history_bucket(1) == "short_1_2"
    assert session_history_bucket(2) == "short_1_2"
    assert session_history_bucket(3) == "medium_3_5"
    assert session_history_bucket(5) == "medium_3_5"
    assert session_history_bucket(6) == "long_6_10"
    assert session_history_bucket(10) == "long_6_10"


def test_seen_history_is_strictly_before_query_time():
    history = {7: (np.array([100, 200, 200, 300]), np.array([1, 2, 3, 4]))}
    assert prior_seen_items(history, 7, 200) == {1}
    assert prior_seen_items(history, 7, 201) == {1, 2, 3}
    assert prior_seen_items(history, 99, 500) == set()


def test_exact_metrics_use_only_current_target():
    protocol = pd.DataFrame(
        {
            "example_id": ["a", "b"],
            "user_id": [1, 2],
            "video_id": [10, 20],
            "target_warm": [True, False],
            "target_repeated_in_recent_history": [False, True],
            "timeline_bucket": ["early", "late"],
            "session_history_bucket_v2": ["short_1_2", "long_6_10"],
        }
    )
    ranked = np.array([[10, 99, 20], [10, 30, 20]])
    result = per_query_metrics(ranked, protocol, (1, 3))
    assert result["hitrate@1"].tolist() == [1.0, 0.0]
    assert result["hitrate@3"].tolist() == [1.0, 1.0]
    assert result["exact_rank"].tolist() == [1, 3]


def test_user_macro_prevents_active_user_from_dominating():
    frame = pd.DataFrame(
        {
            "user_id": [1, 1, 1, 2],
            "hitrate@10": [1.0, 1.0, 1.0, 0.0],
        }
    )
    summary = user_macro_summary(frame, ["hitrate@10"])
    assert summary["hitrate@10"] == 0.5
    ci = clustered_bootstrap(frame, ["hitrate@10"], samples=50, seed=3)
    assert ci["hitrate@10"]["mean"] == 0.5


def test_safe_candidate_pool_uses_train_history_and_zeroes_cold_items(tmp_path):
    columns = {
        "video_type": ["NORMAL", "NORMAL"],
        "upload_type": ["LongImport", "LongImport"],
        "video_duration_sec": [10.0, 20.0],
        "aspect_ratio": [1.0, 1.0],
        "visible_status": [1.0, 1.0],
        "music_type": [0.0, 0.0],
        "item_hist_long_view_rate": [0.8, 0.7],
        "item_hist_like_rate": [0.2, 0.1],
        "item_hist_hate_rate": [0.0, 0.0],
        "item_hist_avg_watch_ratio": [0.9, 0.8],
    }
    validation = pd.DataFrame(
        {
            "video_id": [10, 20],
            "current_time_ms": [10 * 86_400_000, 10 * 86_400_000],
            "upload_age_days_at_event": [5.0, 0.0],
            "item_hist_events": [999.0, 888.0],
            **columns,
        }
    )
    train = validation.iloc[[0]].copy()
    train["item_hist_events"] = 7.0
    (tmp_path / "sequence_examples/train").mkdir(parents=True)
    (tmp_path / "sequence_examples/validation").mkdir(parents=True)
    train.to_parquet(tmp_path / "sequence_examples/train/part-00000.parquet", index=False)
    validation.to_parquet(tmp_path / "sequence_examples/validation/part-00000.parquet", index=False)

    pool = build_safe_candidate_pool(tmp_path, "validation").set_index("video_id")
    assert pool.loc[10, "item_hist_events"] == 7.0
    assert pool.loc[20, "item_hist_events"] == 0.0
    assert bool(pool.loc[10, "target_warm"])
    assert not bool(pool.loc[20, "target_warm"])


def test_masking_excludes_future_and_seen_items_but_restores_repeat_target():
    item_ids = np.array([10, 20, 30])
    item_embeddings = np.eye(3, dtype="float32")
    users = np.array([[1.0, 0.9, 0.1]], dtype="float32")
    protocol = pd.DataFrame(
        {
            "user_id": [1],
            "video_id": [20],
            "query_time_ms": [100],
        }
    )
    # Item 10 is seen, item 30 is not available yet, and repeated target 20 is restored.
    history = {1: (np.array([50, 60]), np.array([10, 20]))}
    ranked, diagnostics = masked_exact_retrieval(
        users,
        users,
        item_embeddings,
        item_ids,
        np.array([0, 0, 200]),
        np.array([3.0, 2.0, 1.0], dtype="float32"),
        protocol,
        history,
        top_k=1,
        batch_size=1,
        blend_weight=0.02,
        device=__import__("torch").device("cpu"),
    )
    assert all(values[0, 0] == 20 for values in ranked.values())
    assert diagnostics["repeat_target_override"] == 1
