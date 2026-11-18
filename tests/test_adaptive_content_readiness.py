import numpy as np
import pandas as pd

from scripts.audit_adaptive_content_readiness import (
    STATIC_COLUMNS,
    add_content_signatures,
    build_feature_quality,
    prior_author_features,
)


def test_content_signature_changes_with_author_but_metadata_signature_does_not():
    frame = pd.DataFrame(
        {
            "video_id": [1, 2],
            "author_id": [10, 20],
            "video_type": ["NORMAL", "NORMAL"],
            "upload_type": ["LongImport", "LongImport"],
            "video_duration_sec": [12.0, 12.0],
            "aspect_ratio": [1.0, 1.0],
            "visible_status": [1, 1],
            "music_type": [0, 0],
        }
    )
    result = add_content_signatures(frame)
    assert result["metadata_signature"].nunique() == 1
    assert result["author_content_signature"].nunique() == 2


def test_prior_author_features_are_strictly_point_in_time():
    protocol = pd.DataFrame(
        {
            "example_id": ["q1", "q2"],
            "user_id": [7, 7],
            "video_id": [20, 30],
            "query_time_ms": [200, 301],
        }
    )
    history = {7: (np.array([100, 200, 300]), np.array([10, 20, 30]))}
    video_to_author = {10: 1, 20: 1, 30: 2}
    result = prior_author_features(protocol, history, video_to_author).set_index("example_id")
    assert bool(result.loc["q1", "target_author_seen_before"])
    assert result.loc["q1", "target_author_events_before"] == 1
    assert bool(result.loc["q2", "target_author_seen_before"])
    assert result.loc["q2", "target_author_events_before"] == 1


def test_feature_quality_uses_renamed_target_video_id():
    candidates = pd.DataFrame({column: [1, 2] for column in STATIC_COLUMNS})
    audit = candidates.rename(columns={"video_id": "target_video_id"}).iloc[[0]].copy()
    quality = build_feature_quality(candidates, audit).set_index("feature")
    assert quality.loc["video_id", "candidate_cardinality"] == 2
    assert quality.loc["video_id", "target_missing_rate"] == 0.0
