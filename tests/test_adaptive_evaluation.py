import json

import numpy as np
import pandas as pd
import torch

from recommender.models.adaptive.checkpoint import (
    ITEM_NUMERIC,
    USER_NUMERIC,
    NumericalPreprocessor,
    V5ResidualRanker,
    V4TransformerRetrieval,
    V5Ranker,
    Vocabulary,
    encode_frame,
    multi_positive_metrics,
)


def _frame() -> pd.DataFrame:
    rows = []
    for index in range(3):
        row = {name: 0.0 for name in (*USER_NUMERIC, *ITEM_NUMERIC)}
        row.update(
            {
                "user_id": index + 1,
                "video_id": 10 + index,
                "user_active_degree": "A",
                "video_type": "NORMAL",
                "upload_type": "USER",
                "recent_video_ids": [10, 11],
                "recent_target_classes": ["AMBIGUOUS_WEAK", "STRONG_POSITIVE"],
                "recent_engagement_strengths": [0.1, 0.8],
                "recent_long_views": [0, 1],
                "recent_likes": [0, 1],
                "recent_hates": [0, 0],
                "recent_clicks": [1, 1],
                "recent_time_ms": [1000, 2000],
                "current_time_ms": 3000,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def test_v4_v5_forward_shapes(tmp_path):
    stats_path = tmp_path / "numeric_stats.json"
    stats_path.write_text(json.dumps({"features": {}}))
    numeric = NumericalPreprocessor(stats_path)
    vocabs = {
        "user_id": Vocabulary("user_id", [1, 2, 3]),
        "video_id": Vocabulary("video_id", [10, 11, 12]),
        "user_active_degree": Vocabulary("user_active_degree", ["A"]),
        "video_type": Vocabulary("video_type", ["NORMAL"]),
        "upload_type": Vocabulary("upload_type", ["USER"]),
        "event_type": Vocabulary(
            "event_type",
            ["AMBIGUOUS_WEAK", "OBSERVED_NEGATIVE", "STRONG_POSITIVE"],
        ),
    }
    config = {
        "retrieval_dim": 8,
        "transformer_dim": 16,
        "transformer_heads": 4,
        "transformer_layers": 1,
        "transformer_dropout": 0.0,
        "user_hidden_dims": [16],
        "item_hidden_dims": [16],
        "dropout": 0.0,
    }
    batch = encode_frame(_frame(), vocabs, numeric, max_len=4)
    v4 = V4TransformerRetrieval({name: vocab.size for name, vocab in vocabs.items()}, config).eval()
    with torch.no_grad():
        user_embeddings = v4.encode_user(batch)
        item_embeddings = v4.encode_item(batch)
        retrieval_scores = (user_embeddings * item_embeddings).sum(dim=1)
        ranker_scores = V5Ranker(8, [16, 8], 0.0).eval()(user_embeddings, item_embeddings, retrieval_scores)

    assert user_embeddings.shape == (3, 8)
    assert item_embeddings.shape == (3, 8)
    assert ranker_scores.shape == (3,)
    assert torch.isfinite(ranker_scores).all()


def test_multi_positive_metrics_distinguish_queries_from_users():
    candidate_ids = np.array([10, 11, 12], dtype=np.int64)
    ranked_idx = np.array([[0, 1, 2], [1, 0, 2], [2, 1, 0]], dtype=np.int64)
    query_user_ids = np.array([1, 1, 2], dtype=np.int64)
    relevance = pd.DataFrame({"user_id": [1, 1, 2], "video_id": [10, 11, 12]})

    metrics = multi_positive_metrics(candidate_ids, ranked_idx, query_user_ids, relevance, (1, 3))

    assert metrics["evaluated_queries"] == 3
    assert metrics["evaluated_users"] == 2
    assert metrics["hitrate@3"] == 1.0


def test_residual_ranker_initially_preserves_v4_order():
    ranker = V5ResidualRanker(8, [16, 8], 0.0).eval()
    user = torch.randn(5, 8)
    item = torch.randn(5, 8)
    v4_scores = torch.tensor([-1.2, -0.4, 0.0, 0.7, 1.6])

    with torch.no_grad():
        ranker_scores = ranker(user, item, v4_scores)

    assert torch.equal(torch.argsort(ranker_scores), torch.argsort(v4_scores))
