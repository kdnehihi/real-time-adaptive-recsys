import numpy as np
import pytest
import torch

from recommender.serving.exact_retrieval import ExactAdaptiveIndex


def make_index() -> ExactAdaptiveIndex:
    return ExactAdaptiveIndex(
        item_ids=np.asarray([10, 20, 30]),
        item_embeddings=np.asarray([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]], dtype="float32"),
        author_ids=np.asarray([1, 2, 1]),
        popularity=np.asarray([0.0, 0.0, 0.0], dtype="float32"),
        eligible_from_ms=np.asarray([0, 0, 200]),
        device=torch.device("cpu"),
    )


def test_exact_index_applies_time_and_seen_masks() -> None:
    result = make_index().retrieve(
        [[1.0, 0.0]], [[0, 0]], [100], seen_item_ids=[{10}], top_k=3,
        author_weight=0.0, popularity_weight=0.0,
    )
    assert result.item_ids.tolist() == [[20, -1, -1]]
    assert result.valid_counts.tolist() == [1]


def test_author_affinity_can_change_order() -> None:
    result = make_index().retrieve(
        [[1.0, 0.0]], [[0, 1]], [300], top_k=3,
        author_weight=2.0, popularity_weight=0.0,
    )
    assert result.item_ids[0, 0] == 10
    assert result.item_ids[0, 1] == 30


def test_exact_index_rejects_unaligned_batches() -> None:
    with pytest.raises(ValueError):
        make_index().retrieve([[1.0, 0.0]], [[1], [2]], [100])
