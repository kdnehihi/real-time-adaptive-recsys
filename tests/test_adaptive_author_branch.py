import numpy as np
import pandas as pd
import torch

from recommender.models.adaptive.author_branch import (
    AuthorResidualRetrieval,
    AuthorVocabulary,
    author_affinity_scores,
    encode_author_inputs,
    encode_raw_recent_authors,
)
from scripts.evaluate_adaptive_author_affinity import deterministic_user_split, score_configs


def test_author_vocab_reserves_oov_and_inputs_are_point_ordered():
    vocab = AuthorVocabulary(np.array([10, 20]))
    frame = pd.DataFrame({"video_id": [3], "recent_video_ids": [[1, 2]]})
    sequence, target = encode_author_inputs(frame, {1: 10, 2: 99, 3: 20}, vocab, 4)
    assert sequence.tolist() == [[0, 0, 1, 0]]
    assert target.tolist() == [2]


def test_author_branch_is_zero_for_empty_or_oov_history():
    model = AuthorResidualRetrieval(4, embedding_dim=8, dropout=0.0)
    user, gate = model.encode_user(torch.zeros((2, 5), dtype=torch.long))
    assert torch.equal(user, torch.zeros_like(user))
    assert torch.equal(gate, torch.zeros_like(gate))


def test_concatenation_preserves_sum_of_dot_products():
    torch.manual_seed(3)
    model = AuthorResidualRetrieval(5, embedding_dim=4, dropout=0.0).eval()
    v4_user = torch.nn.functional.normalize(torch.randn(2, 6), dim=1)
    v4_item = torch.nn.functional.normalize(torch.randn(3, 6), dim=1)
    recent = torch.tensor([[0, 1, 2], [0, 0, 3]])
    item_author = torch.tensor([1, 2, 4])
    combined_user, combined_item = model.concatenate_embeddings(v4_user, v4_item, recent, item_author)
    expected = v4_user @ v4_item.T + model.score(recent, item_author)
    assert torch.allclose(combined_user @ combined_item.T, expected, atol=1e-6)


def test_raw_author_affinity_uses_strict_recent_order_and_oov_zero():
    frame = pd.DataFrame({"recent_video_ids": [[1, 2, 3], [9]], "video_id": [4, 5]})
    recent = encode_raw_recent_authors(frame, {1: 10, 2: 20, 3: 10}, max_len=4)
    assert recent.tolist() == [[0, 10, 20, 10], [0, 0, 0, 0]]

    candidates = torch.tensor([0, 10, 20, 30])
    scores = author_affinity_scores(recent, candidates, recency_tau=1.0)
    assert scores.shape == (2, 4)
    assert scores[0, 1].item() == 1.0
    assert 0.0 < scores[0, 2].item() < scores[0, 1].item()
    assert torch.equal(scores[:, 0], torch.zeros(2))
    assert torch.equal(scores[1], torch.zeros(4))


def test_author_affinity_improves_exact_rank_without_user_split_leakage():
    protocol = pd.DataFrame(
        {"example_id": [1], "user_id": [7], "video_id": [20], "query_time_ms": [100]}
    )
    outputs = score_configs(
        indices=np.array([0]),
        configs=((0.0, 0.0), (0.3, 0.0)),
        protocol=protocol,
        user_embeddings=np.array([[1.0, 0.0]], dtype="float32"),
        item_embeddings=np.array([[1.0, 0.0], [0.8, 0.0], [0.7, 0.0]], dtype="float32"),
        recent_authors=torch.tensor([[0, 20]]),
        candidate_authors=torch.tensor([10, 20, 30]),
        popularity=np.zeros(3, dtype="float32"),
        item_ids=np.array([10, 20, 30]),
        eligible_from_ms=np.zeros(3, dtype="int64"),
        history={},
        batch_size=1,
        recency_tau=1.0,
        device=torch.device("cpu"),
    )
    assert outputs[(0.0, 0.0)][0].tolist() == [2]
    assert outputs[(0.3, 0.0)][0].tolist() == [1]

    users = pd.Series([1, 1, 2, 2, 3])
    split = deterministic_user_split(users, 0.5, seed=44)
    assert split[0] == split[1]
    assert split[2] == split[3]
