import numpy as np
import pandas as pd
import torch

from recommender.models.adaptive.author_branch import (
    AuthorResidualRetrieval,
    AuthorVocabulary,
    encode_author_inputs,
)


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
