from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from recommender.models.adaptive.checkpoint import as_sequence


@dataclass(frozen=True)
class AuthorVocabulary:
    """Train-only author mapping with index zero reserved for OOV."""

    values: np.ndarray

    def __post_init__(self) -> None:
        clean = np.asarray(self.values, dtype=np.int64)
        object.__setattr__(self, "values", np.unique(clean))
        object.__setattr__(self, "index", {int(value): i + 1 for i, value in enumerate(self.values)})

    @property
    def size(self) -> int:
        return len(self.values) + 1

    def encode(self, values: list[Any]) -> np.ndarray:
        return np.asarray([self.index.get(int(value), 0) if value is not None else 0 for value in values], dtype=np.int64)


class AuthorResidualRetrieval(nn.Module):
    """ANN-compatible author branch added to a frozen V4 retrieval model."""

    def __init__(self, author_count: int, embedding_dim: int = 32, dropout: float = 0.1) -> None:
        super().__init__()
        self.author_embedding = nn.Embedding(author_count, embedding_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.log_scale = nn.Parameter(torch.tensor(-1.5))
        self.gate_bias = nn.Parameter(torch.tensor(-0.5))
        self.gate_count_weight = nn.Parameter(torch.tensor(0.5))

    def encode_user(self, recent_author_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        valid = recent_author_ids.ne(0)
        count = valid.sum(dim=1)
        emb = self.dropout(self.author_embedding(recent_author_ids))
        positions = torch.arange(recent_author_ids.shape[1], device=recent_author_ids.device, dtype=emb.dtype)
        recency = torch.exp((positions - positions.max()) / 6.0)[None, :]
        weights = recency * valid.to(emb.dtype)
        pooled = (emb * weights.unsqueeze(-1)).sum(1) / weights.sum(1, keepdim=True).clamp_min(1.0)
        pooled = F.normalize(pooled, dim=1)
        pooled = pooled * count.gt(0).to(pooled.dtype).unsqueeze(1)
        gate = torch.sigmoid(self.gate_bias + self.gate_count_weight * torch.log1p(count.to(emb.dtype)))
        gate = gate * count.gt(0).to(gate.dtype)
        return pooled, gate

    def encode_item(self, author_ids: torch.Tensor) -> torch.Tensor:
        emb = F.normalize(self.author_embedding(author_ids), dim=1)
        return emb * author_ids.ne(0).to(emb.dtype).unsqueeze(1)

    @property
    def scale(self) -> torch.Tensor:
        return F.softplus(self.log_scale)

    def score(self, recent_author_ids: torch.Tensor, item_author_ids: torch.Tensor) -> torch.Tensor:
        user, gate = self.encode_user(recent_author_ids)
        item = self.encode_item(item_author_ids)
        return self.scale * gate[:, None] * (user @ item.T)

    def concatenate_embeddings(
        self,
        v4_user: torch.Tensor,
        v4_item: torch.Tensor,
        recent_author_ids: torch.Tensor,
        item_author_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        author_user, gate = self.encode_user(recent_author_ids)
        author_item = self.encode_item(item_author_ids)
        root = torch.sqrt(self.scale.clamp_min(1e-8))
        return (
            torch.cat([v4_user, root * gate[:, None] * author_user], dim=1),
            torch.cat([v4_item, root * author_item], dim=1),
        )


def encode_author_inputs(
    frame: pd.DataFrame,
    video_to_author: dict[int, int],
    vocabulary: AuthorVocabulary,
    max_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map target and strict-prior video sequences to train-only author IDs."""
    sequence_rows: list[np.ndarray] = []
    for values in frame["recent_video_ids"]:
        videos = as_sequence(values)[-max_len:]
        authors = [video_to_author.get(int(video)) for video in videos]
        encoded = vocabulary.encode(authors)
        padded = np.zeros(max_len, dtype=np.int64)
        if len(encoded):
            padded[-len(encoded) :] = encoded
        sequence_rows.append(padded)
    target_authors = [video_to_author.get(int(video)) for video in frame["video_id"]]
    return (
        torch.as_tensor(np.stack(sequence_rows), dtype=torch.long),
        torch.as_tensor(vocabulary.encode(target_authors), dtype=torch.long),
    )
