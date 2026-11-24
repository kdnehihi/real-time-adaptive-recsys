from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from recommender.models.adaptive.author_branch import author_affinity_scores


@dataclass(frozen=True)
class RetrievalResult:
    """Top-K item IDs, policy scores, and valid-result counts for one batch."""

    item_ids: np.ndarray
    scores: np.ndarray
    valid_counts: np.ndarray


class ExactAdaptiveIndex:
    """Immutable GPU exact index for the fixed Corrected V4.2 retrieval policy."""

    def __init__(
        self,
        item_ids: np.ndarray,
        item_embeddings: np.ndarray,
        author_ids: np.ndarray,
        popularity: np.ndarray,
        eligible_from_ms: np.ndarray,
        device: torch.device,
    ) -> None:
        size = len(item_ids)
        if not all(len(values) == size for values in (item_embeddings, author_ids, popularity, eligible_from_ms)):
            raise ValueError("All serving arrays must have the same item count")
        if len(np.unique(item_ids)) != size:
            raise ValueError("Serving item IDs must be unique")
        self.item_ids_np = np.asarray(item_ids, dtype=np.int64)
        self.item_to_index = {int(value): index for index, value in enumerate(self.item_ids_np)}
        self.item_embeddings = torch.as_tensor(item_embeddings, dtype=torch.float32, device=device)
        self.author_ids = torch.as_tensor(author_ids, dtype=torch.long, device=device)
        self.popularity = torch.as_tensor(popularity, dtype=torch.float32, device=device)
        self.eligible_from_ms = torch.as_tensor(eligible_from_ms, dtype=torch.long, device=device)
        self.device = device

    @torch.inference_mode()
    def retrieve(
        self,
        user_embeddings: np.ndarray | torch.Tensor,
        recent_author_ids: np.ndarray | torch.Tensor,
        query_time_ms: np.ndarray | torch.Tensor,
        seen_item_ids: list[set[int]] | None = None,
        top_k: int = 1000,
        author_weight: float = 0.2,
        popularity_weight: float = 0.02,
        recency_tau: float = 3.0,
    ) -> RetrievalResult:
        """Apply vector, author, popularity, time, and seen-item policy in one exact pass."""
        users = torch.as_tensor(user_embeddings, dtype=torch.float32, device=self.device)
        authors = torch.as_tensor(recent_author_ids, dtype=torch.long, device=self.device)
        clocks = torch.as_tensor(query_time_ms, dtype=torch.long, device=self.device).reshape(-1)
        if users.ndim != 2 or authors.ndim != 2 or len(users) != len(authors) or len(users) != len(clocks):
            raise ValueError("Expected aligned user [B,D], recent-author [B,L], and query-time [B] arrays")
        scores = users @ self.item_embeddings.T
        if author_weight:
            scores += float(author_weight) * author_affinity_scores(
                authors, self.author_ids, recency_tau
            )
        if popularity_weight:
            scores += float(popularity_weight) * self.popularity
        mask = self.eligible_from_ms[None, :] > clocks[:, None]
        if seen_item_ids is not None:
            if len(seen_item_ids) != len(users):
                raise ValueError("seen_item_ids must contain one set per query")
            for row, values in enumerate(seen_item_ids):
                positions = [self.item_to_index[item] for item in values if item in self.item_to_index]
                if positions:
                    mask[row, positions] = True
        scores.masked_fill_(mask, -torch.inf)
        k = min(int(top_k), scores.shape[1])
        values, positions = torch.topk(scores, k=k, dim=1)
        valid = torch.isfinite(values)
        ids = self.item_ids_np[positions.cpu().numpy()]
        ids[~valid.cpu().numpy()] = -1
        return RetrievalResult(
            item_ids=ids,
            scores=values.cpu().numpy(),
            valid_counts=valid.sum(dim=1).cpu().numpy(),
        )
