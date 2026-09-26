from __future__ import annotations

from collections.abc import Iterable

import torch

from recommender.models.two_tower.data import EncodedTwoTowerBatch
from recommender.models.two_tower.model import TwoTowerRetrievalModel


def in_batch_retrieval_metrics(logits: torch.Tensor, top_ks: Iterable[int]) -> dict[str, float]:
    batch_size = logits.shape[0]
    if batch_size == 0:
        return {}
    labels = torch.arange(batch_size, device=logits.device)
    ranks = (torch.argsort(logits, dim=1, descending=True) == labels[:, None]).nonzero()[:, 1] + 1
    metrics: dict[str, float] = {"mrr": torch.mean(1.0 / ranks.float()).item()}
    for raw_k in top_ks:
        k = min(int(raw_k), batch_size)
        hits = ranks <= k
        metrics[f"hitrate@{raw_k}"] = hits.float().mean().item()
        metrics[f"recall@{raw_k}"] = hits.float().mean().item()
        ndcg = torch.where(hits, 1.0 / torch.log2(ranks.float() + 1.0), torch.zeros_like(ranks, dtype=torch.float32))
        metrics[f"ndcg@{raw_k}"] = ndcg.mean().item()
    return metrics


@torch.no_grad()
def evaluate_in_batch(
    model: TwoTowerRetrievalModel,
    dataloader,
    device: torch.device,
    top_ks: tuple[int, ...],
    max_batches: int | None = None,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    examples = 0
    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        assert isinstance(batch, EncodedTwoTowerBatch)
        batch = batch.to(device)
        output = model(batch.user_categorical, batch.user_numeric, batch.item_categorical, batch.item_numeric)
        batch_metrics = in_batch_retrieval_metrics(output.logits, top_ks)
        batch_size = output.logits.shape[0]
        examples += batch_size
        for key, value in batch_metrics.items():
            totals[key] = totals.get(key, 0.0) + value * batch_size
    if examples == 0:
        return {"examples": 0}
    return {key: value / examples for key, value in totals.items()} | {"examples": examples}
