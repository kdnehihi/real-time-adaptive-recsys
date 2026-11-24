from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from recommender.models.adaptive.checkpoint import USER_NUMERIC
from recommender.serving.exact_retrieval import ExactAdaptiveIndex, RetrievalResult
from recommender.serving.preprocessing import ServingUserEncoder


@dataclass(frozen=True)
class RecommendationRequest:
    """Point-in-time request consumed by the standalone adaptive runtime."""

    user_id: int
    current_time_ms: int
    recent_events: tuple[dict[str, Any], ...] = ()
    seen_item_ids: frozenset[int] = frozenset()
    user_active_degree: Any = None
    user_features: dict[str, float] | None = None
    top_k: int = 10


class AdaptiveServingRuntime:
    """Compose standalone user encoding with the immutable exact item index."""

    def __init__(self, encoder: ServingUserEncoder, index: ExactAdaptiveIndex) -> None:
        self.encoder = encoder
        self.index = index
        self.video_to_author = {
            int(video): int(author)
            for video, author in zip(index.item_ids_np, index.author_ids_np)
        }

    @classmethod
    def from_artifacts(
        cls,
        catalog_dir: Path,
        preprocessing_dir: Path,
        checkpoint_path: Path,
        device: torch.device,
    ) -> "AdaptiveServingRuntime":
        """Load all serving artifacts without accessing Gold datasets."""
        encoder = ServingUserEncoder(preprocessing_dir, checkpoint_path, device)
        index = ExactAdaptiveIndex(
            np.load(catalog_dir / "item_ids.npy", allow_pickle=False),
            np.load(catalog_dir / "item_embeddings_fp16.npy", allow_pickle=False),
            np.load(catalog_dir / "author_ids.npy", allow_pickle=False),
            np.load(catalog_dir / "popularity.npy", allow_pickle=False),
            np.load(catalog_dir / "eligible_from_ms.npy", allow_pickle=False),
            device,
        )
        return cls(encoder, index)

    def _frame(self, requests: list[RecommendationRequest]) -> tuple[pd.DataFrame, np.ndarray]:
        """Validate strict-prior events and materialize the model's sequence contract."""
        rows: list[dict[str, Any]] = []
        recent_authors: list[np.ndarray] = []
        max_len = int(self.encoder.config["sequence_max_len"])
        for request in requests:
            events = sorted(request.recent_events, key=lambda event: int(event["time_ms"]))[-max_len:]
            if any(int(event["time_ms"]) >= int(request.current_time_ms) for event in events):
                raise ValueError("Every recent event must be strictly earlier than current_time_ms")
            row: dict[str, Any] = {
                "user_id": int(request.user_id),
                "user_active_degree": request.user_active_degree,
                "current_time_ms": int(request.current_time_ms),
                "recent_video_ids": [int(event["video_id"]) for event in events],
                "recent_target_classes": [event.get("target_class", "AMBIGUOUS_WEAK") for event in events],
                "recent_engagement_strengths": [float(event.get("engagement_strength", 0.0)) for event in events],
                "recent_long_views": [float(event.get("long_view", 0.0)) for event in events],
                "recent_likes": [float(event.get("is_like", 0.0)) for event in events],
                "recent_hates": [float(event.get("is_hate", 0.0)) for event in events],
                "recent_clicks": [float(event.get("is_click", 0.0)) for event in events],
                "recent_time_ms": [int(event["time_ms"]) for event in events],
            }
            features = request.user_features or {}
            row.update({feature: float(features.get(feature, 0.0)) for feature in USER_NUMERIC})
            rows.append(row)
            authors = np.zeros(max_len, dtype=np.int64)
            values = [self.video_to_author.get(int(event["video_id"]), 0) for event in events]
            if values:
                authors[-len(values) :] = values
            recent_authors.append(authors)
        return pd.DataFrame(rows), np.stack(recent_authors)

    def recommend(self, requests: list[RecommendationRequest]) -> RetrievalResult:
        """Encode request state and retrieve fixed-policy recommendations."""
        if not requests:
            raise ValueError("At least one request is required")
        frame, authors = self._frame(requests)
        user_embeddings = self.encoder.encode(frame)
        top_k = max(request.top_k for request in requests)
        result = self.index.retrieve(
            user_embeddings,
            authors,
            frame["current_time_ms"].to_numpy("int64"),
            [set(request.seen_item_ids) for request in requests],
            top_k=top_k,
        )
        return result
