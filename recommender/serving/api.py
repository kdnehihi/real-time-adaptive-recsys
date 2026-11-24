from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from recommender.serving.runtime import AdaptiveServingRuntime, RecommendationRequest


class EventPayload(BaseModel):
    video_id: int
    time_ms: int
    target_class: str = "AMBIGUOUS_WEAK"
    engagement_strength: float = 0.0
    long_view: float = 0.0
    is_like: float = 0.0
    is_hate: float = 0.0
    is_click: float = 0.0


class RecommendPayload(BaseModel):
    user_id: int
    current_time_ms: int
    recent_events: list[EventPayload] = Field(default_factory=list)
    seen_item_ids: list[int] = Field(default_factory=list)
    user_active_degree: Any = None
    user_features: dict[str, float] = Field(default_factory=dict)
    top_k: int = Field(default=10, ge=1, le=1000)


def create_app(runtime: AdaptiveServingRuntime) -> FastAPI:
    """Create a thin API over one preloaded, immutable serving runtime."""
    app = FastAPI(title="Adaptive Recommender API", version="1.0")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "candidate_items": len(runtime.index.item_ids_np)}

    @app.post("/recommend")
    def recommend(payload: RecommendPayload) -> dict[str, Any]:
        try:
            request = RecommendationRequest(
                user_id=payload.user_id,
                current_time_ms=payload.current_time_ms,
                recent_events=tuple(
                    event.model_dump() if hasattr(event, "model_dump") else event.dict()
                    for event in payload.recent_events
                ),
                seen_item_ids=frozenset(payload.seen_item_ids),
                user_active_degree=payload.user_active_degree,
                user_features=payload.user_features,
                top_k=payload.top_k,
            )
            result = runtime.recommend([request])
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {
            "user_id": payload.user_id,
            "item_ids": result.item_ids[0, : payload.top_k].tolist(),
            "scores": result.scores[0, : payload.top_k].tolist(),
            "valid_count": int(result.valid_counts[0]),
        }

    return app
