from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from recommender.serving.runtime import AdaptiveServingRuntime, RecommendationRequest


@dataclass
class _PendingRequest:
    request: RecommendationRequest
    future: asyncio.Future


class DynamicBatcher:
    """Combine concurrent HTTP requests into bounded GPU retrieval batches."""

    def __init__(self, runtime: AdaptiveServingRuntime, max_batch_size: int = 32, wait_ms: float = 5.0) -> None:
        self.runtime = runtime
        self.max_batch_size = int(max_batch_size)
        self.wait_seconds = float(wait_ms) / 1000.0
        self.queue: asyncio.Queue[_PendingRequest] = asyncio.Queue()
        self.worker: asyncio.Task | None = None

    async def start(self) -> None:
        if self.worker is None:
            self.worker = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self.worker is not None:
            self.worker.cancel()
            try:
                await self.worker
            except asyncio.CancelledError:
                pass
            self.worker = None

    async def submit(self, request: RecommendationRequest):
        if self.worker is None:
            raise RuntimeError("Dynamic batcher has not started")
        future = asyncio.get_running_loop().create_future()
        await self.queue.put(_PendingRequest(request, future))
        return await future

    async def _run(self) -> None:
        while True:
            first = await self.queue.get()
            pending = [first]
            deadline = asyncio.get_running_loop().time() + self.wait_seconds
            while len(pending) < self.max_batch_size:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    break
                try:
                    pending.append(await asyncio.wait_for(self.queue.get(), remaining))
                except asyncio.TimeoutError:
                    break
            try:
                result = await asyncio.to_thread(
                    self.runtime.recommend, [item.request for item in pending]
                )
                for row, item in enumerate(pending):
                    if not item.future.cancelled():
                        item.future.set_result(
                            {
                                "item_ids": result.item_ids[row],
                                "scores": result.scores[row],
                                "valid_count": int(result.valid_counts[row]),
                            }
                        )
            except Exception as exc:
                for item in pending:
                    if not item.future.cancelled():
                        item.future.set_exception(exc)


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


def create_app(
    runtime: AdaptiveServingRuntime,
    max_batch_size: int = 32,
    batch_wait_ms: float = 5.0,
) -> FastAPI:
    """Create a thin API over one preloaded, immutable serving runtime."""
    batcher = DynamicBatcher(runtime, max_batch_size=max_batch_size, wait_ms=batch_wait_ms)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        await batcher.start()
        yield
        await batcher.stop()

    app = FastAPI(title="Adaptive Recommender API", version="1.0", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "candidate_items": len(runtime.index.item_ids_np),
            "max_batch_size": max_batch_size,
            "batch_wait_ms": batch_wait_ms,
        }

    @app.post("/recommend")
    async def recommend(payload: RecommendPayload) -> dict[str, Any]:
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
            result = await batcher.submit(request)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {
            "user_id": payload.user_id,
            "item_ids": result["item_ids"][: payload.top_k].tolist(),
            "scores": result["scores"][: payload.top_k].tolist(),
            "valid_count": result["valid_count"],
        }

    return app
