import asyncio
import numpy as np
import pandas as pd
import pytest
import torch
from fastapi.testclient import TestClient

from recommender.serving.api import DynamicBatcher, create_app
from recommender.serving.exact_retrieval import ExactAdaptiveIndex
from recommender.serving.runtime import AdaptiveServingRuntime, RecommendationRequest


class StubEncoder:
    config = {"sequence_max_len": 3}

    def encode(self, frame: pd.DataFrame) -> np.ndarray:
        assert frame["recent_video_ids"].iloc[0] == [10]
        return np.repeat(np.asarray([[1.0, 0.0]], dtype="float32"), len(frame), axis=0)


def runtime() -> AdaptiveServingRuntime:
    index = ExactAdaptiveIndex(
        np.asarray([10, 20]),
        np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype="float32"),
        np.asarray([5, 6]),
        np.zeros(2, dtype="float32"),
        np.zeros(2, dtype="int64"),
        torch.device("cpu"),
    )
    return AdaptiveServingRuntime(StubEncoder(), index)


def test_request_flow_encodes_state_and_filters_seen_item() -> None:
    request = RecommendationRequest(
        user_id=1,
        current_time_ms=200,
        recent_events=({"video_id": 10, "time_ms": 100},),
        seen_item_ids=frozenset({10}),
        top_k=2,
    )
    result = runtime().recommend([request])
    assert result.item_ids.tolist() == [[20, -1]]


def test_request_flow_rejects_same_timestamp_event() -> None:
    request = RecommendationRequest(
        user_id=1,
        current_time_ms=100,
        recent_events=({"video_id": 10, "time_ms": 100},),
    )
    with pytest.raises(ValueError, match="strictly earlier"):
        runtime().recommend([request])


def test_fastapi_health_and_recommendation_contract() -> None:
    with TestClient(create_app(runtime())) as client:
        assert client.get("/health").json() == {
            "status": "ok",
            "candidate_items": 2,
            "max_batch_size": 32,
            "batch_wait_ms": 5.0,
        }
        response = client.post(
            "/recommend",
            json={
                "user_id": 1,
                "current_time_ms": 200,
                "recent_events": [{"video_id": 10, "time_ms": 100}],
                "seen_item_ids": [10],
                "top_k": 2,
            },
        )
        assert response.status_code == 200
        assert response.json()["item_ids"] == [20, -1]


def test_fastapi_rejects_non_prior_event() -> None:
    with TestClient(create_app(runtime())) as client:
        response = client.post(
            "/recommend",
            json={
                "user_id": 1,
                "current_time_ms": 100,
                "recent_events": [{"video_id": 10, "time_ms": 100}],
            },
        )
        assert response.status_code == 422
        assert "strictly earlier" in response.json()["detail"]


def test_dynamic_batcher_combines_concurrent_requests() -> None:
    base = runtime()
    observed: list[int] = []

    class RecordingRuntime:
        def recommend(self, requests):
            observed.append(len(requests))
            return base.recommend(requests)

    request = RecommendationRequest(
        user_id=1,
        current_time_ms=200,
        recent_events=({"video_id": 10, "time_ms": 100},),
        top_k=1,
    )

    async def scenario() -> None:
        batcher = DynamicBatcher(RecordingRuntime(), max_batch_size=8, wait_ms=20)
        await batcher.start()
        try:
            results = await asyncio.gather(*[batcher.submit(request) for _ in range(8)])
            assert all(result["valid_count"] >= 1 for result in results)
        finally:
            await batcher.stop()

    asyncio.run(scenario())
    assert observed == [8]
