import numpy as np
import pandas as pd
import pytest
import torch
from fastapi.testclient import TestClient

from recommender.serving.api import create_app
from recommender.serving.exact_retrieval import ExactAdaptiveIndex
from recommender.serving.runtime import AdaptiveServingRuntime, RecommendationRequest


class StubEncoder:
    config = {"sequence_max_len": 3}

    def encode(self, frame: pd.DataFrame) -> np.ndarray:
        assert frame["recent_video_ids"].iloc[0] == [10]
        return np.asarray([[1.0, 0.0]], dtype="float32")


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
    client = TestClient(create_app(runtime()))
    assert client.get("/health").json() == {"status": "ok", "candidate_items": 2}
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
    client = TestClient(create_app(runtime()))
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
