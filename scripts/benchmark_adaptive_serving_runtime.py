from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from fastapi.testclient import TestClient

from recommender.serving.api import create_app
from recommender.serving.runtime import AdaptiveServingRuntime, RecommendationRequest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark standalone Corrected V4 request serving.")
    parser.add_argument("--catalog-dir", type=Path, required=True)
    parser.add_argument("--preprocessing-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--batch-sizes", default="1,8,32")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def percentile(values: list[float], quantile: float) -> float:
    return float(np.quantile(np.asarray(values), quantile))


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA runtime is required for this serving benchmark")
    load_start = time.perf_counter()
    runtime = AdaptiveServingRuntime.from_artifacts(
        args.catalog_dir, args.preprocessing_dir, args.checkpoint, device
    )
    synchronize(device)
    load_seconds = time.perf_counter() - load_start
    user_values = np.load(args.preprocessing_dir / "vocab_user_id.npy", allow_pickle=False)
    item_values = runtime.index.item_ids_np
    max_clock = int(runtime.index.eligible_from_ms.max().item()) + 1

    def requests(count: int) -> list[RecommendationRequest]:
        output = []
        for offset in range(count):
            recent = tuple(
                {
                    "video_id": int(item_values[(offset * 11 + position) % len(item_values)]),
                    "time_ms": max_clock - (10 - position) * 1000,
                    "target_class": "STRONG_POSITIVE" if position % 3 == 0 else "AMBIGUOUS_WEAK",
                    "engagement_strength": float(position % 4),
                    "long_view": float(position % 2),
                    "is_click": 1.0,
                }
                for position in range(10)
            )
            output.append(
                RecommendationRequest(
                    user_id=int(user_values[offset % len(user_values)]),
                    current_time_ms=max_clock,
                    recent_events=recent,
                    seen_item_ids=frozenset(event["video_id"] for event in recent),
                    top_k=args.top_k,
                )
            )
        return output

    runtime.recommend(requests(2))
    synchronize(device)
    rows = []
    for batch_size in [int(value) for value in args.batch_sizes.split(",")]:
        batch = requests(batch_size)
        timings = []
        for _ in range(args.iterations):
            start = time.perf_counter()
            result = runtime.recommend(batch)
            synchronize(device)
            timings.append((time.perf_counter() - start) * 1000.0)
            if np.any(result.valid_counts < args.top_k):
                raise RuntimeError("Serving runtime returned fewer than top_k valid items")
        rows.append(
            {
                "batch_size": batch_size,
                "iterations": args.iterations,
                "batch_latency_p50_ms": percentile(timings, 0.50),
                "batch_latency_p95_ms": percentile(timings, 0.95),
                "mean_ms_per_request": statistics.mean(timings) / batch_size,
                "throughput_requests_per_second": 1000.0 * batch_size / statistics.mean(timings),
            }
        )

    client = TestClient(create_app(runtime))
    health = client.get("/health")
    if health.status_code != 200 or health.json()["candidate_items"] != len(item_values):
        raise RuntimeError("FastAPI health smoke failed")
    api_payload = {
        "user_id": int(user_values[0]),
        "current_time_ms": max_clock,
        "recent_events": list(requests(1)[0].recent_events),
        "seen_item_ids": list(requests(1)[0].seen_item_ids),
        "top_k": args.top_k,
    }
    api_response = client.post("/recommend", json=api_payload)
    if api_response.status_code != 200 or len(api_response.json()["item_ids"]) != args.top_k:
        raise RuntimeError(f"FastAPI recommendation smoke failed: {api_response.text}")
    report = {
        "benchmark": "corrected_v4_2_standalone_serving_v1",
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "load_seconds": load_seconds,
        "candidate_items": len(item_values),
        "top_k": args.top_k,
        "results": rows,
        "api_smoke": {"health": True, "recommend": True},
        "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        "gold_data_read": False,
        "locked_test_used": False,
    }
    (args.output_dir / "serving_benchmark.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
