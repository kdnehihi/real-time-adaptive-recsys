import pandas as pd
import torch

from scripts.evaluate_corrected_v4_warm_retention import (
    apply_static_retention,
    select_calibration_config,
)


def test_static_retention_can_target_only_warm_candidates():
    base = torch.tensor([[1.0, 1.0, 1.0]])
    static = torch.tensor([[0.5, 0.25, -0.5]])
    warm = torch.tensor([[1.0, 0.0, 1.0]])

    all_scores = apply_static_retention(base, static, warm, (0.2, "all"))
    warm_scores = apply_static_retention(base, static, warm, (0.2, "warm"))

    assert torch.allclose(all_scores, torch.tensor([[1.1, 1.05, 0.9]]))
    assert torch.allclose(warm_scores, torch.tensor([[1.1, 1.0, 0.9]]))


def test_calibration_selection_requires_overall_warm_and_cold_non_regression():
    rows = [
        {
            "static_weight": 0.0,
            "static_mode": "none",
            "hitrate@1000": 0.02,
            "ndcg@1000": 0.01,
            "warm_hitrate@1000": 0.04,
            "cold_hitrate@1000": 0.005,
        },
        {
            "static_weight": 0.2,
            "static_mode": "all",
            "hitrate@1000": 0.023,
            "ndcg@1000": 0.011,
            "warm_hitrate@1000": 0.05,
            "cold_hitrate@1000": 0.004,
        },
        {
            "static_weight": 0.3,
            "static_mode": "warm",
            "hitrate@1000": 0.022,
            "ndcg@1000": 0.012,
            "warm_hitrate@1000": 0.048,
            "cold_hitrate@1000": 0.005,
        },
    ]

    assert select_calibration_config(rows) == (0.3, "warm")


def test_calibration_falls_back_when_no_safe_static_config_exists():
    rows = [
        {
            "static_weight": 0.0,
            "static_mode": "none",
            "hitrate@1000": 0.02,
            "ndcg@1000": 0.01,
            "warm_hitrate@1000": 0.04,
            "cold_hitrate@1000": 0.005,
        },
        {
            "static_weight": 0.2,
            "static_mode": "warm",
            "hitrate@1000": 0.019,
            "ndcg@1000": 0.02,
            "warm_hitrate@1000": 0.05,
            "cold_hitrate@1000": 0.005,
        },
    ]

    assert select_calibration_config(rows) == (0.0, "none")
