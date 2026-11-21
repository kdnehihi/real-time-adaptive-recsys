from argparse import Namespace
import json
import sys

import pandas as pd
import pytest
import torch

from recommender.models.adaptive.checkpoint import (
    INPUT_COLUMNS,
    ITEM_NUMERIC,
    USER_NUMERIC,
    V4TransformerRetrieval,
)
from scripts import compare_corrected_adaptive_v4 as comparison
from scripts import train_corrected_adaptive_v4 as training
from scripts.train_corrected_adaptive_v4 import checkpoint_payload, prepare_positive_split


def base_row(example_id: str, target_class: str = "STRONG_POSITIVE") -> dict:
    row = {name: 0 for name in INPUT_COLUMNS}
    row.update(
        {
            "example_id": example_id,
            "user_id": 7,
            "video_id": 100,
            "as_of_time": pd.Timestamp("2022-04-10T00:00:02"),
            "current_time_ms": 2000,
            "target_class": target_class,
            "user_active_degree": "FULL_ACTIVE",
            "video_type": "NORMAL",
            "upload_type": "LongImport",
            "recent_video_ids": [999],
            "recent_target_classes": ["AMBIGUOUS_WEAK"],
            "recent_engagement_strengths": [0.1],
            "recent_long_views": [0],
            "recent_likes": [0],
            "recent_hates": [0],
            "recent_clicks": [0],
            "recent_time_ms": [2000],
            "recent_event_count": 1,
            "seconds_since_last_event": 0.0,
        }
    )
    return row


def corrected_row(example_id: str) -> dict:
    return {
        "example_id": example_id,
        "current_time_ms": 2000,
        "recent_video_ids": [10],
        "recent_target_classes": ["STRONG_POSITIVE"],
        "recent_engagement_strengths": [1.0],
        "recent_long_views": [1],
        "recent_likes": [0],
        "recent_hates": [0],
        "recent_clicks": [1],
        "recent_time_ms": [1000],
        "recent_event_count": 1,
        "seconds_since_last_event": 1.0,
    }


def test_prepare_positive_split_replaces_legacy_sequence_out_of_core(tmp_path):
    adaptive = tmp_path / "adaptive"
    corrected = tmp_path / "corrected"
    base_dir = adaptive / "sequence_examples/train"
    delta_dir = corrected / "sequences/split=train"
    base_dir.mkdir(parents=True)
    delta_dir.mkdir(parents=True)
    pd.DataFrame([base_row("keep"), base_row("skip", "AMBIGUOUS_WEAK")]).to_parquet(
        base_dir / "part.parquet", index=False
    )
    pd.DataFrame([corrected_row("keep"), corrected_row("skip")]).to_parquet(
        delta_dir / "part.parquet", index=False
    )
    args = Namespace(
        adaptive_root=adaptive,
        corrected_root=corrected,
        work_dir=tmp_path / "work",
        overwrite_prepared=False,
    )
    args.work_dir.mkdir()

    output = prepare_positive_split(args, "train", 0)
    frame = pd.read_parquet(output)

    assert frame["example_id"].tolist() == ["keep"]
    assert frame["recent_video_ids"].iloc[0].tolist() == [10]
    assert frame["recent_time_ms"].iloc[0].tolist() == [1000]


def test_corrected_queries_preserve_order_and_require_strict_prior(monkeypatch, tmp_path):
    legacy = pd.DataFrame([base_row("second"), base_row("first")])
    delta = pd.DataFrame([corrected_row("first"), corrected_row("second")])
    monkeypatch.setattr(comparison, "read_parquet", lambda *args, **kwargs: delta.copy())

    result = comparison.corrected_queries(legacy, tmp_path)

    assert result["example_id"].tolist() == ["second", "first"]
    assert [list(value) for value in result["recent_video_ids"]] == [[10], [10]]
    assert [list(value) for value in result["recent_time_ms"]] == [[1000], [1000]]

    leaking = delta.copy()
    leaking["recent_time_ms"] = [[2000], [2000]]
    monkeypatch.setattr(comparison, "read_parquet", lambda *args, **kwargs: leaking.copy())
    with pytest.raises(AssertionError, match="strict-prior"):
        comparison.corrected_queries(legacy, tmp_path)


def test_resume_checkpoint_uses_fp16_floating_weights():
    model = torch.nn.Linear(3, 2)
    payload = checkpoint_payload(model, {"retrieval_dim": 2}, {"video_id": 4}, 2, [], 0.1, 2)

    assert payload["epoch"] == 2
    assert payload["model_state_dict"]["weight"].dtype == torch.float16
    assert payload["resume_contract"] == "weights_and_history_only_optimizer_resets"


def test_tiny_corrected_v4_training_smoke(monkeypatch, tmp_path):
    adaptive = tmp_path / "adaptive"
    corrected = tmp_path / "corrected"
    base_gold = tmp_path / "base_gold"
    for split, count in (("train", 4), ("validation", 2)):
        base_dir = adaptive / f"sequence_examples/{split}"
        delta_dir = corrected / f"sequences/split={split}"
        base_dir.mkdir(parents=True)
        delta_dir.mkdir(parents=True)
        rows = []
        deltas = []
        for index in range(count):
            example_id = f"{split}-{index}"
            row = base_row(example_id)
            row["user_id"] = index % 2 + 1
            row["video_id"] = index + 10
            row["current_time_ms"] = 2000 + index
            row["as_of_time"] = pd.to_datetime(row["current_time_ms"], unit="ms")
            rows.append(row)
            delta = corrected_row(example_id)
            delta["current_time_ms"] = row["current_time_ms"]
            deltas.append(delta)
        pd.DataFrame(rows).to_parquet(base_dir / "part.parquet", index=False)
        pd.DataFrame(deltas).to_parquet(delta_dir / "part.parquet", index=False)

    for name, values in {
        "user_active_degree": ["FULL_ACTIVE"],
        "video_type": ["NORMAL"],
        "upload_type": ["LongImport"],
    }.items():
        path = base_gold / "vocabularies" / name
        path.mkdir(parents=True)
        pd.DataFrame({name: values, f"{name}_idx": [1]}).to_parquet(path / "part.parquet", index=False)
    stats = base_gold / "transforms/numeric_stats.json"
    stats.parent.mkdir(parents=True)
    stats.write_text(json.dumps({"features": {}}))

    output = tmp_path / "output"
    argv = [
        "train_corrected_adaptive_v4.py",
        "--adaptive-root", str(adaptive),
        "--base-gold-root", str(base_gold),
        "--corrected-root", str(corrected),
        "--work-dir", str(tmp_path / "work"),
        "--output-dir", str(output),
        "--max-train-examples", "4",
        "--max-validation-examples", "2",
        "--batch-size", "2",
        "--eval-batch-size", "2",
        "--negative-sample-count", "4",
        "--validation-batches", "1",
        "--epochs", "1",
        "--checkpoint-every", "1",
        "--device", "cpu",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    training.main()

    checkpoint = torch.load(output / "best_model.pt", map_location="cpu", weights_only=False)
    assert checkpoint["config"]["data_contract"] == "corrected_adaptive_sequences_v2"
    assert checkpoint["epoch"] == 1
    assert (output / "metrics.json").exists()


def test_sequence_encoder_supports_bfloat16_autocast_with_empty_rows():
    config = {
        "retrieval_dim": 8,
        "transformer_dim": 8,
        "transformer_heads": 2,
        "transformer_layers": 1,
        "transformer_dropout": 0.0,
        "user_hidden_dims": (16,),
        "item_hidden_dims": (16,),
        "dropout": 0.0,
    }
    sizes = {
        "user_id": 4,
        "user_active_degree": 3,
        "video_id": 20,
        "video_type": 3,
        "upload_type": 3,
        "event_type": 4,
    }
    model = V4TransformerRetrieval(sizes, config)
    batch = {
        "user_id": torch.tensor([1, 2]),
        "user_active_degree": torch.tensor([1, 1]),
        "video_id": torch.tensor([3, 4]),
        "video_type": torch.tensor([1, 1]),
        "upload_type": torch.tensor([1, 1]),
        "user_numeric": torch.zeros(2, len(USER_NUMERIC)),
        "item_numeric": torch.zeros(2, len(ITEM_NUMERIC)),
        "seq_video_id": torch.tensor([[0, 0, 0], [0, 5, 6]]),
        "seq_event_type": torch.tensor([[0, 0, 0], [0, 1, 2]]),
        "seq_signals": torch.zeros(2, 3, 5),
        "seq_time_bucket": torch.zeros(2, 3, dtype=torch.long),
        "seq_padding_mask": torch.tensor([[True, True, True], [True, False, False]]),
    }
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        mixed = model.encode_user(batch)
        empty = model.encode_user({**batch, "seq_padding_mask": torch.ones(2, 3, dtype=torch.bool)})

    assert mixed.shape == (2, 8)
    assert empty.shape == (2, 8)
    assert torch.isfinite(mixed).all()
    assert torch.isfinite(empty).all()


def test_comparison_annotates_strict_prior_repeated_targets():
    protocol = pd.DataFrame(
        {
            "user_id": [1, 1, 2],
            "video_id": [10, 20, 30],
            "query_time_ms": [2000, 1000, 3000],
        }
    )
    history = {
        1: (
            torch.tensor([500, 1500]).numpy(),
            torch.tensor([20, 10]).numpy(),
        ),
        2: (
            torch.tensor([1000]).numpy(),
            torch.tensor([99]).numpy(),
        ),
    }

    result = comparison.annotate_target_repeats(protocol, history)

    assert result["target_repeated_full_history"].tolist() == [True, True, False]
