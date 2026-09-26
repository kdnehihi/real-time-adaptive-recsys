from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from recommender.models.two_tower.config import DEFAULT_CONFIG, TwoTowerFeatureConfig, embedding_table_configs
from recommender.models.two_tower.data import EncodedTwoTowerBatch, TwoTowerCollator, TwoTowerFrameDataset, load_joined_split
from recommender.models.two_tower.evaluation import evaluate_in_batch
from recommender.models.two_tower.inspection import embedding_parameter_reports
from recommender.models.two_tower.model import TwoTowerRetrievalModel
from recommender.models.two_tower.preprocessing import FeatureBatchPreprocessor, NumericalPreprocessor
from recommender.models.two_tower.vocab import Vocabulary, load_parquet_vocabulary


@dataclass
class TwoTowerTrainConfig:
    run_name: str = "two_tower_smoke"
    gold_root: str = "data/gold/two_tower/v1_ready"
    output_dir: str = "data/models/two_tower/smoke"
    als_summary_path: str | None = "data/gold/als/v1_colab/evaluation_summary.json"
    mlflow_tracking_uri: str | None = "data/mlruns"
    mlflow_experiment: str = "two_tower_baseline"
    use_mlflow: bool = True
    seed: int = 42
    max_train_examples: int | None = 50_000
    max_validation_examples: int | None = 20_000
    target_classes: tuple[str, ...] = ("STRONG_POSITIVE",)
    batch_size: int = 512
    eval_batch_size: int = 512
    epochs: int = 1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    retrieval_dim: int = 32
    user_hidden_dims: tuple[int, ...] = (64,)
    item_hidden_dims: tuple[int, ...] = (64,)
    dropout: float = 0.1
    temperature: float = 0.07
    top_ks: tuple[int, ...] = (10, 50)
    eval_max_batches: int | None = 20
    early_stopping_patience: int | None = None
    early_stopping_metric: str = "validation_ndcg@50"
    early_stopping_mode: str = "max"
    early_stopping_min_delta: float = 1e-5
    num_workers: int = 0
    device: str = "auto"
    full_video_vocab_size: int | None = None
    notes: str = ""


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_config(path: str | Path) -> TwoTowerTrainConfig:
    with Path(path).open() as f:
        raw = json.load(f)
    for key in ("target_classes", "user_hidden_dims", "item_hidden_dims", "top_ks"):
        if key in raw and isinstance(raw[key], list):
            raw[key] = tuple(raw[key])
    return TwoTowerTrainConfig(**raw)


def save_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=str)


def resolve_device(raw: str) -> torch.device:
    if raw == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(raw)


def build_preprocessor_and_tables(
    config: TwoTowerTrainConfig,
    feature_config: TwoTowerFeatureConfig,
    train_frame,
) -> tuple[FeatureBatchPreprocessor, dict, dict, dict[str, Vocabulary]]:
    gold_root = Path(config.gold_root)
    vocabularies: dict[str, Vocabulary] = {
        "user_active_degree": load_parquet_vocabulary(gold_root / "vocabularies" / "user_active_degree", "user_active_degree"),
        "video_type": load_parquet_vocabulary(gold_root / "vocabularies" / "video_type", "video_type"),
        "upload_type": load_parquet_vocabulary(gold_root / "vocabularies" / "upload_type", "upload_type"),
        "user_id": Vocabulary.from_values("user_id", train_frame["user_id"].tolist()),
        "video_id": Vocabulary.from_values("video_id", train_frame["video_id"].tolist()),
    }
    numeric = NumericalPreprocessor.from_json(gold_root / "transforms" / "numeric_stats.json", feature_config.numerical)
    preprocessor = FeatureBatchPreprocessor(feature_config, vocabularies, numeric)

    user_vocab_sizes = {name: vocabularies[name].size for name in feature_config.user_id_features + feature_config.user_categorical_features}
    item_vocab_sizes = {name: vocabularies[name].size for name in feature_config.item_categorical_features}
    item_vocab_sizes["video_id"] = config.full_video_vocab_size or vocabularies["video_id"].size
    user_tables = embedding_table_configs(feature_config.user_id_features + feature_config.user_categorical_features, user_vocab_sizes, feature_config)
    item_tables = embedding_table_configs(feature_config.item_id_features + feature_config.item_categorical_features, item_vocab_sizes, feature_config)
    return preprocessor, user_tables, item_tables, vocabularies


def baseline_summary(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    with p.open() as f:
        data = json.load(f)
    return {
        "popularity_test": data.get("popularity_test"),
        "als_test": data.get("final_als_test"),
        "selected_als_hyperparameters": data.get("selected_als_hyperparameters"),
    }


def train_two_tower(config: TwoTowerTrainConfig) -> dict[str, Any]:
    set_seed(config.seed)
    feature_config = DEFAULT_CONFIG
    device = resolve_device(config.device)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "config.json", asdict(config))

    train_frame = load_joined_split(config.gold_root, "train", config.max_train_examples, config.target_classes, feature_config)
    validation_frame = load_joined_split(
        config.gold_root,
        "validation",
        config.max_validation_examples,
        config.target_classes,
        feature_config,
    )
    if train_frame.empty:
        raise ValueError("No training examples after filtering target_classes")
    if validation_frame.empty:
        raise ValueError("No validation examples after filtering target_classes")

    preprocessor, user_tables, item_tables, vocabularies = build_preprocessor_and_tables(config, feature_config, train_frame)
    model = TwoTowerRetrievalModel(
        user_tables=user_tables,
        item_tables=item_tables,
        user_numeric_features=feature_config.user_numeric_features,
        item_numeric_features=feature_config.item_numeric_features,
        retrieval_dim=config.retrieval_dim,
        user_hidden_dims=config.user_hidden_dims,
        item_hidden_dims=config.item_hidden_dims,
        dropout=config.dropout,
        temperature=config.temperature,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    collator = TwoTowerCollator(preprocessor)
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        TwoTowerFrameDataset(train_frame, preprocessor),
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=config.num_workers,
        generator=generator,
    )
    validation_loader = DataLoader(
        TwoTowerFrameDataset(validation_frame, preprocessor),
        batch_size=config.eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=config.num_workers,
    )

    history: list[dict[str, float]] = []
    best_metric: float | None = None
    best_epoch: int | None = None
    bad_epochs = 0
    best_model_path = output_dir / "best_model.pt"
    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = 0.0
        total_examples = 0
        for batch in train_loader:
            assert isinstance(batch, EncodedTwoTowerBatch)
            batch = batch.to(device)
            output = model(batch.user_categorical, batch.user_numeric, batch.item_categorical, batch.item_numeric)
            labels = torch.arange(output.logits.shape[0], device=device)
            loss = loss_fn(output.logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            batch_size = output.logits.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_examples += batch_size
        epoch_metrics = {"epoch": epoch, "train_loss": total_loss / max(total_examples, 1), "train_examples": total_examples}
        epoch_metrics.update({f"validation_{k}": v for k, v in evaluate_in_batch(model, validation_loader, device, config.top_ks, config.eval_max_batches).items()})
        history.append(epoch_metrics)
        print(epoch_metrics)

        current_metric = epoch_metrics.get(config.early_stopping_metric)
        if isinstance(current_metric, (int, float)):
            improved = (
                best_metric is None
                or (
                    config.early_stopping_mode == "max"
                    and current_metric > best_metric + config.early_stopping_min_delta
                )
                or (
                    config.early_stopping_mode == "min"
                    and current_metric < best_metric - config.early_stopping_min_delta
                )
            )
            if improved:
                best_metric = float(current_metric)
                best_epoch = epoch
                bad_epochs = 0
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "config": asdict(config),
                        "best_epoch": best_epoch,
                        "best_metric": best_metric,
                    },
                    best_model_path,
                )
            else:
                bad_epochs += 1
                if config.early_stopping_patience is not None and bad_epochs >= config.early_stopping_patience:
                    print(
                        f"Early stopping at epoch {epoch}; best {config.early_stopping_metric}="
                        f"{best_metric} at epoch {best_epoch}"
                    )
                    break

    final_validation = evaluate_in_batch(model, validation_loader, device, config.top_ks, config.eval_max_batches)
    model_path = output_dir / "model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(config),
            "feature_config": {
                "user_numeric_features": feature_config.user_numeric_features,
                "item_numeric_features": feature_config.item_numeric_features,
            },
        },
        model_path,
    )
    vocab_summary = {name: {"size": vocab.size, "cardinality_excluding_oov": vocab.cardinality_excluding_oov} for name, vocab in vocabularies.items()}
    reports = [asdict(report) for report in embedding_parameter_reports({**user_tables, **item_tables})]
    metrics = {
        "run_name": config.run_name,
        "device": str(device),
        "train_rows_loaded": len(train_frame),
        "validation_rows_loaded": len(validation_frame),
        "history": history,
        "final_validation": final_validation,
        "best_epoch": best_epoch,
        "best_metric": best_metric,
        "best_model_path": str(best_model_path) if best_model_path.exists() else None,
        "embedding_parameter_report": reports,
        "model_parameter_count": sum(p.numel() for p in model.parameters()),
        "baseline_summary": baseline_summary(config.als_summary_path),
    }
    save_json(output_dir / "metrics.json", metrics)
    save_json(output_dir / "vocab_summary.json", vocab_summary)
    save_json(
        output_dir / "comparison_summary.json",
        {
            "two_tower_validation": final_validation,
            "baseline_summary": metrics["baseline_summary"],
            "note": "Two-Tower metrics are in-batch retrieval metrics; ALS/popularity use logged top-K evaluation, so compare directionally only.",
        },
    )

    if config.use_mlflow:
        try:
            import mlflow

            tracking_uri = config.mlflow_tracking_uri
            if tracking_uri:
                mlflow.set_tracking_uri(str(Path(tracking_uri).resolve()))
            mlflow.set_experiment(config.mlflow_experiment)
            with mlflow.start_run(run_name=config.run_name):
                mlflow.log_params(
                    {
                        "batch_size": config.batch_size,
                        "epochs": config.epochs,
                        "learning_rate": config.learning_rate,
                        "retrieval_dim": config.retrieval_dim,
                        "user_hidden_dims": str(config.user_hidden_dims),
                        "item_hidden_dims": str(config.item_hidden_dims),
                        "temperature": config.temperature,
                        "train_rows_loaded": len(train_frame),
                        "validation_rows_loaded": len(validation_frame),
                    }
                )
                for key, value in final_validation.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(f"validation_{key}", value)
                if history:
                    mlflow.log_metric("train_loss", history[-1]["train_loss"])
                mlflow.log_artifacts(str(output_dir))
        except Exception as exc:
            metrics["mlflow_error"] = repr(exc)
            save_json(output_dir / "metrics.json", metrics)

    return metrics
