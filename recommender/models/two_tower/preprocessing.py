from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import torch

from recommender.models.two_tower.config import (
    TARGET_ONLY_COLUMNS,
    NumericalTransformConfig,
    TwoTowerFeatureConfig,
)
from recommender.models.two_tower.vocab import Vocabulary


@dataclass(frozen=True)
class NumericalPreprocessor:
    """Applies train-only imputation, clipping, and normalization."""

    stats: Mapping[str, Mapping[str, float]]
    config: NumericalTransformConfig = NumericalTransformConfig()

    @classmethod
    def from_json(cls, path: str | Path, config: NumericalTransformConfig = NumericalTransformConfig()) -> "NumericalPreprocessor":
        with Path(path).open() as f:
            payload = json.load(f)
        stats = payload.get("features", payload)
        if not isinstance(stats, dict):
            raise ValueError(f"Unsupported numeric stats format in {path}")
        return cls(stats=stats, config=config)

    def transform_frame(self, frame: pd.DataFrame, features: tuple[str, ...]) -> pd.DataFrame:
        missing = sorted(set(features) - set(frame.columns))
        if missing:
            raise ValueError(f"Missing numerical columns: {missing}")

        output = pd.DataFrame(index=frame.index)
        for feature in features:
            feature_stats = self.stats.get(feature)
            if feature_stats is None:
                raise ValueError(f"No train numerical stats found for {feature}")

            values = pd.to_numeric(frame[feature], errors="coerce").astype("float32")
            fill_value = self._fill_value(feature_stats)
            values = values.fillna(fill_value)

            if self.config.clip:
                lower = feature_stats.get("p01", feature_stats.get("min"))
                upper = feature_stats.get("p99", feature_stats.get("max"))
                if lower is not None and upper is not None:
                    values = values.clip(float(lower), float(upper))

            if self.config.normalize:
                mean = float(feature_stats.get("mean", 0.0) or 0.0)
                std = float(feature_stats.get("stddev", 0.0) or 0.0)
                if std > self.config.eps:
                    values = (values - mean) / std
                else:
                    values = values * 0.0

            clean = np.nan_to_num(values.to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            output[feature] = clean
        return output

    def transform_tensor(self, frame: pd.DataFrame, features: tuple[str, ...]) -> torch.Tensor:
        transformed = self.transform_frame(frame, features)
        return torch.as_tensor(transformed.to_numpy(dtype=np.float32), dtype=torch.float32)

    def _fill_value(self, feature_stats: Mapping[str, float]) -> float:
        if self.config.impute_strategy == "zero":
            return 0.0
        if self.config.impute_strategy == "median":
            return float(feature_stats.get("median", 0.0) or 0.0)
        return float(feature_stats.get("mean", 0.0) or 0.0)


@dataclass(frozen=True)
class FeatureBatch:
    user_categorical: dict[str, torch.Tensor]
    user_numeric: torch.Tensor
    item_categorical: dict[str, torch.Tensor]
    item_numeric: torch.Tensor


@dataclass
class FeatureBatchPreprocessor:
    """Converts raw joined Gold rows into tensors consumed by encoders."""

    config: TwoTowerFeatureConfig
    vocabularies: Mapping[str, Vocabulary]
    numerical_preprocessor: NumericalPreprocessor

    def check_no_target_leakage(self, columns: set[str] | list[str] | tuple[str, ...]) -> None:
        leaked = sorted(set(columns) & set(self.config.target_only_columns | TARGET_ONLY_COLUMNS))
        if leaked:
            raise ValueError(f"Target/current-event outcome columns cannot be model inputs: {leaked}")

    def encode_categorical_frame(self, frame: pd.DataFrame, features: tuple[str, ...]) -> dict[str, torch.Tensor]:
        missing = sorted(set(features) - set(frame.columns))
        if missing:
            raise ValueError(f"Missing categorical columns: {missing}")

        encoded: dict[str, torch.Tensor] = {}
        for feature in features:
            vocab = self.vocabularies.get(feature)
            if vocab is None:
                raise ValueError(f"No vocabulary found for categorical feature {feature}")
            values = vocab.encode_many(frame[feature].tolist())
            encoded[feature] = torch.as_tensor(values, dtype=torch.long)
        return encoded

    def encode_user(self, frame: pd.DataFrame) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        self.check_no_target_leakage(self.config.user_input_features)
        categorical_features = self.config.user_id_features + self.config.user_categorical_features
        categorical = self.encode_categorical_frame(frame, categorical_features)
        numeric = self.numerical_preprocessor.transform_tensor(frame, self.config.user_numeric_features)
        return categorical, numeric

    def encode_item(self, frame: pd.DataFrame) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        self.check_no_target_leakage(self.config.item_input_features)
        categorical_features = self.config.item_id_features + self.config.item_categorical_features
        categorical = self.encode_categorical_frame(frame, categorical_features)
        numeric = self.numerical_preprocessor.transform_tensor(frame, self.config.item_numeric_features)
        return categorical, numeric

    def encode_batch(self, user_frame: pd.DataFrame, item_frame: pd.DataFrame) -> FeatureBatch:
        user_categorical, user_numeric = self.encode_user(user_frame)
        item_categorical, item_numeric = self.encode_item(item_frame)
        return FeatureBatch(
            user_categorical=user_categorical,
            user_numeric=user_numeric,
            item_categorical=item_categorical,
            item_numeric=item_numeric,
        )
