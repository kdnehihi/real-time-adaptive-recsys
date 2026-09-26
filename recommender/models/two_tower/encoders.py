from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn

from recommender.models.two_tower.config import EmbeddingTableConfig


class CategoricalEmbeddingBlock(nn.Module):
    """A collection of independent trainable embedding tables."""

    def __init__(self, tables: Mapping[str, EmbeddingTableConfig]) -> None:
        super().__init__()
        self.table_configs = dict(tables)
        self.embeddings = nn.ModuleDict(
            {
                name: nn.Embedding(
                    num_embeddings=config.num_embeddings,
                    embedding_dim=config.embedding_dim,
                    padding_idx=config.oov_index,
                )
                for name, config in self.table_configs.items()
            }
        )

    @property
    def output_dim(self) -> int:
        return sum(config.embedding_dim for config in self.table_configs.values())

    def forward(self, indices: Mapping[str, torch.Tensor]) -> torch.Tensor:
        missing = sorted(set(self.embeddings.keys()) - set(indices.keys()))
        if missing:
            raise ValueError(f"Missing categorical tensors: {missing}")

        pieces = []
        batch_size: int | None = None
        for name, embedding in self.embeddings.items():
            values = indices[name].long()
            if values.ndim != 1:
                raise ValueError(f"{name} indices must be a 1D tensor")
            if values.numel() and (values.min() < 0 or values.max() >= embedding.num_embeddings):
                raise ValueError(f"{name} indices are outside embedding table bounds")
            batch_size = values.shape[0] if batch_size is None else batch_size
            if values.shape[0] != batch_size:
                raise ValueError("All categorical tensors must share the same batch size")
            pieces.append(embedding(values))

        if pieces:
            return torch.cat(pieces, dim=1)
        if batch_size is None:
            batch_size = 0
        return torch.empty((batch_size, 0), dtype=torch.float32)


class NumericalFeatureBlock(nn.Module):
    """Pass-through numerical block with finite-value validation."""

    def __init__(self, feature_names: tuple[str, ...]) -> None:
        super().__init__()
        self.feature_names = feature_names

    @property
    def output_dim(self) -> int:
        return len(self.feature_names)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = values.float()
        if values.ndim != 2:
            raise ValueError("Numerical tensor must be 2D: [batch_size, num_features]")
        if values.shape[1] != self.output_dim:
            raise ValueError(f"Expected {self.output_dim} numerical features, got {values.shape[1]}")
        if not torch.isfinite(values).all():
            raise ValueError("Numerical tensor contains NaN or Inf")
        return values


@dataclass(frozen=True)
class EncoderOutput:
    vector: torch.Tensor
    categorical_embeddings: torch.Tensor
    numerical_features: torch.Tensor


class _FeatureEncoder(nn.Module):
    def __init__(
        self,
        categorical_tables: Mapping[str, EmbeddingTableConfig],
        numerical_features: tuple[str, ...],
    ) -> None:
        super().__init__()
        self.categorical = CategoricalEmbeddingBlock(categorical_tables)
        self.numerical = NumericalFeatureBlock(numerical_features)

    @property
    def output_dim(self) -> int:
        return self.categorical.output_dim + self.numerical.output_dim

    def forward(self, categorical_indices: Mapping[str, torch.Tensor], numerical_values: torch.Tensor) -> torch.Tensor:
        return self.forward_with_parts(categorical_indices, numerical_values).vector

    def forward_with_parts(
        self,
        categorical_indices: Mapping[str, torch.Tensor],
        numerical_values: torch.Tensor,
    ) -> EncoderOutput:
        categorical = self.categorical(categorical_indices)
        numerical = self.numerical(numerical_values)
        return EncoderOutput(vector=torch.cat([categorical, numerical], dim=1), categorical_embeddings=categorical, numerical_features=numerical)


class UserFeatureEncoder(_FeatureEncoder):
    """Encodes user-side Gold features into a dense feature representation."""


class ItemFeatureEncoder(_FeatureEncoder):
    """Encodes item-side Gold features into a dense feature representation."""
