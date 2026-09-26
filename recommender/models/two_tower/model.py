from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from recommender.models.two_tower.config import EmbeddingTableConfig
from recommender.models.two_tower.encoders import ItemFeatureEncoder, UserFeatureEncoder


def build_mlp(input_dim: int, hidden_dims: tuple[int, ...], output_dim: int, dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous = input_dim
    for hidden in hidden_dims:
        layers.append(nn.Linear(previous, hidden))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        previous = hidden
    layers.append(nn.Linear(previous, output_dim))
    return nn.Sequential(*layers)


@dataclass(frozen=True)
class TwoTowerOutput:
    user_embedding: torch.Tensor
    item_embedding: torch.Tensor
    logits: torch.Tensor


class TwoTowerRetrievalModel(nn.Module):
    """Basic Two-Tower retrieval model with in-batch dot-product scoring."""

    def __init__(
        self,
        user_tables: Mapping[str, EmbeddingTableConfig],
        item_tables: Mapping[str, EmbeddingTableConfig],
        user_numeric_features: tuple[str, ...],
        item_numeric_features: tuple[str, ...],
        retrieval_dim: int = 32,
        user_hidden_dims: tuple[int, ...] = (64,),
        item_hidden_dims: tuple[int, ...] = (64,),
        dropout: float = 0.1,
        temperature: float = 0.07,
    ) -> None:
        super().__init__()
        self.user_feature_encoder = UserFeatureEncoder(user_tables, user_numeric_features)
        self.item_feature_encoder = ItemFeatureEncoder(item_tables, item_numeric_features)
        self.user_tower = build_mlp(self.user_feature_encoder.output_dim, user_hidden_dims, retrieval_dim, dropout)
        self.item_tower = build_mlp(self.item_feature_encoder.output_dim, item_hidden_dims, retrieval_dim, dropout)
        self.temperature = temperature

    def encode_user(self, categorical_indices: Mapping[str, torch.Tensor], numerical_values: torch.Tensor) -> torch.Tensor:
        features = self.user_feature_encoder(categorical_indices, numerical_values)
        return F.normalize(self.user_tower(features), dim=1)

    def encode_item(self, categorical_indices: Mapping[str, torch.Tensor], numerical_values: torch.Tensor) -> torch.Tensor:
        features = self.item_feature_encoder(categorical_indices, numerical_values)
        return F.normalize(self.item_tower(features), dim=1)

    def forward(
        self,
        user_categorical: Mapping[str, torch.Tensor],
        user_numeric: torch.Tensor,
        item_categorical: Mapping[str, torch.Tensor],
        item_numeric: torch.Tensor,
    ) -> TwoTowerOutput:
        user_embedding = self.encode_user(user_categorical, user_numeric)
        item_embedding = self.encode_item(item_categorical, item_numeric)
        logits = user_embedding @ item_embedding.T / self.temperature
        return TwoTowerOutput(user_embedding=user_embedding, item_embedding=item_embedding, logits=logits)
