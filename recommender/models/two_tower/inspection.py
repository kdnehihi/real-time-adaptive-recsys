from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from recommender.models.two_tower.config import EmbeddingTableConfig


@dataclass(frozen=True)
class EmbeddingParameterReport:
    feature_name: str
    num_embeddings: int
    embedding_dim: int
    parameters: int
    fp32_memory_mb: float


def table_parameter_report(config: EmbeddingTableConfig) -> EmbeddingParameterReport:
    parameters = config.num_embeddings * config.embedding_dim
    return EmbeddingParameterReport(
        feature_name=config.feature_name,
        num_embeddings=config.num_embeddings,
        embedding_dim=config.embedding_dim,
        parameters=parameters,
        fp32_memory_mb=parameters * 4 / (1024 * 1024),
    )


def embedding_parameter_reports(tables: Mapping[str, EmbeddingTableConfig]) -> list[EmbeddingParameterReport]:
    return [table_parameter_report(config) for config in tables.values()]


def total_embedding_parameters(tables: Mapping[str, EmbeddingTableConfig]) -> int:
    return sum(config.num_embeddings * config.embedding_dim for config in tables.values())
