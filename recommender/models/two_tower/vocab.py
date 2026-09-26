from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd

from recommender.models.two_tower.config import OOV_INDEX


@dataclass(frozen=True)
class Vocabulary:
    """Train-fitted value -> integer id mapping with index 0 reserved for OOV."""

    feature_name: str
    value_to_index: Mapping[object, int]
    oov_index: int = OOV_INDEX

    def __post_init__(self) -> None:
        bad = [idx for idx in self.value_to_index.values() if idx <= self.oov_index]
        if bad:
            raise ValueError(f"{self.feature_name} vocabulary indices must be > OOV index")

    @property
    def size(self) -> int:
        if not self.value_to_index:
            return self.oov_index + 1
        return max(self.value_to_index.values()) + 1

    @property
    def cardinality_excluding_oov(self) -> int:
        return len(self.value_to_index)

    def encode_one(self, value: object) -> int:
        if pd.isna(value):
            return self.oov_index
        return int(self.value_to_index.get(value, self.oov_index))

    def encode_many(self, values: Iterable[object]) -> list[int]:
        return [self.encode_one(value) for value in values]

    @classmethod
    def from_values(cls, feature_name: str, values: Iterable[object]) -> "Vocabulary":
        unique = sorted({value for value in values if not pd.isna(value)})
        return cls(feature_name, {value: idx + 1 for idx, value in enumerate(unique)})


def load_parquet_vocabulary(path: str | Path, feature_name: str) -> Vocabulary:
    """Load a Spark-written vocabulary directory or parquet file.

    Expected columns are `<feature_name>` and `<feature_name>_idx`.
    """

    path = Path(path)
    df = pd.read_parquet(path)
    value_col = feature_name
    idx_col = f"{feature_name}_idx"
    missing = {value_col, idx_col} - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing vocabulary columns: {sorted(missing)}")
    mapping = dict(zip(df[value_col].tolist(), df[idx_col].astype(int).tolist(), strict=True))
    return Vocabulary(feature_name, mapping)


def build_vocabulary_from_frame(frame: pd.DataFrame, feature_name: str) -> Vocabulary:
    if feature_name not in frame.columns:
        raise ValueError(f"Cannot build {feature_name} vocabulary; column is missing")
    return Vocabulary.from_values(feature_name, frame[feature_name].tolist())
