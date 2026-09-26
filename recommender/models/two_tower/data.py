from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import pyarrow.dataset as ds
import torch
from torch.utils.data import Dataset

from recommender.models.two_tower.config import TwoTowerFeatureConfig
from recommender.models.two_tower.preprocessing import FeatureBatchPreprocessor


def parquet_files(path: str | Path) -> list[Path]:
    path = Path(path)
    if path.is_file():
        return [path] if path.suffix == ".parquet" else []
    if not path.exists():
        raise FileNotFoundError(f"Parquet path does not exist: {path}")
    files = sorted(p for p in path.rglob("*.parquet") if p.is_file() and not p.name.startswith("."))
    if not files:
        visible = sorted(str(p.relative_to(path)) for p in path.iterdir())[:20] if path.is_dir() else []
        raise FileNotFoundError(f"No parquet files found under {path}. Visible entries: {visible}")
    return files


def read_parquet_head(path: str | Path, max_rows: int | None, columns: list[str] | None = None) -> pd.DataFrame:
    files = parquet_files(path)
    dataset = ds.dataset([str(file) for file in files], format="parquet")
    if columns:
        missing = sorted(set(columns) - set(dataset.schema.names))
        if missing:
            raise ValueError(f"{path} is missing columns {missing}. Available columns: {dataset.schema.names}")
    if max_rows is None:
        return dataset.to_table(columns=columns).to_pandas()
    return dataset.head(max_rows, columns=columns).to_pandas()


def load_joined_split(
    gold_root: str | Path,
    split: str,
    max_examples: int | None,
    target_classes: Iterable[str] = ("STRONG_POSITIVE",),
    config: TwoTowerFeatureConfig = TwoTowerFeatureConfig(),
) -> pd.DataFrame:
    """Load a split into a pandas frame for the first PyTorch baseline.

    This intentionally reads a bounded prefix for local smoke and moderate Colab
    runs. The Gold contract keeps `example_id` stable across split, user_state,
    item_features, and targets tables.
    """

    gold_root = Path(gold_root)
    target_set = set(target_classes)
    target_cols = [
        "example_id",
        "user_id",
        "video_id",
        "as_of_time",
        "target_class",
        "is_positive",
        "is_observed_negative",
        "is_ambiguous",
    ]
    user_cols = ["example_id", *config.user_input_features]
    item_cols = ["example_id", *config.item_input_features]

    targets = read_parquet_head(gold_root / "targets" / split, max_examples, target_cols)
    user_state = read_parquet_head(gold_root / "user_state" / split, max_examples, user_cols)
    item_features = read_parquet_head(gold_root / "item_features" / "point_in_time" / split, max_examples, item_cols)

    frame = (
        targets.merge(user_state, on=["example_id", "user_id"], how="inner")
        .merge(item_features, on=["example_id", "video_id"], how="inner")
    )
    if target_set:
        frame = frame[frame["target_class"].isin(target_set)]
    return frame.reset_index(drop=True)


@dataclass(frozen=True)
class EncodedTwoTowerBatch:
    user_categorical: dict[str, torch.Tensor]
    user_numeric: torch.Tensor
    item_categorical: dict[str, torch.Tensor]
    item_numeric: torch.Tensor

    def to(self, device: torch.device | str) -> "EncodedTwoTowerBatch":
        return EncodedTwoTowerBatch(
            user_categorical={key: value.to(device) for key, value in self.user_categorical.items()},
            user_numeric=self.user_numeric.to(device),
            item_categorical={key: value.to(device) for key, value in self.item_categorical.items()},
            item_numeric=self.item_numeric.to(device),
        )


class TwoTowerFrameDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, preprocessor: FeatureBatchPreprocessor) -> None:
        self.frame = frame.reset_index(drop=True)
        self.preprocessor = preprocessor

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict:
        return self.frame.iloc[index].to_dict()


class TwoTowerCollator:
    def __init__(self, preprocessor: FeatureBatchPreprocessor) -> None:
        self.preprocessor = preprocessor

    def __call__(self, rows: list[dict]) -> EncodedTwoTowerBatch:
        frame = pd.DataFrame(rows)
        batch = self.preprocessor.encode_batch(frame, frame)
        return EncodedTwoTowerBatch(
            user_categorical=batch.user_categorical,
            user_numeric=batch.user_numeric,
            item_categorical=batch.item_categorical,
            item_numeric=batch.item_numeric,
        )
