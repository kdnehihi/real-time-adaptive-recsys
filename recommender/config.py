from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class DataPaths:
    root: Path
    raw: Path
    bronze: Path
    silver: Path
    gold: Path
    features: Path


def data_paths() -> DataPaths:
    root = Path(os.getenv("RECSYS_DATA_ROOT", project_root() / "data"))
    return DataPaths(
        root=root,
        raw=Path(os.getenv("RECSYS_RAW_DIR", root / "raw")),
        bronze=Path(os.getenv("RECSYS_BRONZE_DIR", root / "bronze")),
        silver=Path(os.getenv("RECSYS_SILVER_DIR", root / "silver")),
        gold=Path(os.getenv("RECSYS_GOLD_DIR", root / "gold")),
        features=Path(os.getenv("RECSYS_FEATURE_DIR", root / "features")),
    )
