from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from recommender.data.convert_csv_to_parquet import convert_csv_directory  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert local KuaiRand CSV files to bronze Parquet.")
    parser.add_argument("--input-dir", type=Path, default=PROJECT_ROOT / "data/raw/kuairand")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data/bronze/kuairand")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert_csv_directory(args.input_dir, args.output_dir, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
