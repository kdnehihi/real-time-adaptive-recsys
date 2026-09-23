from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from recommender.data.build_silver import write_silver_tables  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build KuaiRand silver tables from bronze Parquet.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--bronze-dir",
        type=Path,
        default=PROJECT_ROOT / "data/bronze/kuairand",
        help="Bronze Parquet input directory.",
    )
    parser.add_argument(
        "--silver-dir",
        type=Path,
        default=PROJECT_ROOT / "data/silver/kuairand",
        help="Silver Parquet output directory.",
    )
    parser.add_argument(
        "--tables",
        nargs="+",
        choices=["interactions", "users", "videos_basic", "videos_statistics"],
        help="Optional subset of silver tables to build.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing silver tables.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_silver_tables(
        args.bronze_dir,
        args.silver_dir,
        overwrite=args.overwrite,
        tables_to_build=args.tables,
    )


if __name__ == "__main__":
    main()
