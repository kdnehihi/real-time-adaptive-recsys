from __future__ import annotations

import argparse
from pathlib import Path

from recommender.data.kuairand import read_csv
from recommender.spark import get_spark


def convert_csv_directory(input_dir: Path, output_dir: Path, overwrite: bool = False) -> None:
    csv_files = sorted(input_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")

    spark = get_spark("kuairand-csv-to-parquet")
    mode = "overwrite" if overwrite else "errorifexists"
    output_dir.mkdir(parents=True, exist_ok=True)

    for csv_file in csv_files:
        table_name = csv_file.stem
        target = output_dir / table_name
        df = read_csv(spark, csv_file)
        df.write.mode(mode).parquet(str(target))
        print(f"Wrote {csv_file} -> {target}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert KuaiRand CSV files to Parquet.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-dir", type=Path, default=Path("data/raw"), help="Directory containing raw CSV files.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/bronze/kuairand"), help="Bronze Parquet output directory.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing bronze tables.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert_csv_directory(args.input_dir, args.output_dir, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
