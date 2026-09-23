from __future__ import annotations

import argparse
from pathlib import Path

from recsys.data.kuairand import read_csv
from recsys.spark import get_spark


def convert_csv_directory(input_dir: Path, output_dir: Path, overwrite: bool = False) -> None:
    spark = get_spark("kuairand-csv-to-parquet")
    csv_files = sorted(input_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")

    mode = "overwrite" if overwrite else "errorifexists"
    output_dir.mkdir(parents=True, exist_ok=True)

    for csv_file in csv_files:
        table_name = csv_file.stem
        target = output_dir / table_name
        df = read_csv(spark, csv_file)
        df.write.mode(mode).parquet(str(target))
        print(f"Wrote {csv_file} -> {target}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert KuaiRand CSV files to Parquet.")
    parser.add_argument("--input-dir", type=Path, default=Path("data/raw/kuairand"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/bronze/kuairand"))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert_csv_directory(args.input_dir, args.output_dir, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
