# Storage Layer

This project uses a small lakehouse-style layout that can run locally now and map to S3, GCS, HDFS, or Delta/Iceberg later.

## Directories

- `data/raw/`: immutable source files, initially KuaiRand CSV files.
- `data/bronze/`: Parquet converted from raw CSV with minimal changes.
- `data/silver/`: cleaned and typed interaction, user, and item tables.
- `data/gold/`: analysis-ready tables such as daily aggregates or session tables.
- `data/features/`: offline feature tables for model training and future online feature sync.

## Current Flow

1. Put KuaiRand CSV files under `data/raw/kuairand/` or keep a local raw directory outside git.
2. Convert CSV to Parquet under `data/bronze/kuairand/`.
3. Run EDA and downstream preprocessing from Parquet.

Raw data and generated datasets are ignored by git. Commit schemas, notebooks, configs, and code only.
