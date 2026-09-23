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
3. Build cleaned silver tables under `data/silver/kuairand/`.
4. Run EDA and downstream preprocessing from Parquet.

## Silver Tables

- `interactions`: full interaction log with normalized timestamps, source table, watch ratio, valid-play flag, and broad positive/negative labels.
- `users`: user profile features, preserving original fields and adding missing indicators for nullable one-hot features.
- `videos_basic`: video metadata, preserving raw fields and adding upload date, duration seconds, aspect ratio, and missing indicators for nullable metadata.
- `videos_statistics`: historical video statistics, preserving all raw counters and adding simple rate features such as play-per-show and like/comment/share rates.

Video metadata and historical statistics stay separate in silver to keep this layer complete but storage-friendly. Task-specific joins and feature selection belong in gold tables.

Raw data and generated datasets are ignored by git. Commit schemas, notebooks, configs, and code only.
