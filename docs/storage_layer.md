# Storage Layer

This project uses a small lakehouse-style layout that can run locally now and map to S3, GCS, HDFS, or Delta/Iceberg later.

## Directories

- `data/raw/`: immutable source files, initially KuaiRand CSV files.
- `data/bronze/`: Parquet converted from raw CSV with minimal changes.
- `data/silver/`: cleaned and typed interaction, user, and item tables.
- `data/gold/`: analysis-ready tables such as daily aggregates or session tables.
- `data/features/`: offline feature tables for model training and future online feature sync.

## Current Flow

1. Put KuaiRand CSV files under `data/raw/` or keep a local raw directory outside git.
2. Convert CSV to Parquet under `data/bronze/kuairand/` with `python scripts/convert_kuairand_to_bronze.py --overwrite`.
3. Build cleaned silver tables under `data/silver/kuairand/` with `python scripts/build_kuairand_silver.py --overwrite`.
4. Run EDA and downstream preprocessing from Parquet.

## Silver Tables

- `interactions`: full interaction log with normalized timestamps, source table, watch ratio, valid-play flag, and broad positive/negative labels.
- `users`: user profile features, preserving original fields and adding missing indicators for nullable one-hot features.
- `videos_basic`: video metadata, preserving raw fields and adding upload date, duration seconds, aspect ratio, and missing indicators for nullable metadata.
- `videos_statistics`: historical video statistics, preserving all raw counters and adding simple rate features such as play-per-show and like/comment/share rates.

Video metadata and historical statistics stay separate in silver to keep this layer complete but storage-friendly. Task-specific joins and feature selection belong in gold tables.

## ALS Gold Tables

`data/gold/als/v1/` contains the collaborative-filtering baseline artifacts. Build it with `python scripts/build_kuairand_als_gold.py --overwrite`, then train/evaluate with `python scripts/train_kuairand_als_baseline.py --overwrite`.

- `train_interactions`: train-only `(user_idx, video_idx, interaction_strength)` for implicit ALS.
- `validation_relevance` and `test_relevance`: future strong-engagement labels for ranking evaluation.
- `user_mapping` and `item_mapping`: deterministic train-entity ID mappings.
- `train_validation_interactions`: final Train+Validation interactions used for the final ALS model.
- `final_user_mapping` and `final_item_mapping`: mappings for the final Train+Validation model.
- `user_factors` and `item_factors`: Spark ALS latent vectors.
- `manifest.json`: split cutoffs, formula, metrics, cold-start coverage, selected ALS parameters, and artifact paths.

MLflow experiment runs are stored locally in `data/mlruns/` unless another tracking directory is passed.

Raw data and generated datasets are ignored by git. Commit schemas, notebooks, configs, and code only.
