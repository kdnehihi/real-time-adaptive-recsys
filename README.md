# Real-Time Adaptive Recommender System

Minimal project foundation for a near-real-time adaptive recommender system using KuaiRand data.

The current scope is data understanding and batch preprocessing. Kafka, streaming jobs, model training, and serving APIs are intentionally left as extension points.

## Project Structure

- `notebooks/`: Jupyter notebooks for exploration. Start with `01_kuairand_data_understanding.ipynb`.
- `src/recsys/data/`: data ingestion and preprocessing utilities, including CSV to Parquet conversion.
- `src/recsys/features/`: placeholder for offline and future online feature generation.
- `src/recsys/models/`: placeholder for training, evaluation, and model artifacts.
- `src/recsys/streaming/`: placeholder for future streaming consumers/processors.
- `src/recsys/serving/`: placeholder for future recommendation API code.
- `configs/`: example local configuration.
- `docs/`: project notes, including the storage layer design.
- `data/`: local raw/processed/feature data. This directory is ignored by git except `.gitkeep` placeholders.
- `tests/`: test suite placeholder.

## Storage Layout

- `data/raw/`: raw KuaiRand CSV files.
- `data/bronze/`: Parquet converted from raw CSV.
- `data/silver/`: cleaned and typed intermediate tables.
- `data/gold/`: analysis-ready aggregate/session tables.
- `data/features/`: feature tables for model training and future online sync.

See `docs/storage_layer.md` for more detail.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Convert CSV to Parquet

Put CSV files under `data/raw/kuairand/`, then run:

```bash
PYTHONPATH=src python -m recsys.data.convert_csv_to_parquet \
  --input-dir data/raw/kuairand \
  --output-dir data/bronze/kuairand \
  --overwrite
```

The EDA notebook can also convert any local KuaiRand CSV directory to Parquet before analysis.
