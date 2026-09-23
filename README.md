# Real-Time Adaptive Recommender System

Minimal project foundation for a near-real-time adaptive recommender system using KuaiRand data.

The current scope is data understanding and batch preprocessing. Kafka, streaming jobs, model training, and serving APIs are intentionally left as extension points.

## Project Structure

- `notebooks/`: Jupyter notebooks for exploration. Start with `01_kuairand_data_understanding.ipynb`.
- `recommender/data/`: data ingestion and preprocessing utilities, including CSV to Parquet conversion.
- `recommender/features/`: placeholder for offline and future online feature generation.
- `recommender/models/`: placeholder for training, evaluation, and model artifacts.
- `recommender/streaming/`: placeholder for future streaming consumers/processors.
- `recommender/serving/`: placeholder for future recommendation API code.
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

Recommended local environment: Conda env `recsys` with Python 3.11, Java 17, and PySpark 3.5.6.

```bash
conda create -n recsys -c conda-forge python=3.11 openjdk=17 pip=24.2 -y
conda activate recsys
pip install -r requirements.txt
python -m ipykernel install --user --name recsys --display-name "Python (recsys)"
```

Alternative using the checked-in environment file:

```bash
conda env create -f environment.yml
conda activate recsys
python -m ipykernel install --user --name recsys --display-name "Python (recsys)"
```

Quick verification:

```bash
python -c "import pyspark; print(pyspark.__version__)"
```

Run project scripts from the repository root:

```bash
python scripts/convert_kuairand_to_bronze.py --help
python scripts/build_kuairand_silver.py --help
```

If a notebook Spark cell was already executed before environment or Spark config changes, restart the Jupyter kernel. Spark driver settings are applied only when the JVM starts.

## Convert CSV to Parquet

Put CSV files under `data/raw/`, then run:

```bash
python scripts/convert_kuairand_to_bronze.py \
  --input-dir data/raw \
  --output-dir data/bronze/kuairand \
  --overwrite
```

The EDA notebook only checks whether bronze Parquet already exists and prints this command if it is missing.

## Build Silver Tables

After bronze Parquet exists, build cleaned silver tables:

```bash
python scripts/build_kuairand_silver.py \
  --bronze-dir data/bronze/kuairand \
  --silver-dir data/silver/kuairand \
  --overwrite
```

To rebuild only selected tables:

```bash
python scripts/build_kuairand_silver.py --tables videos_basic videos_statistics --overwrite
```

Outputs:

- `data/silver/kuairand/interactions/`: typed interaction events with event time, watch ratio, and basic labels.
- `data/silver/kuairand/users/`: user features with one-hot missing indicators and filled one-hot values.
- `data/silver/kuairand/videos_basic/`: video metadata with upload date, duration seconds, aspect ratio, and missing indicators.
- `data/silver/kuairand/videos_statistics/`: historical video statistics with all original counters plus basic rate features.
