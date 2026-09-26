# Two-Tower Training Pipeline

This document describes the first PyTorch training pipeline for the Two-Tower retrieval baseline.

The local pipeline uses only Gold data:

```text
data/gold/two_tower/v1_ready/
```

Silver data is not needed unless you want to rebuild Gold.

## Current Training Flow

```text
Gold split tables
-> joined pandas frame for a bounded prefix
-> FeatureBatchPreprocessor
-> UserFeatureEncoder + ItemFeatureEncoder
-> small User MLP + Item MLP
-> L2-normalized retrieval embeddings
-> in-batch dot-product logits
-> cross-entropy loss
```

This is a baseline retrieval model. It does not yet use sequence encoders, hard negatives, ANN indexing, or full-catalog evaluation.

## Local Smoke Run

Run:

```bash
python scripts/train_two_tower_baseline.py \
  --config configs/two_tower_smoke.json
```

Outputs:

```text
data/models/two_tower/smoke_local/
  config.json
  metrics.json
  comparison_summary.json
  model.pt
  vocab_summary.json
```

The smoke config uses:

```text
max_train_examples: 50,000 prefix rows
target: STRONG_POSITIVE only
batch_size: 512
epochs: 1
retrieval_dim: 32
MLP hidden dims: [64]
MLflow: enabled, file tracking under data/mlruns
```

Because it filters to `STRONG_POSITIVE`, the actual train rows are fewer than the loaded prefix.

## Colab Baseline Run

Use:

```bash
python scripts/train_two_tower_baseline.py \
  --config configs/two_tower_colab_baseline.json
```

The Colab config assumes your Drive artifact is stored at:

```text
/content/drive/MyDrive/recsys/data/gold/two_tower/v1_colab/
```

and writes artifacts to:

```text
/content/drive/MyDrive/recsys/artifacts/two_tower/baseline_v1/
```

MLflow is disabled for Colab by default. Metrics and artifacts are always written to disk.

The full Colab config uses all available positive Train/Validation examples:

```text
max_train_examples: null
max_validation_examples: null
batch_size: 4096
epochs: 10
retrieval_dim: 128
MLP hidden dims: [512, 256]
early stopping: validation_ndcg@50, patience 3
```

It writes both `model.pt` and `best_model.pt`.

## Metrics

The current Two-Tower metrics are in-batch retrieval metrics:

```text
MRR
HitRate@K
Recall@K
NDCG@K
```

Each validation batch forms a small candidate set where the diagonal item is the positive item for each user. This is useful for comparing Two-Tower training runs, checking that the model learns, and observing regressions.

These metrics are not yet directly equivalent to the Spark ALS/popularity metrics, which evaluate top-K recommendations over a larger item candidate set.

## Resource Estimate

Local smoke:

```text
CPU: enough
RAM: 8-16 GB
GPU: not required
Disk: < 2 GB extra artifacts
```

Colab baseline:

```text
GPU: recommended, T4 is acceptable
System RAM: 25-50 GB is better
GPU RAM: 15 GB T4 should be OK with batch_size 4096 for the current MLP, but reduce to 2048 if memory spikes
Disk/Drive free space: 10-30 GB
```

For the full config, use a High-RAM runtime if possible. If Colab RAM is not enough, reduce `max_train_examples` to `2_000_000` and `max_validation_examples` to `300_000`, then rerun. The first likely bottleneck is Parquet-to-pandas loading, not the MLP itself.

## Current Limitations

- Training reads a bounded prefix into pandas. This is deliberate for the first baseline and Colab simplicity.
- Evaluation is in-batch, not full-catalog.
- `AMBIGUOUS_WEAK` and `OBSERVED_NEGATIVE` are not used in the training loss yet.
- History sequences are not modeled yet; only point-in-time aggregate history features are used.
