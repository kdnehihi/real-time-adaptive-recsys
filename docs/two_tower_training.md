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

## Colab V2 Run

V2 is intended to improve the first neural baseline with two changes:

```text
full_item_vocab_from_train_items: true
negative_sample_count: 4096
```

This means `video_id` embeddings are allocated from all train item features, not only positive train pairs, and each training batch scores the positive batch items plus extra sampled item negatives.

Run the self-contained notebook:

```text
notebooks/08_colab_two_tower_v2_training.ipynb
```

or use the config:

```bash
python scripts/train_two_tower_baseline.py \
  --config configs/two_tower_colab_v2.json
```

The default V2 artifact path is:

```text
/content/drive/MyDrive/recsys/artifacts/two_tower/baseline_v2/
```

V2 uses `batch_size=2048` because sampled negatives increase the softmax candidate matrix size. If memory is tight, reduce `negative_sample_count` to `2048` before reducing model size.

## Colab V3 Final Run

V3 is the strongest current baseline configuration. It keeps the V2 full item vocabulary and sampled-negative setup, then makes the negatives harder:

```text
negative_sample_count: 8192
negative_sampling_strategy: mixed
popular_negative_fraction: 0.7
item_popularity_column: item_hist_events
item_popularity_alpha: 0.75
temperature: 0.05
MLP hidden dims: [768, 384]
```

The intuition is that random negatives are often too easy in a huge short-video catalog. Popularity-weighted negatives are more competitive because they are videos the system is more likely to consider/recommend.

Run:

```text
notebooks/09_colab_two_tower_v3_final_training.ipynb
```

or:

```bash
python scripts/train_two_tower_baseline.py \
  --config configs/two_tower_colab_v3.json
```

Artifacts are written to:

```text
/content/drive/MyDrive/recsys/artifacts/two_tower/baseline_v3/
```

If memory is tight, reduce `negative_sample_count` from `8192` to `4096`, then reduce `batch_size` from `2048` to `1024`.

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

The Colab notebook also computes a candidate-based validation protocol after training:

```text
validation user query
-> score observed validation candidate items
-> rank candidates
-> compute HitRate/Recall/NDCG@K against that user's validation STRONG_POSITIVE items
```

These metrics are saved as `candidate_validation` in `metrics.json` and `two_tower_candidate_validation` in `comparison_summary.json`. They are much more useful for directional comparison with ALS/popularity than in-batch metrics, although they still use the observed validation candidate set rather than a production full-catalog ANN index.

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
