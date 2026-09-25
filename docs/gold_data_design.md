# Gold Data Design

This document describes the current Gold datasets used for recommendation baselines and future model training.

The project keeps two separate Gold families:

- `data/gold/als/`: collaborative-filtering baseline data and Spark ALS artifacts.
- `data/gold/two_tower/`: point-in-time training data for a future adaptive Two-Tower retrieval model.

Generated data is ignored by git. The schemas, code, notebooks, and documentation are versioned; local Parquet artifacts should be regenerated or transferred separately.

## Shared Split Design

Both current Colab-built ALS and Two-Tower artifacts use the same chronological cutoffs:

| Split | Time range |
| --- | --- |
| Train | earliest events before `2022-04-30T07:30:11` |
| Validation | `2022-04-30T07:30:11` to before `2022-05-04T08:45:20` |
| Test | latest events from `2022-05-04T08:45:20` onward |

The exact cutoffs are stored as:

```json
{
  "train_cutoff_time_ms": 1651303811643,
  "validation_cutoff_time_ms": 1651653920917
}
```

The split is chronological, not random. This keeps offline evaluation closer to the real serving setting where future interactions are not available when earlier recommendations are made.

## ALS Baseline Gold

### Purpose

The ALS Gold dataset supports a collaborative-filtering baseline. It answers:

> Given historical user-item interactions, can implicit matrix factorization retrieve future positively engaged items better than popularity?

It is not the final adaptive model, but it gives a useful baseline and produces item/user factors for comparison.

### Current Artifact

Recommended local path:

```text
data/gold/als/v1_colab/
```

Important files/tables:

```text
manifest.json
evaluation_summary.json

train_interactions/
validation_relevance/
test_relevance/

user_mapping/
item_mapping/

train_validation_interactions/
final_user_mapping/
final_item_mapping/

user_factors/
item_factors/
```

### Interaction Strength

ALS consumes one aggregated implicit preference value per `(user_id, video_id)` pair.

The event-level signal is:

```text
0.5 * clip(watch_ratio, 0, 3)
+ 1.0 * long_view
+ 1.5 * is_like
+ 1.5 * is_comment
+ 1.5 * is_forward
+ 2.0 * is_follow
```

Then repeated events for the same user-item pair are aggregated:

```text
interaction_strength = log1p(sum(event_strength))
```

`is_hate` is retained in Silver, but ALS v1 does not encode explicit negative ratings because Spark ALS implicit feedback expects non-negative confidence-style observations.

### ALS Tables

#### `train_interactions/`

Train-only implicit feedback:

```text
user_id
video_id
user_idx
video_idx
interaction_strength
```

`user_idx` and `video_idx` are deterministic integer mappings fit from Train data only.

#### `validation_relevance/` and `test_relevance/`

Future positive ground truth for ranking evaluation. An item is relevant if a future observed interaction contains at least one strong positive signal:

```text
long_view OR is_like OR is_comment OR is_forward OR is_follow
```

These tables preserve warm/cold flags:

```text
is_warm_user
is_warm_item
```

Ranking metrics evaluate warm users/items, while cold-start rates are reported separately.

#### `train_validation_interactions/`

After selecting hyperparameters on Validation, the final model is fit on Train+Validation and evaluated on Test. This table uses mappings fit on Train+Validation.

#### `user_factors/` and `item_factors/`

Spark ALS latent vectors:

```text
user_factors: user_idx, features: array<float>
item_factors: video_idx, features: array<float>
```

These are model artifacts, not raw features.

### Current Baseline Results

From `data/gold/als/v1_colab/evaluation_summary.json`:

| Model | NDCG@10 | Recall@10 | HitRate@10 |
| --- | ---: | ---: | ---: |
| Popularity test | 0.00290 | 0.00442 | 0.04317 |
| ALS final test | 0.01152 | 0.01786 | 0.16327 |

Selected ALS hyperparameters:

```json
{
  "rank": 64,
  "reg_param": 0.05,
  "alpha": 10.0,
  "max_iter": 8
}
```

These absolute metrics are low because the catalog is very large and offline logs only reveal a small subset of what each user might have liked. The important baseline result is that ALS clearly beats popularity under the same protocol.

## Two-Tower Gold

### Purpose

The Two-Tower Gold dataset prepares model-ready retrieval training examples. It does not train embeddings yet.

Each row represents:

```text
(
  user_state_before_t,
  item_at_t,
  observed_reaction_at_t
)
```

The core invariant is point-in-time correctness:

> For an example at time `t`, features may only use information available strictly before `t`. The target may use the current observed reaction at `t`.

### Current Artifact

Recommended local training root:

```text
data/gold/two_tower/v1_ready/
```

This folder is a clean local view of the downloaded Colab artifact. It currently uses symlinks to avoid duplicating large Parquet files that were split across multiple Google Drive zip downloads.

Use this metadata file locally:

```text
data/gold/two_tower/v1_ready/manifest.local.json
```

The original Colab manifest is still available as:

```text
data/gold/two_tower/v1_ready/manifest.json
```

but its `artifact_paths` point to `/content/...` paths from Colab.

### Folder Contract

```text
data/gold/two_tower/v1_ready/
  manifest.local.json
  manifest.json
  feature_catalog.json

  train/
  validation/
  test/

  targets/
    train/
    validation/
    test/

  user_state/
    train/
    validation/
    test/

  item_features/
    static/
    point_in_time/
      train/
      validation/
      test/

  history/
    events/

  vocabularies/
    manifest.json
    user_active_degree/
    video_type/
    upload_type/

  transforms/
    numeric_stats.json
```

### Row Counts

Validated locally with Spark:

| Table | Rows |
| --- | ---: |
| `train/` | 8,219,912 |
| `validation/` | 1,767,868 |
| `test/` | 1,768,293 |
| `targets/train/` | 8,219,912 |
| `targets/validation/` | 1,767,868 |
| `targets/test/` | 1,768,293 |
| `user_state/train/` | 8,219,912 |
| `user_state/validation/` | 1,767,868 |
| `user_state/test/` | 1,768,293 |
| `item_features/point_in_time/train/` | 8,219,912 |
| `item_features/point_in_time/validation/` | 1,767,868 |
| `item_features/point_in_time/test/` | 1,768,293 |
| `item_features/static/` | 4,371,868 |
| `history/events/` | 11,756,073 |

### Core Example Tables

`train/`, `validation/`, and `test/` are lightweight index tables. They contain identifiers and references, not all features:

```text
example_id
context_id
event_id
user_id
video_id
session_id
user_event_index
session_event_index
history_end_user_event_index
history_end_session_event_index
as_of_time
time_ms
split
is_warm_user
is_warm_item
has_user_history
has_item_metadata
```

This keeps core split tables compact and lets the future training Dataset join only the feature groups it needs.

### User-State Tables

`user_state/{split}/` contains features for the User Tower. These are either static user profile features or dynamic point-in-time aggregates computed before the current event.

Feature groups include:

```text
Static user/profile:
  user_active_degree
  is_lowactive_period
  is_live_streamer
  is_video_author
  follow_user_num
  fans_user_num
  friend_user_num
  register_days
  onehot_feat0 ... onehot_feat17

Long-term user history before t:
  user_hist_events
  user_hist_long_view_rate
  user_hist_like_rate
  user_hist_comment_rate
  user_hist_forward_rate
  user_hist_follow_rate
  user_hist_hate_rate
  user_hist_avg_watch_ratio
  user_hist_avg_play_time_sec

Current-session state before t:
  session_event_index
  session_elapsed_sec
  session_prior_long_view_rate
  session_prior_like_rate
  session_prior_hate_rate
  session_prior_avg_watch_ratio

Adaptive drift:
  session_vs_user_long_view_delta
  session_vs_user_watch_ratio_delta

Request/context:
  event_hour
  event_dayofweek
  tab
  is_rand
```

Important leakage rule:

```text
watch_ratio, play_time_ms, long_view, like/comment/follow on the current item are not User Tower features.
```

They are current-event outcomes and are stored under `targets/`.

### Item Feature Tables

#### `item_features/static/`

One row per video metadata record:

```text
video_id
author_id
video_type
upload_type
upload_date
video_duration_sec
aspect_ratio
visible_status
music_type
```

This table is independent of the current user, so it can be used later to precompute item embeddings.

#### `item_features/point_in_time/{split}/`

One row per example with item-side features available at prediction time:

```text
example_id
video_id
as_of_time
video_type
upload_type
video_duration_sec
aspect_ratio
visible_status
music_type
upload_age_days_at_event
item_hist_events
item_hist_long_view_rate
item_hist_like_rate
item_hist_hate_rate
item_hist_avg_watch_ratio
```

`item_hist_*` features are computed from events strictly before the current example time. They are not user-item cross features, so they remain valid for the Item Tower.

### Target Tables

`targets/{split}/` stores current-event outcomes and raw feedback components:

```text
example_id
context_id
user_id
video_id
as_of_time
target_class
is_positive
is_observed_negative
is_ambiguous
engagement_strength
watch_ratio_clipped
long_view
is_like
is_comment
is_forward
is_follow
is_hate
is_click
is_profile_enter
play_time_ms
duration_ms
watch_ratio
```

Target definition:

```text
STRONG_POSITIVE:
  long_view OR is_like OR is_comment OR is_forward OR is_follow

OBSERVED_NEGATIVE:
  not STRONG_POSITIVE
  AND (is_hate = 1 OR clicked watch_ratio <= 0.20)

AMBIGUOUS_WEAK:
  observed event without strong positive or reliable observed negative evidence

UNKNOWN:
  unobserved user-item pairs are not materialized and are never assumed negative
```

Current target distribution:

| Split | Strong positive | Observed negative | Ambiguous/weak |
| --- | ---: | ---: | ---: |
| Train | 2,234,745 | 551,981 | 5,433,186 |
| Validation | 477,290 | 118,510 | 1,172,068 |
| Test | 468,278 | 114,793 | 1,185,222 |

### History Table

`history/events/` stores the chronological event stream with session and event indices:

```text
event_id
user_id
video_id
event_ts
time_ms
session_id
user_event_index
session_event_index
target_class
engagement_strength
direct feedback columns
```

Core example rows contain:

```text
history_end_user_event_index
history_end_session_event_index
```

This allows a future PyTorch Dataset/DataLoader to choose its own history strategy without rebuilding Gold:

- last `N` user events
- last `N` session events
- session-only history
- longer-term history windows
- pooling, GRU, Transformer, or attention encoders

The current item is excluded from its own history by construction.

### Categorical Vocabularies

Vocabularies are fit on Train only:

```text
vocabularies/user_active_degree/
vocabularies/video_type/
vocabularies/upload_type/
```

Current cardinalities excluding OOV:

| Feature | Cardinality |
| --- | ---: |
| `user_active_degree` | 7 |
| `video_type` | 3 |
| `upload_type` | 32 |

Validation/Test unseen categories should map to an OOV/unknown id during model training. This prevents leakage from future splits into the training vocabulary.

### Numerical Transforms

Train-only numerical statistics are stored at:

```text
transforms/numeric_stats.json
```

For each numerical feature, the file stores:

```text
non_null
mean
stddev
min
p01
median
p99
max
```

These stats should be used later for imputation, clipping, and normalization. Validation/Test must reuse Train stats instead of fitting their own.

### Cold-Start Coverage

Cold-start information is preserved rather than dropped.

| Split | Cold user rate | Cold item rate | Has item metadata | Has user history |
| --- | ---: | ---: | ---: | ---: |
| Train | 0.0000 | 0.0000 | 1.0000 | 0.9999 |
| Validation | 0.0025 | 0.6457 | 1.0000 | 1.0000 |
| Test | 0.0037 | 0.7759 | 1.0000 | 1.0000 |

The high cold-item rate is expected for short-video recommendation: many future videos were not present in Train interactions. This is one reason item metadata is important for Two-Tower retrieval.

## Reserved and Dropped Features

Reserved for later ranker or experiments:

```text
tag
user_item_similarity
future ALS/user-item cross scores
```

Dropped from Two-Tower v1:

```text
videos_statistics global counters/rates with unclear snapshot time
current-event watch/play outcomes as input features
```

The main reason is serving-time validity and leakage control. Global video statistics from Silver do not have a guaranteed point-in-time snapshot, and current-event outcomes are only known after the recommendation.

## Recommended Training Reads

For future Two-Tower training, use:

```text
GOLD_ROOT = data/gold/two_tower/v1_ready
```

Typical read pattern:

1. Read core split: `train/`, `validation/`, or `test/`.
2. Join `user_state/{split}` on `example_id` or `context_id`.
3. Join `item_features/point_in_time/{split}` on `example_id`.
4. Join `targets/{split}` on `example_id`.
5. Use `vocabularies/` and `transforms/` for encoding.
6. Use `history/events/` and the history indices if sequence features are needed.

Do not use unobserved user-item pairs as negatives directly. Negative sampling, in-batch negatives, hard negatives, and popularity-aware sampling are training decisions and should be implemented in the model-training pipeline, not baked into Gold v1.
