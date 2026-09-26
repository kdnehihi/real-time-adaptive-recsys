# Two-Tower Feature Encoding

This document describes the current feature representation layer for the KuaiRand Two-Tower retrieval model.

The layer converts raw Gold features into dense tensors:

```text
raw Gold features
-> categorical/OOV ids + normalized numerics
-> trainable feature embeddings
-> concatenated feature representation
-> future User/Item Tower MLP
-> future final retrieval embedding
```

This layer is not the final retrieval model. It does not train a loss, sample negatives, or produce ANN-ready embeddings.

## Gold Input

Use the local Gold root:

```text
data/gold/two_tower/v1_ready/
```

The encoder reads existing train-fitted metadata:

```text
vocabularies/user_active_degree/
vocabularies/video_type/
vocabularies/upload_type/
transforms/numeric_stats.json
```

Validation and test preprocessing must reuse these train-fitted vocabularies and train numerical statistics.

## User Encoder Features

Categorical/entity features:

```text
user_id
user_active_degree
```

Numerical features:

```text
is_lowactive_period
is_live_streamer
is_video_author
follow_user_num
fans_user_num
friend_user_num
register_days
onehot_feat0 ... onehot_feat17
user_hist_events
user_hist_long_view_rate
user_hist_like_rate
user_hist_comment_rate
user_hist_forward_rate
user_hist_follow_rate
user_hist_hate_rate
user_hist_avg_watch_ratio
user_hist_avg_play_time_sec
session_event_index
session_elapsed_sec
session_prior_long_view_rate
session_prior_like_rate
session_prior_hate_rate
session_prior_avg_watch_ratio
session_vs_user_long_view_delta
session_vs_user_watch_ratio_delta
event_hour
event_dayofweek
tab
is_rand
```

The User encoder has a placeholder-friendly shape: a future history encoder can append a sequence representation before the future User Tower MLP.

## Item Encoder Features

Categorical/entity features:

```text
video_id
video_type
upload_type
```

Numerical features:

```text
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

The Item encoder does not depend exclusively on `video_id`. Cold/unseen videos map to the `video_id` OOV row but can still produce a vector from metadata and item numerical features.

## Embeddings And OOV

Each categorical/entity feature owns its own PyTorch embedding table.

Default dimensions:

| Feature | Default dim |
| --- | ---: |
| `user_active_degree` | 4 |
| `video_type` | 3 |
| `upload_type` | 8 |
| `user_id` | 8 |
| `video_id` | 16 |
| `author_id` | 8 if enabled later |

Index `0` is reserved for OOV/unknown values. Existing Gold vocabularies start at index `1`, so unseen validation/test categories naturally map to `0`.

`author_id` is supported by the config pattern but disabled by default because there is no current train-fitted `author_id` vocabulary in Gold v1.

## Large ID Memory

Inspected train cardinalities:

| ID | Train cardinality | Default dim | Approx fp32 memory |
| --- | ---: | ---: | ---: |
| `user_id` | 994 | 8 | < 0.1 MB |
| `video_id` | 3,215,507 | 16 | ~196 MB |

`video_id` dim 32 would be about 393 MB, so dim 16 is the conservative default.

## Numerical Preprocessing

Numerical preprocessing uses `transforms/numeric_stats.json`, fit on Train only:

1. Cast to float.
2. Impute missing values using train mean by default.
3. Clip to train p01/p99 bounds.
4. Normalize with train mean/std.
5. Replace residual NaN/Inf with finite values.

Validation/test must never fit new vocabularies or new numerical statistics.

## Prohibited Inputs

Current-event outcomes and target fields must never be encoder inputs:

```text
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

They can be displayed in notebooks or used as labels, but they are blocked from model input feature groups.
