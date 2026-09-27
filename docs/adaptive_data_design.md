# Adaptive Gold Data Design

This document describes the adaptive data layer prepared by:

```text
notebooks/11_colab_adaptive_gold_builder.ipynb
```

The notebook reads the existing Two-Tower Gold dataset:

```text
/content/drive/MyDrive/recsys/data/gold/two_tower/v1_colab/
```

and writes adaptive-ready data to:

```text
/content/drive/MyDrive/recsys/data/gold/adaptive/v1/
```

Silver data is not required.

## Output Layout

```text
data/gold/adaptive/v1/
  manifest.json
  sequence_examples/
    train/
    validation/
    test/
  event_stream/
  serving_queries/
    train/
    validation/
    test/
  model_views/
    v4_sequence_retrieval/
      contract.json
    v5_sequence_ranker/
      contract.json
    v6_online_state/
      contract.json
```

## `sequence_examples`

This is the main physical table for future V4/V5 model notebooks.

Each row is a point-in-time recommendation example:

```text
target event
+ point-in-time user state
+ point-in-time item features
+ recent user history before the target event
```

The recent history fields are arrays, for example:

```text
recent_video_ids
recent_target_classes
recent_engagement_strengths
recent_long_views
recent_likes
recent_hates
recent_clicks
recent_time_ms
```

The notebook also creates session-local arrays:

```text
session_recent_video_ids
session_recent_target_classes
session_recent_engagement_strengths
session_recent_time_ms
```

These features are point-in-time safe: only interactions before the current example are included.

## V4 Contract

`model_views/v4_sequence_retrieval/contract.json` defines the retrieval/candidate-generation view.

Intended use:

```text
recent user sequence
+ long-term user features
+ candidate item features
-> retrieval score
```

This is the natural next version after the V3 Two-Tower baseline. V4 should learn a short-term user representation from recent events instead of relying only on aggregate user state.

## V5 Contract

`model_views/v5_sequence_ranker/contract.json` defines the adaptive ranker view.

V5 can use richer labels and objectives:

```text
STRONG_POSITIVE classification
long_view prediction
engagement_strength regression/ranking
multi-task like/follow/hate heads
```

Current-event outcome fields are labels only, not input features.

## V6 Contract

`model_views/v6_online_state/contract.json` defines the online/adaptive simulation view.

V6 uses:

```text
event_stream/
serving_queries/{split}/
```

`event_stream` is ordered by event time and can be replayed to update an online short-term user state. `serving_queries` contains lightweight request rows linked to `sequence_examples` by `example_id`.

This layer is the bridge from offline sequence modeling to near-real-time adaptive serving.

## Leakage Checks

The notebook writes leakage check results to `manifest.json`.

The key check is:

```text
max(recent_time_ms) < current_time_ms
```

for every row with recent history. Any violation should be treated as a blocking data bug before model training.
