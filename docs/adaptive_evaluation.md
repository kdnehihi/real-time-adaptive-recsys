# Adaptive Retrieval and Ranking Evaluation

## Data status

The adaptive train and validation splits are complete and use point-in-time
sequence sanitation. The adaptive test split is incomplete and must not be
used for final reporting until it is rebuilt successfully.

`11_colab_adaptive_gold_builder.ipynb` filters every recent-event array so that
sequence timestamps are strictly earlier than the target `as_of_time`. The
model encoder also handles valid all-padding batches by falling back to a zero
short-term representation instead of invoking Transformer attention on an
empty sequence.

## Controlled validation setup

The current audit uses:

- validation split only;
- 984 users, with the latest strong-positive event as each user's query;
- 854,893 validation candidate items;
- exact matrix retrieval;
- point-in-time sanitized user sequences;
- Popularity, raw V4, and V4 with popularity blend `0.02`;
- Top-K metrics through Top-1000;
- 1,000 user/query bootstrap samples for 95% confidence intervals.

The catalog contains every exact query target (`target_catalog_coverage=1.0`),
while only 29.88% of validation targets are warm relative to train.

## Retrieval audit result

### Multi-positive metrics

| Variant | HitRate@10 | NDCG@10 | HitRate@1000 | Recall@1000 |
| --- | ---: | ---: | ---: | ---: |
| Popularity | 0.0793 | 0.0099 | 0.9085 | 0.0227 |
| V4 raw | 0.1778 | 0.0242 | 0.8648 | 0.0259 |
| V4 + popularity 0.02 | **0.2520** | **0.0373** | 0.9065 | **0.0335** |

The NDCG@10 bootstrap intervals do not overlap:

- Popularity: `[0.0076, 0.0125]`
- V4 raw: `[0.0203, 0.0287]`
- V4 + popularity 0.02: `[0.0324, 0.0426]`

These numbers show that V4 learns broad/repeat preference better than the
popularity baseline. They are not yet valid final next-item metrics because the
multi-positive relevance set contains positive interactions that occurred
before the selected latest query.

### Exact-next retrieval

| Variant | Exact HR@10 | Exact HR@100 | Exact HR@500 | Exact HR@1000 |
| --- | ---: | ---: | ---: | ---: |
| Popularity | 0.0010 | 0.0030 | 0.0061 | 0.0122 |
| V4 raw | 0.0000 | 0.0010 | 0.0152 | **0.0285** |
| V4 + popularity 0.02 | 0.0000 | **0.0051** | 0.0132 | 0.0234 |

The exact target exists in the candidate catalog for every query, but V4 raw
retrieves it into Top-1000 for only 28 of 984 queries. Popularity blending
moves a few exact targets closer to the top while reducing total exact-target
coverage at Top-1000. This is a retrieval bottleneck that a second-stage ranker
cannot repair.

All selected latest-user queries have at least 20 prior events, so this audit
does not yet identify behavior for empty, short, or medium histories.

## V5 v2 result

V5 v2 is a residual listwise ranker selected by internal validation NDCG@10.
Its best checkpoint is epoch 1 and is numerically valid, but it is rejected by
the controlled same-candidate evaluation:

| Metric | V4 ordering | V5 v2 ordering | Relative change |
| --- | ---: | ---: | ---: |
| HitRate@10 | 0.2520 | 0.1748 | -30.6% |
| NDCG@10 | 0.0373 | 0.0216 | -42.0% |
| HitRate@50 | 0.5163 | 0.4553 | -11.8% |
| NDCG@100 | 0.0269 | 0.0208 | -22.8% |

Only 1.22% of train targets appeared naturally in V4 Top-128 candidates. Most
training positives were therefore forced into sampled candidate groups, which
created a train-serving distribution mismatch. V5 v2 must not be used for
serving or V6 conclusions.

## Decision

- Keep V3 as the static historical benchmark.
- Keep raw V4 and V4 blend `0.02` as adaptive retrieval references.
- Do not deploy either existing V5 ranker.
- Do not optimize another ranker until exact candidate coverage is improved.
- Treat current multi-positive metrics as broad/repeat-interest diagnostics,
  not final next-item quality.

## Required next protocol

The next evaluation must sample multiple point-in-time queries across each
user timeline instead of only the latest event:

1. Sequence events must be strictly earlier than query time.
2. Primary relevance is the exact positive at the query time.
3. Optional multi-positive relevance may include only a declared future window,
   never events before the query.
4. Previously seen items are filtered from recommendations, except when the
   current target is intentionally a repeated item.
5. Queries are stratified by empty, short, medium, and long history as well as
   warm/cold target status.
6. Confidence intervals use clustered bootstrap by user.

If exact Recall@1000 remains low under this corrected protocol, the next model
should be a multi-interest, content-aware retrieval model rather than another
second-stage ranker.

### Step A1 implementation

`21_colab_build_adaptive_eval_protocol_v2.ipynb` and
`scripts/build_adaptive_eval_protocol_v2.py` materialize this protocol as a
reusable Drive artifact. The builder selects up to 12 deterministic queries per
user, prioritizes history and early/middle/late timeline strata, labels
warm/cold and recent-repeat targets, sanitizes sequence timestamps, and stores
exact-target relevance metadata. It does not train or evaluate a model.

The raw session sequence is capped at ten events. Downstream evaluation uses
the explicit buckets `empty`, `short_1_2`, `medium_3_5`, and `long_6_10` rather
than treating the capped length as a user's total lifetime history.

### Step A2 implementation

`22_colab_evaluate_adaptive_protocol_v2.ipynb` and
`scripts/evaluate_adaptive_protocol_v2.py` perform the first corrected
retrieval comparison. They use:

- the fixed A1 query IDs and exact current target;
- candidate upload-time eligibility at each query;
- dynamic item statistics frozen at the train cutoff, with zero history for
  cold items;
- all strict-prior train and validation events for seen-item filtering;
- an exception that keeps an intentionally repeated exact target eligible;
- user-clustered bootstrap confidence intervals;
- Popularity, V4 with its recent sequence removed, raw adaptive V4, and the
  existing V4 popularity blend under identical masks.

The static V4 ablation is the primary controlled measurement of short-term
sequence value because it holds model weights and long-term state constant.
The historical V3 checkpoint remains a reference and can be added to the same
protocol once its inference artifact is available in the shared artifact
store.

### Step A2 result

The corrected protocol evaluated 11,758 queries from 984 users against 854,893
time-eligible candidates. V4 adaptive improved exact HR@1000 from 1.482% for
the controlled static ablation to 2.356%. The paired user-bootstrap delta was
0.874 percentage points with a 95% interval of `[0.620, 1.137]`, confirming
that recent-sequence modeling adds useful retrieval signal.

The remaining bottleneck is cold retrieval:

- V4 adaptive warm-target HR@1000: 5.174%;
- V4 adaptive cold-target HR@1000: 0.367%;
- cold targets account for 61.1% of protocol queries.

The global popularity blend reached 2.500% overall HR@1000, but reduced cold
HR@1000 to 0.090%. Its paired improvement over raw V4 was not conclusive, so it
is not the serving default. A second-stage ranker remains blocked until target
coverage in the retrieval candidate set improves.

### Step A3 implementation

`23_colab_adaptive_content_readiness_audit.ipynb` and
`scripts/audit_adaptive_content_readiness.py` inspect the item inputs needed to
fix cold retrieval before another expensive training run. The audit measures:

- missingness and cardinality of the current static item fields;
- content-signature collision sizes with and without `author_id`;
- the share of cold targets whose author is known from train;
- strict point-in-time target-author exposure in each user's prior history;
- semantic fields still absent from Gold, including tag, text, audio, and
  visual embeddings.

This step is CPU-only. Its output decides whether an author-aware branch is
already viable or whether richer content features must be materialized first.

### Step A3 result and V4.1

Author identity is available for every protocol target. Among cold targets,
80.88% belong to an author observed in train and 44.53% have a strict-prior
interaction from the same user with that author. Adding author identity to the
basic metadata signature reduces median cold-target collision size from 16,522
videos to 2. Basic metadata alone is therefore insufficient, while an author
branch is justified.

`24_colab_v4_1_author_retrieval.ipynb` trains V4.1 as an ANN-compatible
residual over frozen V4:

- a train-only shared author embedding with OOV index zero;
- recent video IDs mapped to strict-prior author IDs;
- recency-weighted author preference pooling;
- a learned history-count gate and residual scale;
- concatenated V4/author user and item vectors whose dot product equals the
  sum of the V4 and author scores.

The full V4 model stays frozen for the first controlled experiment. Training
checkpoints are written after every epoch and can resume after a Colab runtime
failure. The final evaluator reuses the exact A2 eligibility, seen filtering,
warm/cold segmentation, and clustered user bootstrap.

### V4.1 result and V4.2 decision

V4.1 selected epoch 1, after which its internal validation loss rose while
train loss continued to fall. On A2, the pure author branch tied V4 at HR@10,
improved HR@50 by only three exact hits, and reduced HR@1000 from 2.356% to
2.288%. All paired user-bootstrap intervals for V4.1 minus V4 crossed zero.
The same conclusion holds after controlling for the identical popularity
blend: author plus popularity reduced HR@1000 by 0.085 percentage points and
its interval also crossed zero. V4.1 is therefore rejected for serving.

`25_colab_v4_2_author_affinity.ipynb` tests a less sparse cold-start strategy.
It adds a recency-weighted exact-author affinity between each strict-prior
sequence and candidate metadata, without learning 1.1 million author
embeddings. Hyperparameters are selected on a deterministic calibration-user
partition and reported once on disjoint holdout users. The acceptance gate is
a positive paired HR@1000 delta whose clustered 95% lower bound is above zero
against V4 with the same popularity weight. This sparse affinity can later be
implemented as a recent-author candidate source alongside ANN retrieval.

### V4.2 result

V4.2 passed its predeclared acceptance gate on 7,655 holdout queries from 642
users, disjoint from the 342 calibration users. The selected configuration is
`author_weight=0.2`, `popularity_weight=0.02`, and `recency_tau=3.0`.

| Holdout variant | HR@10 | HR@100 | HR@1000 | NDCG@1000 |
| --- | ---: | ---: | ---: | ---: |
| V4 raw | 0.0649% | 0.3245% | 2.3650% | 0.00319 |
| V4 + popularity | 0.0779% | 0.4803% | 2.4039% | 0.00350 |
| **V4.2 author affinity + popularity** | **0.1298%** | **0.5582%** | **2.5337%** | **0.00411** |

Against V4 with the same popularity weight, V4.2 improved HR@1000 by 0.130
percentage points with paired user-bootstrap CI95 `[0.039, 0.221]` percentage
points. NDCG@1000 improved by 0.000617 with CI95 `[0.000136, 0.001207]`.
The gain is strongest for intentionally repeated targets, but cold-target
HR@1000 also improves significantly relative to the same-popularity baseline.

V4.2 is now the selected adaptive retrieval variant. In production it should
be implemented as a union of V4 ANN candidates and candidates from recent
authors, followed by the selected score biases. It does not solve semantic
retrieval for unseen authors: cold HR@1000 remains below raw V4 when compared
without the popularity control, so richer text/tag/audio/visual item features
remain the next major retrieval improvement.
