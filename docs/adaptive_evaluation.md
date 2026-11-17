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
