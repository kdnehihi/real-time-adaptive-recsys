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

### V6 replay implementation

`26_colab_v6_v4_2_online_replay.ipynb` and
`scripts/run_adaptive_v6_v4_2_replay.py` lock the selected V4.2 weights and
replay the user-disjoint A2 holdout in global non-decreasing timestamp order.
Every sequence timestamp is asserted to be strictly earlier than its query.
The replay compares controlled static V4, adaptive V4, adaptive V4 with the
selected popularity bias, and V4.2 under identical eligibility and seen-item
masks. It exports paired user-bootstrap deltas, warm/cold and preference-shift
segments, large rank-improvement examples, and offline exact-matrix
throughput. This throughput is diagnostic only and is not presented as ANN
serving latency.

No V3 inference checkpoint exists in the shared artifact repository, so the
replay records V3 as unavailable instead of mixing historical metrics from a
different protocol. V3 can be added later only after a compatible inference
artifact is exported and evaluated against the same A2 query/candidate set.

### V6 replay result

The replay validated 325,710 recent-event timestamps across 11,758 protocol
queries with zero point-in-time violations, then reported the untouched V4.2
holdout of 7,655 queries from 642 users.

| Replay variant | HR@10 | HR@100 | HR@1000 | NDCG@1000 |
| --- | ---: | ---: | ---: | ---: |
| V4 static ablation | 0.0130% | 0.2077% | 1.4278% | 0.00189 |
| V4 adaptive | 0.0649% | 0.3245% | 2.3650% | 0.00319 |
| V4 adaptive + popularity | 0.0779% | 0.4803% | 2.4039% | 0.00350 |
| **V4.2 author affinity** | **0.1298%** | **0.5582%** | **2.5337%** | **0.00411** |

Adaptive V4 improves HR@1000 over its static ablation by 0.937 percentage
points with paired CI95 `[0.623, 1.251]`. V4.2 then improves HR@1000 over V4
with the same popularity bias by 0.130 points with CI95 `[0.052, 0.221]` and
improves NDCG@1000 by 0.000617 with a fully positive interval. The
recent-preference-shift segment also improves directionally at HR@1000.

The exact-matrix benchmark processed 57.3 replay queries per second while
computing all four variants after item/user encoding; total notebook runtime
was 9.8 minutes. This is an offline throughput measurement, not per-request
production latency.

This experiment is accurately described as timestamp-ordered point-in-time
snapshot replay. Gold supplies each strict-prior state snapshot; the replay
does not yet consume every event and mutate an online state store itself. A
true serving simulation still needs an event-by-event state reducer, state
parity checks against Gold snapshots, ANN/multi-source candidate retrieval,
and request-level latency measurement.

### True online-state reducer implementation

`27_colab_v6_true_online_state_replay.ipynb`,
`scripts/run_adaptive_v6_true_online_state.py`, and
`recommender/models/adaptive/online_state.py` implement the next serving
milestone without repeating the expensive retrieval benchmark.

The reducer keeps a bounded 30-event user deque and 10-event user/session
deque. Events are spooled from train and validation to a local SQLite database
so memory does not grow with full history. Replay uses this exact contract:

1. Order events by event time and the Gold event-ID tie-breaker.
2. At timestamp `t`, answer every selected query from state containing only
   events with time `< t`.
3. Append all events at `t` only after those queries have read state.
4. Compare rebuilt global/session arrays and counts field by field with the
   locked A2 Gold snapshots.

Core array/count parity is the acceptance gate. Scalar elapsed-time and rate
fields are reported as diagnostics because the original Gold builder computed
some of those scalars before removing ambiguous same-timestamp sequence
events. Long-term aggregate state remains outside this reducer and needs a
separate incremental aggregation contract.

Notebook 27 is CPU-only. Its large SQLite spool stays on Colab local disk;
only a small JSON/CSV report bundle is copied to Drive. Passing parity means
the online reducer produces the same V4 sequence tensors as the already
evaluated snapshot replay. It does not by itself measure ANN serving latency.

### True online-state parity result and correction

Notebook 27 replayed 6,676,246 canonicalized train/validation rows for 7,655
holdout queries. The reducer itself sustained about 92,261 state events per
second, but core parity with legacy Gold was only 21.24%, so the serving gate
failed.

The dominant cause is the legacy builder's order of operations. It first took
30 global or 10 session rows using an event-ID order that included other rows
from the current timestamp, then removed current-timestamp rows without
backfilling older valid history. In the first 100 mismatch examples, 89 online
sequences were longer than their legacy counterparts and 76 legacy sequences
were exact suffixes of the corrected online sequence. Global event-count
parity was 25.83%; session event-count parity was 49.68%.

This does not invalidate V4.2's locked legacy-snapshot metrics, but it means a
correct production reducer would feed V4 a different sequence distribution
than it saw during training. Compatibility with the flawed no-backfill rule is
not adopted because it depends on arbitrary within-timestamp ordering and
throws away valid prior events.

`28_colab_corrected_sequence_parity_audit.ipynb` therefore adds a data-only
gate before rebuilding or retraining. Its Spark builder groups events into
timestamp batches, removes the complete current batch, and only then keeps the
latest 30 global and 10 session events. An independent grouped event reducer
reconstructs the same A2 query states. Full Gold materialization and V4
retraining are blocked until those two implementations reach exact core
parity.

### Corrected sequence contract accepted

Notebook 28 passed the corrected data-contract gate with exact parity for all
18 audited sequence fields across 7,655 queries from 642 users. The audit
replayed 6,676,246 canonical train/validation events and found no mismatches
between the Spark timestamp-batch builder and the independent event reducer.
This result accepts the corrected builder implementation; it does not make the
legacy snapshots correct or make old V4 checkpoints compatible with corrected
input sequences.

`29_colab_build_corrected_sequence_delta.ipynb` is the next data step. It
rebuilds only the corrected global/session sequence columns for the existing
train and validation examples, preserving `example_id` as the join key. The
output is a compact versioned delta rather than another complete copy of Gold.
Its acceptance gates require complete key coverage, aligned sequence arrays,
strictly prior timestamps, and the configured 30-event global and 10-event
session limits.

The notebook keeps Spark work and intermediate contexts on Colab local disk,
uploads the large final delta to the private Hugging Face artifact repository
at `corrected_sequences/v2/`, and writes only a small manifest/location record
to Drive. V4 retraining should start only after this manifest reports all
quality gates as passed. Old V4/V4.2 metrics remain historical legacy-contract
results and must not be directly compared with a corrected-contract model
until both are evaluated under one locked protocol.

Notebook 29 completed that materialization for 8,219,912 train and 1,767,868
validation examples. All 9,987,780 example IDs were unique and covered, and
the reported alignment, strict-prior timestamp, and sequence-cap violations
were zero.

### Corrected V4 retraining contract

`30_colab_v4_corrected_retraining.ipynb` keeps the selected V4 Transformer,
feature encoders, in-batch negatives, popularity-weighted negatives, random
negatives, and primary optimization settings fixed. Base point-in-time
features are joined to the corrected sequence delta by `example_id` using an
out-of-core DuckDB preparation step. This makes legacy V4 to corrected V4 a
data-contract experiment rather than an architecture change.

Training writes FP16 weight/history checkpoints to private Hugging Face
storage every two epochs. They intentionally omit AdamW state, which would
add several GB; a resumed job restores weights and history but restarts the
optimizer. A real-data smoke run must pass before the full job begins.

The controlled A2 comparison reports:

1. legacy V4 with its legacy sequence input;
2. legacy V4 with corrected sequence input;
3. corrected-retrained V4 with corrected sequence input.

Every variant uses identical exact targets, candidate eligibility, seen-item
masks, popularity weight, and query set. The input-shift delta separates the
effect of changing sequence values from the retraining delta. Corrected V4 is
accepted only when the user-clustered paired HR@1000 confidence interval for
variant 3 versus variant 2 is positive. Train loss and in-batch validation
metrics are diagnostics and do not replace this retrieval gate.

Notebook 30 selected epoch 3. On the locked A2 comparison, corrected retraining
improved HR@1000 and MRR@1000 over legacy V4 receiving the same corrected
sequence input, but the paired HR@1000 interval still crossed zero. The gain
was concentrated in warm items and long-history users while cold-item HR@1000
regressed. The checkpoint remains an experimental input to the next controlled
branch rather than the serving default.

### Corrected V4 author-affinity calibration

`31_colab_corrected_v4_author_affinity.ipynb` evaluates whether the inexpensive
V4.2 strict-prior author signal complements the corrected checkpoint. It does
not retrain the Transformer. Users are deterministically separated into a
calibration partition and a holdout partition, so no user's queries can tune a
weight and evaluate that same weight. Calibration searches author-affinity and
popularity weights; the selected pair is then frozen.

The holdout comparison uses one A2 query set, candidate catalog, target,
time-eligibility mask, and prior-seen mask for all variants. It reports:

1. raw Corrected V4;
2. Corrected V4 with the selected popularity weight but no author affinity;
3. Corrected V4 with the selected author and popularity weights;
4. the historical Legacy V4.2 policy with author `0.2`, popularity `0.02`, and
   recency tau `3.0`.

The branch is promoted only when its user-clustered paired HR@1000 confidence
interval is strictly positive against both the matched-popularity Corrected V4
and Legacy V4.2, and cold-item HR@1000 does not regress against the matched
Corrected V4 baseline. This prevents a warm-author gain from hiding a further
cold-start loss.

Notebook 31 selected author weight `0.2` with popularity weight `0.02`. Against
the same-popularity Corrected V4, holdout HR@1000 improved by about 6.1% with a
strictly positive paired confidence interval. Cold HR@1000 also recovered from
`0.000143` to `0.001886`. Against Legacy V4.2, the point estimate improved by
about 7.7%, but the paired interval still crossed zero. The corrected author
policy is therefore retained as the best experimental candidate but is not yet
the serving default.

### Independent adaptive test rebuild

`32_colab_rebuild_adaptive_test_protocol.ipynb` repairs the incomplete physical
test split before final evaluation. It reconstructs `sequence_examples/test`
from the complete Two-Tower test targets, user state, item snapshots, and the
materialized legacy sequence context. Quality gates require complete example
coverage, unique keys, complete user/item joins, positive examples, and zero
same/future sequence timestamps.

The notebook then builds a deterministic exact-target test protocol using the
same A2 revision as validation. Existing train data is read only to label warm
and cold targets; it is not copied. Large test artifacts are published under
`adaptive_test/v1/` in private Hugging Face storage, while Drive receives only
small manifests and logs. The fixed Notebook 31 weights must remain unchanged
in the subsequent test evaluation.
