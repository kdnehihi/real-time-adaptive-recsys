# Adaptive Retrieval and Ranking Evaluation

## Validation protocol

The controlled V4/V5 evaluation uses:

- validation split only;
- 984 users, with the latest strong-positive event as each user's query state;
- 854,893 validation candidate items;
- exact matrix retrieval;
- V4 Top-1000 candidates shared unchanged with V5;
- all strong-positive validation items for user-level multi-positive relevance.

The adaptive test split is incomplete and is not used. Final test reporting must wait until that split is rebuilt successfully.

## Current result

V4 benefits from a small popularity blend, but the gain is concentrated at the top of the list. With blend weight `0.02`, V4 reaches:

| Metric | V3 historical reference | V4 blend 0.02 | Delta |
| --- | ---: | ---: | ---: |
| HitRate@10 | 0.2439 | 0.2510 | +0.0071 |
| HitRate@50 | 0.5356 | 0.5132 | -0.0224 |
| HitRate@100 | 0.6606 | 0.6291 | -0.0315 |
| NDCG@10 | 0.0351 | 0.0370 | +0.0019 |
| NDCG@50 | 0.0306 | 0.0297 | -0.0009 |
| NDCG@100 | 0.0281 | 0.0268 | -0.0013 |

The V3 values are a historical candidate-validation reference. They have the same user and candidate counts, but should not be presented as a strict production A/B result until both checkpoints are evaluated by the same executable.

V5 is not ready for serving. On the exact same V4 candidates, it decreases every multi-positive ranking metric. For example:

| Metric | V4 raw order | V5 reranked | Delta |
| --- | ---: | ---: | ---: |
| HitRate@10 | 0.1789 | 0.0854 | -0.0935 |
| HitRate@50 | 0.4096 | 0.2825 | -0.1270 |
| HitRate@100 | 0.5356 | 0.4116 | -0.1240 |
| NDCG@10 | 0.0242 | 0.0092 | -0.0150 |

The likely cause is the current V5 training design: sampled pointwise BCE, checkpoint selection by training loss, and no validation ranking early stopping. Falling training loss therefore did not imply better candidate ordering.

## Decision

- Keep V3 as the stable benchmark.
- Keep V4 as an experimental adaptive retriever; use blend `0.02` only when Top-10 quality is the product priority.
- Do not deploy the current V5 checkpoint.
- Treat the first V6 report as a pipeline smoke test, not a representative adaptation result: its original bounded reader selected 1,000 early rows containing only 13 users.

The V6 evaluator now samples users and their timelines deterministically to avoid that bias and reports distinct `evaluated_queries` and `evaluated_users`.

## Next implementation

1. Train V5 v2 with candidate groups produced by V4 and a ranking-aware objective.
2. Use validation NDCG@10 or NDCG@50 for checkpoint selection and early stopping.
3. Include harder negatives near the top of V4's ordering, while retaining some popularity and random negatives.
4. Calibrate or normalize the V4 score before feeding it to V5.
5. Re-run the controlled same-candidate comparison; accept V5 only when `ranker_delta_vs_v4` is positive on validation.
6. Re-run balanced V6 replay for adaptation segments.
7. Rebuild the adaptive test split and run one final locked test evaluation.
