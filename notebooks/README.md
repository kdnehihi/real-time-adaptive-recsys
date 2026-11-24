# Notebook Guide

The notebooks are organized by pipeline stage. Superseded experiment notebooks are intentionally removed so the active path is unambiguous.

## Data and baselines

1. `01_kuairand_data_understanding.ipynb` - inspect KuaiRand files, schema, quality, and interaction distributions.
2. `02_kuairand_feature_analysis_selection.ipynb` - analyze candidate user, item, and interaction features.
3. `03_kuairand_als_baseline.ipynb` - inspect the local ALS baseline workflow.
4. `04_colab_als_baseline.ipynb` - build and evaluate the full ALS baseline on Colab.

## Two-Tower V3 baseline

1. `04_kuairand_two_tower_data_preparation.ipynb` - understand and prepare Two-Tower Gold data.
2. `05_colab_two_tower_gold_build.ipynb` - materialize Two-Tower Gold data on Colab.
3. `06_two_tower_feature_encoding_walkthrough.ipynb` - inspect feature encoding and embedding tensors.
4. `09_colab_two_tower_v3_final_training.ipynb` - train the selected static Two-Tower V3 model.
5. `10_colab_two_tower_compare.ipynb` - compare static Two-Tower versions and baselines.

The earlier V1 and V2 training notebooks were removed after V3 became the selected static model.

## Adaptive recommender

1. `11_colab_adaptive_gold_builder.ipynb` - build point-in-time adaptive Gold data.
2. `12_colab_v4_adaptive_retrieval_training.ipynb` - train and export the V4 Transformer retrieval checkpoint.
3. `18_colab_adaptive_v4_v5_v6_evaluation.ipynb` - run controlled V4/V5 evaluation and V6 replay.
4. `19_colab_v5_ranker_v2_listwise_training.ipynb` - train the validation-driven residual listwise V5 v2 ranker.
5. `20_colab_adaptive_retrieval_evaluation_audit.ipynb` - audit controlled Top-1000 candidate coverage, bootstrap uncertainty, and cold/warm plus history-length segments before introducing a stronger retrieval algorithm.
6. `21_colab_build_adaptive_eval_protocol_v2.ipynb` - materialize the corrected multi-query exact-target evaluation protocol before re-evaluating retrieval models.
7. `22_colab_evaluate_adaptive_protocol_v2.ipynb` - evaluate Popularity, a controlled static V4 ablation, and adaptive V4 under the same time-eligible, seen-filtered exact-target protocol.
8. `23_colab_adaptive_content_readiness_audit.ipynb` - audit author/content coverage and metadata collisions before training the next cold-aware retrieval model.
9. `24_colab_v4_1_author_retrieval.ipynb` - train a resumable author-aware residual branch over frozen V4 and evaluate it on the locked A2 protocol.
10. `25_colab_v4_2_author_affinity.ipynb` - calibrate a strict-prior recent-author candidate affinity on one user partition and evaluate it on disjoint holdout users.
11. `26_colab_v6_v4_2_online_replay.ipynb` - replay the selected V4.2 policy in global timestamp order and compare it with controlled static/adaptive V4 variants on the locked holdout.
12. `27_colab_v6_true_online_state_replay.ipynb` - rebuild bounded recent/session state event by event, verify exact parity with locked A2 Gold snapshots, and benchmark the state reducer without rerunning exact retrieval.
13. `28_colab_corrected_sequence_parity_audit.ipynb` - compare a corrected timestamp-batch sequence builder with an independent event reducer before rebuilding Gold or retraining V4.
14. `29_colab_build_corrected_sequence_delta.ipynb` - build the accepted corrected train/validation sequence delta, validate full key coverage and strict-prior arrays, then publish the large artifact to private Hugging Face storage.
15. `30_colab_v4_corrected_retraining.ipynb` - retrain the unchanged V4 architecture on corrected sequences with resumable Hugging Face checkpoints, then compare legacy-input, corrected-input, and corrected-retrained V4 under one locked A2 protocol.
16. `31_colab_corrected_v4_author_affinity.ipynb` - calibrate strict-prior recent-author affinity over Corrected V4 on user-disjoint calibration users, then compare it with matched Corrected V4 and Legacy V4.2 on holdout users.
17. `32_colab_rebuild_adaptive_test_protocol.ipynb` - rebuild the incomplete adaptive test split from complete Two-Tower Gold, lock an exact-target test protocol, and publish the large test artifacts directly to private Hugging Face storage.
18. `33_colab_corrected_v4_locked_test_evaluation.ipynb` - reconstruct corrected strict-prior sequences for the locked test queries and make the final no-retuning promotion decision against Legacy V4.2.
19. `34_colab_corrected_v4_3_warm_retention.ipynb` - calibrate a validation-only static/long-term retention branch over Corrected V4.2 to recover warm-item quality without sacrificing corrected cold-item gains.
20. `35_colab_corrected_v4_ann_serving_benchmark.ipynb` - benchmark FAISS HNSW against exact Corrected V4.2 full-policy search, including Top-1000 parity, target quality, latency, index size, and complete-output gates.
21. `36_colab_corrected_v4_exact_serving_bundle.ipynb` - export the selected exact-GPU item catalog, smoke-test the immutable retrieval policy, and publish the compact FP16 storage bundle to private Hugging Face storage.
22. `37_colab_corrected_v4_serving_preprocessing.ipynb` - export train-fitted entity/categorical vocabularies, numeric transforms, and a standalone request schema so the user encoder no longer reads Gold data at runtime.
23. `38_colab_adaptive_serving_api_benchmark.ipynb` - load only private serving artifacts, exercise strict-prior request encoding plus exact retrieval through FastAPI, and report GPU latency, throughput, and memory.

The pointwise V5, standalone V6, and checkpoint-only combined notebooks were removed because notebooks 18 and 19 replace those workflows.

## Current model status

- V3 remains the stable static benchmark.
- V4 with popularity blend `0.02` improves Top-10 quality but not all deeper-ranking metrics.
- The original pointwise V5 checkpoint is rejected.
- V5 v2 is retained as an experiment but is rejected for serving because controlled same-candidate validation is below V4.
- Notebook 20 exposed temporal contamination in the historical multi-positive relevance definition.
- Notebook 21 builds the corrected point-in-time query set; notebook 22 is the active A2 evaluation step.
- A2 confirms that adaptive sequence modeling helps, but cold-item coverage remains the retrieval bottleneck; notebook 23 is the active A3 input-readiness step.
- A3 confirms that author identity is the strongest available cold-start signal, but V4.1's sparse learned author embeddings do not produce a statistically reliable gain and are rejected.
- Notebook 25 selected V4.2 with author weight `0.2`, popularity weight `0.02`, and recency tau `3.0`; it passes the paired holdout HR@1000 acceptance gate and is the current adaptive retrieval choice.
- Notebook 26 is the active production-style replay step. It fixes V4.2 parameters, enforces strict-prior state, reports preference-shift/warm-cold/history segments, and records exact-matrix throughput separately from future ANN latency.
- Notebook 26 confirms both adaptation stages on holdout: V4 adaptive beats the static ablation and V4.2 beats the same-popularity V4 with positive paired HR@1000 confidence intervals. It remains a point-in-time snapshot replay; an event-mutating online state engine is the next serving milestone.
- Notebook 27 is the active online-state milestone. It processes all same-timestamp queries before applying that timestamp's events, uses local disk rather than Drive for its temporary event spool, and treats exact core state parity as a prerequisite for future ANN serving work.
- Notebook 27 exposed a legacy no-backfill bug: strict online state matched only 21.24% of legacy core snapshots because same-timestamp rows were removed after limiting history.
- Notebook 28 passed the corrected contract with exact parity on all 18 audited fields across 7,655 queries. Notebook 29 is now the active data step: it materializes a compact corrected sequence delta without retraining a model.
- Notebook 29 passed all full-data gates for 9,987,780 train/validation examples. Notebook 30 is now the active model step. Corrected V4 is accepted only if its paired user-bootstrap HR@1000 interval is positive against legacy V4 receiving the same corrected inputs; training loss alone is not an acceptance criterion.
- Notebook 30 improved Corrected V4 holdout HR@1000 by about 7.3% and MRR@1000 by about 21.4% over legacy V4 receiving corrected sequences, but the paired HR@1000 interval crossed zero and cold-item HR regressed. Corrected V4 is therefore not promoted by itself. Notebook 31 is the active lightweight recovery experiment: it tunes author/popularity weights on disjoint calibration users and requires both statistically positive holdout gains and cold-item non-regression.
- Notebook 31 selected author weight `0.2` and popularity weight `0.02`. Author affinity significantly improves matched Corrected V4 and restores cold-item retrieval, but the paired HR@1000 interval versus Legacy V4.2 still crosses zero. The branch remains the best experimental candidate rather than the serving default.
- Notebook 32 is the active data step. It rebuilds the incomplete adaptive test split and locks the independent test protocol before any final promotion decision; no test metric is used to tune weights.
- Notebook 32 passed all quality gates for 1,768,293 unique test examples and produced 11,737 locked queries from 983 users with zero timestamp violations.
- Notebook 33 confirms that Corrected V4.2 significantly beats its matched corrected-input base and improves cold-item HR@1000 over Legacy V4.2, but its overall paired interval versus Legacy V4.2 crosses zero because warm-item retrieval regresses. It is not promoted yet.
- Notebook 34 is the active validation-only recovery experiment. It adds a calibrated static/long-term retention score, keeps the locked Test untouched, and requires overall and warm paired gains plus cold non-regression before another final test is justified.
- Notebook 34 selected the zero-weight fallback: static retention did not recover warm retrieval and is rejected without another Test run.
- Notebook 35 is the active serving-readiness step. It freezes Corrected V4.2, uses exact full-policy search as the oracle, and evaluates whether an over-generated HNSW shortlist preserves Top-1000 retrieval after identical author/popularity reranking and point-in-time filtering.
- Notebook 35 rejected HNSW: Top-1000 overlap remained near 76%, target HitRate regressed, some queries returned fewer than 1,000 items, and high-recall settings were slower than the 6.04 ms/query exact GPU oracle.
- Notebook 36 is the active serving step. It packages the exact item matrix in FP16 storage form and verifies full-policy retrieval before publishing the bundle outside Drive.
- Notebook 36 completed and its remote files match the manifest. Notebook 37 is the active contract step: it removes the remaining Gold-data dependency from request-time user encoding.
- Notebook 37 passed its standalone encoder smoke test with checkpoint-compatible vocab sizes. Notebook 38 is the active request-level serving benchmark.
