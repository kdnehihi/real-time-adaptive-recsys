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

The pointwise V5, standalone V6, and checkpoint-only combined notebooks were removed because notebooks 18 and 19 replace those workflows.

## Current model status

- V3 remains the stable static benchmark.
- V4 with popularity blend `0.02` improves Top-10 quality but not all deeper-ranking metrics.
- The original pointwise V5 checkpoint is rejected.
- V5 v2 is the active ranker experiment and must pass the controlled same-candidate validation before use in V6.
- The adaptive test split must be rebuilt before final locked test reporting.
