import numpy as np

from scripts.benchmark_corrected_v4_ann import (
    overlap_recall,
    select_ann_config,
    target_hits,
)


def test_overlap_recall_is_query_level_set_recall() -> None:
    reference = np.asarray([[1, 2, 3], [4, 5, 6]])
    candidate = np.asarray([[3, 2, 9], [8, 7, 6]])
    np.testing.assert_allclose(overlap_recall(reference, candidate), [2 / 3, 1 / 3])


def test_target_hits_uses_candidate_indices() -> None:
    top = np.asarray([[1, 2, 3], [4, 5, 6]])
    np.testing.assert_array_equal(target_hits(top, np.asarray([2, 9])), [True, False])


def test_select_ann_config_uses_fastest_eligible() -> None:
    rows = [
        {"ef_search": 64, "mean_topk_overlap_recall": 0.94, "target_hitrate_delta_vs_exact": 0.0, "queries_with_fewer_than_k": 0, "search_rerank_ms_per_query": 1.0},
        {"ef_search": 128, "mean_topk_overlap_recall": 0.97, "target_hitrate_delta_vs_exact": -0.0005, "queries_with_fewer_than_k": 0, "search_rerank_ms_per_query": 2.0},
        {"ef_search": 256, "mean_topk_overlap_recall": 0.99, "target_hitrate_delta_vs_exact": 0.0, "queries_with_fewer_than_k": 0, "search_rerank_ms_per_query": 4.0},
    ]
    assert select_ann_config(rows, 0.95, 0.001)["ef_search"] == 128


def test_select_ann_config_rejects_short_results() -> None:
    rows = [
        {"ef_search": 128, "mean_topk_overlap_recall": 0.99, "target_hitrate_delta_vs_exact": 0.0, "queries_with_fewer_than_k": 1, "search_rerank_ms_per_query": 2.0},
    ]
    assert select_ann_config(rows, 0.95, 0.001) is None
