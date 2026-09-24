from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from pyspark.sql import functions as F

from recommender.gold.als_baseline import (
    DEFAULT_WEIGHTS,
    FORMULA_VERSION,
    add_event_strength,
    apply_mappings,
    build_mappings,
    build_relevance,
    build_train_interactions,
    cold_start_report,
    git_commit,
    read_interactions,
    split_events,
    split_summary,
    tune_strength_weights,
    write_json,
    write_parquet,
)
from recommender.spark import get_spark


def build_gold_dataset(args: argparse.Namespace) -> dict:
    spark = get_spark("kuairand-build-als-gold", reset=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw_events = read_interactions(spark, args.silver_dir)
    split, split_meta = split_events(raw_events, args.train_quantile, args.validation_quantile)
    split = split.cache()
    weighted = add_event_strength(split, DEFAULT_WEIGHTS).cache()
    train_events = weighted.where(F.col("split") == "train").cache()
    train_original = build_train_interactions(train_events).cache()
    user_mapping, item_mapping = build_mappings(train_original)
    user_mapping = user_mapping.cache()
    item_mapping = item_mapping.cache()
    train_interactions = apply_mappings(train_original, user_mapping, item_mapping).cache()

    validation_relevance = build_relevance(weighted, user_mapping, item_mapping, "validation", DEFAULT_WEIGHTS).cache()
    selected_weights = tune_strength_weights(
        split.where(F.col("split") == "train"),
        validation_relevance,
        user_mapping,
        item_mapping,
        args.strength_tuning_trials,
        args.k,
    )
    if selected_weights != DEFAULT_WEIGHTS:
        weighted = add_event_strength(split, selected_weights).cache()
        train_events = weighted.where(F.col("split") == "train").cache()
        train_original = build_train_interactions(train_events).cache()
        user_mapping, item_mapping = build_mappings(train_original)
        user_mapping = user_mapping.cache()
        item_mapping = item_mapping.cache()
        train_interactions = apply_mappings(train_original, user_mapping, item_mapping).cache()
        validation_relevance = build_relevance(weighted, user_mapping, item_mapping, "validation", selected_weights).cache()
    test_relevance = build_relevance(weighted, user_mapping, item_mapping, "test", selected_weights).cache()

    train_validation_events = weighted.where(F.col("split").isin("train", "validation")).cache()
    train_validation_original = build_train_interactions(train_validation_events).cache()
    final_user_mapping, final_item_mapping = build_mappings(train_validation_original)
    final_user_mapping = final_user_mapping.cache()
    final_item_mapping = final_item_mapping.cache()
    train_validation_interactions = apply_mappings(
        train_validation_original, final_user_mapping, final_item_mapping
    ).cache()
    final_test_relevance = build_relevance(
        weighted, final_user_mapping, final_item_mapping, "test", selected_weights
    ).cache()

    write_parquet(
        train_interactions.select("user_idx", "video_idx", "interaction_strength", "user_id", "video_id"),
        args.output_dir / "train_interactions",
        args.overwrite,
    )
    write_parquet(validation_relevance, args.output_dir / "validation_relevance", args.overwrite)
    write_parquet(test_relevance, args.output_dir / "test_relevance", args.overwrite)
    write_parquet(user_mapping, args.output_dir / "user_mapping", args.overwrite)
    write_parquet(item_mapping, args.output_dir / "item_mapping", args.overwrite)
    write_parquet(
        train_validation_interactions.select("user_idx", "video_idx", "interaction_strength", "user_id", "video_id"),
        args.output_dir / "train_validation_interactions",
        args.overwrite,
    )
    write_parquet(final_user_mapping, args.output_dir / "final_user_mapping", args.overwrite)
    write_parquet(final_item_mapping, args.output_dir / "final_item_mapping", args.overwrite)
    write_parquet(final_test_relevance, args.output_dir / "final_test_relevance", args.overwrite)

    manifest = {
        "gold_dataset_version": args.version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_silver_path": str(args.silver_dir),
        "output_path": str(args.output_dir),
        "code_commit": git_commit(),
        "interaction_strength": {
            "formula_version": FORMULA_VERSION,
            "formula": "interaction_strength = log1p(sum(0.5*clip(watch_ratio,0,3)+1.0*long_view+1.5*like+1.5*comment+1.5*forward+2.0*follow))",
            "weights": selected_weights,
            "strength_tuning_trials": args.strength_tuning_trials,
            "negative_feedback_policy": "is_hate is retained in silver but not encoded as negative ALS rating in v1",
        },
        "temporal_split": split_meta,
        "split_summary": split_summary(split),
        "mapping_statistics_train_only": {
            "users": user_mapping.count(),
            "items": item_mapping.count(),
            "train_user_item_rows": train_interactions.count(),
        },
        "mapping_statistics_train_validation": {
            "users": final_user_mapping.count(),
            "items": final_item_mapping.count(),
            "train_validation_user_item_rows": train_validation_interactions.count(),
        },
        "cold_start": {
            "validation": cold_start_report(validation_relevance),
            "test": cold_start_report(test_relevance),
            "final_test_after_train_validation": cold_start_report(final_test_relevance),
        },
        "evaluation_protocol": {
            "split": "event-level chronological split before user-item aggregation",
            "k": args.k,
            "train_history_filter": "recommendations exclude items seen in the corresponding training history",
            "metrics": [f"ndcg@{args.k}", f"recall@{args.k}", f"hitrate@{args.k}"],
            "relevance": "binary relevant if any future long_view/like/comment/forward/follow; graded relevance from positive engagement weights",
            "cold_start": "ranking metrics evaluate warm users and warm items; cold rates are reported separately",
        },
        "artifact_paths": {
            "train_interactions": str(args.output_dir / "train_interactions"),
            "validation_relevance": str(args.output_dir / "validation_relevance"),
            "test_relevance": str(args.output_dir / "test_relevance"),
            "user_mapping": str(args.output_dir / "user_mapping"),
            "item_mapping": str(args.output_dir / "item_mapping"),
            "train_validation_interactions": str(args.output_dir / "train_validation_interactions"),
            "final_user_mapping": str(args.output_dir / "final_user_mapping"),
            "final_item_mapping": str(args.output_dir / "final_item_mapping"),
            "final_test_relevance": str(args.output_dir / "final_test_relevance"),
        },
    }
    write_json(manifest, args.output_dir / "manifest.json")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    spark.stop()
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build KuaiRand ALS gold datasets from full silver interactions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--silver-dir", type=Path, default=Path("data/silver/kuairand"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/gold/als/v1"))
    parser.add_argument("--version", default="als_v1")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--train-quantile", type=float, default=0.70)
    parser.add_argument("--validation-quantile", type=float, default=0.85)
    parser.add_argument("--strength-tuning-trials", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    build_gold_dataset(parse_args())


if __name__ == "__main__":
    main()
