from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import mlflow
from pyspark.sql import functions as F

from recommender.gold.als_baseline import (
    AlsParams,
    als_recommendations,
    parse_float_list,
    parse_int_list,
    popularity_recommendations,
    ranking_metrics,
    train_als,
    write_json,
    write_parquet,
)
from recommender.spark import get_spark


def log_metrics(prefix: str, metrics: dict) -> None:
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            mlflow.log_metric(f"{prefix}_{key.replace('@', '_at_')}", float(value))


def load_manifest(gold_dir: Path) -> dict:
    path = gold_dir / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing manifest: {path}. Run scripts/build_kuairand_als_gold.py first.")
    return json.loads(path.read_text())


def train_baseline(args: argparse.Namespace) -> dict:
    spark = get_spark("kuairand-train-als-baseline", reset=True)
    manifest = load_manifest(args.gold_dir)
    args.tracking_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(args.tracking_dir.resolve().as_uri())
    mlflow.set_experiment(args.experiment_name)

    train = spark.read.parquet(str(args.gold_dir / "train_interactions")).cache()
    validation_relevance = spark.read.parquet(str(args.gold_dir / "validation_relevance")).cache()
    test_relevance = spark.read.parquet(str(args.gold_dir / "test_relevance")).cache()
    train_validation = spark.read.parquet(str(args.gold_dir / "train_validation_interactions")).cache()
    final_test_relevance = spark.read.parquet(str(args.gold_dir / "final_test_relevance")).cache()

    train_history = train.select("user_idx", "video_idx").cache()
    train_validation_history = train_validation.select("user_idx", "video_idx").cache()
    validation_users = validation_relevance.where(F.col("is_warm_user")).select("user_idx").distinct()
    test_users = test_relevance.where(F.col("is_warm_user")).select("user_idx").distinct()
    final_test_users = final_test_relevance.where(F.col("is_warm_user")).select("user_idx").distinct()

    grid = [
        AlsParams(rank=rank, reg_param=reg, alpha=alpha, max_iter=args.max_iter)
        for rank in args.ranks
        for reg in args.reg_params
        for alpha in args.alphas
    ]
    if args.max_grid_models:
        grid = grid[: args.max_grid_models]
    if not grid:
        raise ValueError("ALS grid is empty")

    with mlflow.start_run(run_name=args.run_name) as run:
        mlflow.log_params(
            {
                "gold_dir": str(args.gold_dir),
                "k": args.k,
                "max_iter": args.max_iter,
                "als_over_generate": args.als_over_generate,
                "formula_version": manifest["interaction_strength"]["formula_version"],
                "gold_dataset_version": manifest["gold_dataset_version"],
            }
        )
        for name, weight in manifest["interaction_strength"]["weights"].items():
            mlflow.log_param(f"strength_weight_{name}", weight)

        pop_val_recs = popularity_recommendations(train, validation_users, train_history, args.k).cache()
        pop_test_recs = popularity_recommendations(train, test_users, train_history, args.k).cache()
        pop_validation = ranking_metrics(pop_val_recs, validation_relevance, k=args.k)
        pop_test = ranking_metrics(pop_test_recs, test_relevance, k=args.k)
        log_metrics("popularity_validation", pop_validation)
        log_metrics("popularity_test", pop_test)

        best: tuple[float, AlsParams] | None = None
        als_results = []
        for idx, params in enumerate(grid):
            with mlflow.start_run(run_name=f"als_grid_{idx}", nested=True):
                mlflow.log_params(
                    {
                        "rank": params.rank,
                        "reg_param": params.reg_param,
                        "alpha": params.alpha,
                        "max_iter": params.max_iter,
                    }
                )
                model = train_als(train, params)
                recs = als_recommendations(
                    model, validation_users, train_history, args.k, over_generate=args.als_over_generate
                ).cache()
                metrics = ranking_metrics(recs, validation_relevance, k=args.k)
                log_metrics("validation", metrics)
                result = {"params": vars(params), "validation": metrics}
                als_results.append(result)
                score = metrics[f"ndcg@{args.k}"]
                mlflow.log_metric("validation_impact_ndcg_vs_popularity", score - pop_validation[f"ndcg@{args.k}"])
                if best is None or score > best[0]:
                    best = (score, params)
                recs.unpersist()

        assert best is not None
        selected_params = best[1]
        mlflow.log_params(
            {
                "selected_rank": selected_params.rank,
                "selected_reg_param": selected_params.reg_param,
                "selected_alpha": selected_params.alpha,
                "selected_max_iter": selected_params.max_iter,
            }
        )

        final_model = train_als(train_validation, selected_params)
        final_recs = als_recommendations(
            final_model, final_test_users, train_validation_history, args.k, over_generate=args.als_over_generate
        ).cache()
        final_test = ranking_metrics(final_recs, final_test_relevance, k=args.k)
        log_metrics("final_als_test", final_test)
        mlflow.log_metric("final_test_impact_ndcg_vs_popularity", final_test[f"ndcg@{args.k}"] - pop_test[f"ndcg@{args.k}"])
        mlflow.log_metric("final_test_impact_recall_vs_popularity", final_test[f"recall@{args.k}"] - pop_test[f"recall@{args.k}"])
        mlflow.log_metric("final_test_impact_hitrate_vs_popularity", final_test[f"hitrate@{args.k}"] - pop_test[f"hitrate@{args.k}"])

        write_parquet(final_model.userFactors.withColumnRenamed("id", "user_idx"), args.gold_dir / "user_factors", args.overwrite)
        write_parquet(final_model.itemFactors.withColumnRenamed("id", "video_idx"), args.gold_dir / "item_factors", args.overwrite)

        evaluation_summary = {
            "mlflow_run_id": run.info.run_id,
            "mlflow_tracking_uri": mlflow.get_tracking_uri(),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "popularity_validation": pop_validation,
            "popularity_test": pop_test,
            "als_grid_results": als_results,
            "selected_als_hyperparameters": vars(selected_params),
            "final_als_test": final_test,
            "impact_vs_popularity_test": {
                f"ndcg@{args.k}": final_test[f"ndcg@{args.k}"] - pop_test[f"ndcg@{args.k}"],
                f"recall@{args.k}": final_test[f"recall@{args.k}"] - pop_test[f"recall@{args.k}"],
                f"hitrate@{args.k}": final_test[f"hitrate@{args.k}"] - pop_test[f"hitrate@{args.k}"],
            },
        }
        write_json(evaluation_summary, args.gold_dir / "evaluation_summary.json")

        manifest.update(
            {
                "trained_at": evaluation_summary["generated_at"],
                "mlflow": {
                    "tracking_uri": mlflow.get_tracking_uri(),
                    "experiment_name": args.experiment_name,
                    "run_id": run.info.run_id,
                },
                "popularity_baseline": {"validation": pop_validation, "test": pop_test},
                "als_grid_results": als_results,
                "selected_als_hyperparameters": vars(selected_params),
                "final_als_test_metrics": final_test,
                "impact_vs_popularity_test": evaluation_summary["impact_vs_popularity_test"],
                "artifact_paths": {
                    **manifest.get("artifact_paths", {}),
                    "user_factors": str(args.gold_dir / "user_factors"),
                    "item_factors": str(args.gold_dir / "item_factors"),
                },
            }
        )
        write_json(manifest, args.gold_dir / "manifest.json")
        mlflow.log_artifact(str(args.gold_dir / "manifest.json"))
        mlflow.log_artifact(str(args.gold_dir / "evaluation_summary.json"))
        print(json.dumps(evaluation_summary, indent=2, sort_keys=True))

    spark.stop()
    return evaluation_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train popularity and Spark ALS baselines from an existing KuaiRand ALS gold dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--gold-dir", type=Path, default=Path("data/gold/als/v1"))
    parser.add_argument("--tracking-dir", type=Path, default=Path("data/mlruns"))
    parser.add_argument("--experiment-name", default="kuairand_als_baseline")
    parser.add_argument("--run-name", default="als_baseline_v1")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--ranks", type=parse_int_list, default=[32, 64])
    parser.add_argument("--reg-params", type=parse_float_list, default=[0.05, 0.1])
    parser.add_argument("--alphas", type=parse_float_list, default=[10.0, 20.0])
    parser.add_argument("--max-iter", type=int, default=8)
    parser.add_argument("--max-grid-models", type=int, default=0)
    parser.add_argument("--als-over-generate", type=int, default=300)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    train_baseline(parse_args())


if __name__ == "__main__":
    main()
