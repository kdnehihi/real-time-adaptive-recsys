from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from pyspark.ml.recommendation import ALS
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from recommender.spark import get_spark


DEFAULT_WEIGHTS = {
    "watch_ratio": 0.5,
    "long_view": 1.0,
    "is_like": 1.5,
    "is_comment": 1.5,
    "is_forward": 1.5,
    "is_follow": 2.0,
}
FORMULA_VERSION = "implicit_strength_v1"


@dataclass(frozen=True)
class AlsParams:
    rank: int = 64
    reg_param: float = 0.05
    alpha: float = 20.0
    max_iter: int = 8


def parse_float_list(value: str) -> list[float]:
    return [float(v.strip()) for v in value.split(",") if v.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def read_interactions(spark: SparkSession, silver_dir: Path) -> DataFrame:
    path = silver_dir / "interactions"
    if not path.exists():
        raise FileNotFoundError(f"Missing silver interactions table: {path}")
    df = spark.read.parquet(str(path))
    required = {
        "user_id",
        "video_id",
        "event_ts",
        "time_ms",
        "watch_ratio",
        "long_view",
        "is_like",
        "is_comment",
        "is_forward",
        "is_follow",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Silver interactions is missing required columns: {missing}")
    return df


def split_events(events: DataFrame, train_q: float = 0.70, validation_q: float = 0.85) -> tuple[DataFrame, dict]:
    cut_train, cut_validation = events.approxQuantile("time_ms", [train_q, validation_q], 0.001)
    split = (
        events.where(F.col("event_ts").isNotNull())
        .withColumn(
            "split",
            F.when(F.col("time_ms") < F.lit(cut_train), F.lit("train"))
            .when(F.col("time_ms") < F.lit(cut_validation), F.lit("validation"))
            .otherwise(F.lit("test")),
        )
    )
    split_stats = {
        "train_cutoff_time_ms": int(cut_train),
        "validation_cutoff_time_ms": int(cut_validation),
        "train_cutoff_ts": split.where(F.col("time_ms") >= cut_train).agg(F.min("event_ts")).first()[0].isoformat(),
        "validation_cutoff_ts": split.where(F.col("time_ms") >= cut_validation).agg(F.min("event_ts")).first()[0].isoformat(),
    }
    return split, split_stats


def add_event_strength(events: DataFrame, weights: dict[str, float]) -> DataFrame:
    clipped_watch_ratio = F.least(F.greatest(F.coalesce(F.col("watch_ratio"), F.lit(0.0)), F.lit(0.0)), F.lit(3.0))
    strength = F.lit(weights["watch_ratio"]) * clipped_watch_ratio
    for col in ["long_view", "is_like", "is_comment", "is_forward", "is_follow"]:
        strength = strength + F.lit(weights[col]) * F.coalesce(F.col(col), F.lit(0)).cast("double")
    return events.withColumn("event_strength", F.greatest(strength, F.lit(0.0)))


def strong_relevance_expr() -> F.Column:
    return (
        (F.coalesce(F.col("long_view"), F.lit(0)) == 1)
        | (F.coalesce(F.col("is_like"), F.lit(0)) == 1)
        | (F.coalesce(F.col("is_comment"), F.lit(0)) == 1)
        | (F.coalesce(F.col("is_forward"), F.lit(0)) == 1)
        | (F.coalesce(F.col("is_follow"), F.lit(0)) == 1)
    )


def graded_relevance_expr(weights: dict[str, float]) -> F.Column:
    return (
        F.lit(weights["long_view"]) * F.coalesce(F.col("long_view"), F.lit(0)).cast("double")
        + F.lit(weights["is_like"]) * F.coalesce(F.col("is_like"), F.lit(0)).cast("double")
        + F.lit(weights["is_comment"]) * F.coalesce(F.col("is_comment"), F.lit(0)).cast("double")
        + F.lit(weights["is_forward"]) * F.coalesce(F.col("is_forward"), F.lit(0)).cast("double")
        + F.lit(weights["is_follow"]) * F.coalesce(F.col("is_follow"), F.lit(0)).cast("double")
    )


def split_summary(events: DataFrame) -> list[dict]:
    rows = (
        events.groupBy("split")
        .agg(
            F.count("*").alias("events"),
            F.countDistinct("user_id").alias("users"),
            F.countDistinct("video_id").alias("items"),
            F.min("event_ts").alias("min_ts"),
            F.max("event_ts").alias("max_ts"),
        )
        .orderBy("split")
        .collect()
    )
    return [
        {
            "split": r["split"],
            "events": int(r["events"]),
            "users": int(r["users"]),
            "items": int(r["items"]),
            "min_ts": r["min_ts"].isoformat() if r["min_ts"] else None,
            "max_ts": r["max_ts"].isoformat() if r["max_ts"] else None,
        }
        for r in rows
    ]


def build_train_interactions(train_events: DataFrame) -> DataFrame:
    return (
        train_events.groupBy("user_id", "video_id")
        .agg(F.log1p(F.sum("event_strength")).alias("interaction_strength"))
        .where(F.col("interaction_strength") > 0)
    )


def build_mappings(train_interactions: DataFrame) -> tuple[DataFrame, DataFrame]:
    user_window = Window.orderBy("user_id")
    item_window = Window.orderBy("video_id")
    users = train_interactions.select("user_id").distinct().withColumn("user_idx", F.row_number().over(user_window) - 1)
    items = train_interactions.select("video_id").distinct().withColumn("video_idx", F.row_number().over(item_window) - 1)
    return users, items


def apply_mappings(interactions: DataFrame, users: DataFrame, items: DataFrame) -> DataFrame:
    return (
        interactions.join(F.broadcast(users), on="user_id", how="inner")
        .join(items, on="video_id", how="inner")
        .select("user_id", "video_id", "user_idx", "video_idx", "interaction_strength")
    )


def build_relevance(events: DataFrame, users: DataFrame, items: DataFrame, split: str, weights: dict[str, float]) -> DataFrame:
    relevant = (
        events.where(F.col("split") == split)
        .where(strong_relevance_expr())
        .withColumn("graded_relevance_event", graded_relevance_expr(weights))
        .groupBy("user_id", "video_id")
        .agg(
            F.lit(1).alias("relevant"),
            F.max("graded_relevance_event").alias("graded_relevance"),
            F.count("*").alias("relevant_events"),
        )
    )
    return (
        relevant.join(F.broadcast(users), on="user_id", how="left")
        .join(items, on="video_id", how="left")
        .withColumn("is_warm_user", F.col("user_idx").isNotNull())
        .withColumn("is_warm_item", F.col("video_idx").isNotNull())
    )


def cold_start_report(relevance: DataFrame) -> dict:
    total = relevance.count()
    if total == 0:
        return {"rows": 0, "warm_user_rate": 0.0, "warm_item_rate": 0.0, "cold_user_rate": 0.0, "cold_item_rate": 0.0}
    row = relevance.agg(
        F.avg(F.col("is_warm_user").cast("double")).alias("warm_user_rate"),
        F.avg(F.col("is_warm_item").cast("double")).alias("warm_item_rate"),
        F.countDistinct("user_id").alias("users"),
        F.countDistinct("video_id").alias("items"),
    ).first()
    return {
        "rows": int(total),
        "users": int(row["users"]),
        "items": int(row["items"]),
        "warm_user_rate": float(row["warm_user_rate"] or 0.0),
        "warm_item_rate": float(row["warm_item_rate"] or 0.0),
        "cold_user_rate": float(1.0 - (row["warm_user_rate"] or 0.0)),
        "cold_item_rate": float(1.0 - (row["warm_item_rate"] or 0.0)),
    }


def ranking_metrics(recommendations: DataFrame, relevance: DataFrame, k: int = 10) -> dict:
    rel = relevance.where(F.col("is_warm_user") & F.col("is_warm_item")).select(
        "user_idx", "video_idx", "graded_relevance"
    )
    rel_by_user = rel.groupBy("user_idx").agg(F.countDistinct("video_idx").alias("num_relevant"))
    evaluated_users = rel_by_user.count()
    if evaluated_users == 0:
        return {"evaluated_users": 0, f"recall@{k}": 0.0, f"ndcg@{k}": 0.0, f"hitrate@{k}": 0.0}

    joined = (
        recommendations.where(F.col("rank") <= k)
        .join(rel, on=["user_idx", "video_idx"], how="left")
        .withColumn("is_hit", F.col("graded_relevance").isNotNull().cast("int"))
        .withColumn("gain", F.coalesce(F.col("graded_relevance"), F.lit(0.0)))
        .withColumn("dcg_term", F.col("gain") / (F.log2(F.col("rank") + F.lit(1.0))))
    )
    per_user_dcg = joined.groupBy("user_idx").agg(
        F.sum("is_hit").alias("hits"),
        F.sum("dcg_term").alias("dcg"),
        F.countDistinct("video_idx").alias("recommended_items"),
    )

    ideal_window = Window.partitionBy("user_idx").orderBy(F.desc("graded_relevance"), F.asc("video_idx"))
    ideal = (
        rel.withColumn("ideal_rank", F.row_number().over(ideal_window))
        .where(F.col("ideal_rank") <= k)
        .withColumn("idcg_term", F.col("graded_relevance") / F.log2(F.col("ideal_rank") + F.lit(1.0)))
        .groupBy("user_idx")
        .agg(F.sum("idcg_term").alias("idcg"))
    )
    metrics = (
        rel_by_user.join(per_user_dcg, on="user_idx", how="left")
        .join(ideal, on="user_idx", how="left")
        .fillna({"hits": 0.0, "dcg": 0.0, "recommended_items": 0.0, "idcg": 0.0})
        .withColumn("recall", F.col("hits") / F.least(F.col("num_relevant"), F.lit(k)))
        .withColumn("hitrate", (F.col("hits") > 0).cast("double"))
        .withColumn("ndcg", F.when(F.col("idcg") > 0, F.col("dcg") / F.col("idcg")).otherwise(F.lit(0.0)))
    )
    row = metrics.agg(
        F.count("*").alias("evaluated_users"),
        F.avg("recall").alias("recall"),
        F.avg("ndcg").alias("ndcg"),
        F.avg("hitrate").alias("hitrate"),
        F.avg("recommended_items").alias("avg_recommended_items"),
    ).first()
    return {
        "evaluated_users": int(row["evaluated_users"]),
        f"recall@{k}": float(row["recall"] or 0.0),
        f"ndcg@{k}": float(row["ndcg"] or 0.0),
        f"hitrate@{k}": float(row["hitrate"] or 0.0),
        "avg_recommended_items": float(row["avg_recommended_items"] or 0.0),
    }


def popularity_recommendations(
    train_interactions: DataFrame,
    users_to_eval: DataFrame,
    train_history: DataFrame,
    k: int,
    candidate_multiplier: int = 300,
) -> DataFrame:
    top_n = max(k * candidate_multiplier, 1000)
    popular = (
        train_interactions.groupBy("video_idx")
        .agg(F.sum("interaction_strength").alias("popularity_score"))
        .orderBy(F.desc("popularity_score"), F.asc("video_idx"))
        .limit(top_n)
    )
    candidates = users_to_eval.select("user_idx").distinct().crossJoin(F.broadcast(popular))
    filtered = candidates.join(train_history.select("user_idx", "video_idx"), on=["user_idx", "video_idx"], how="left_anti")
    window = Window.partitionBy("user_idx").orderBy(F.desc("popularity_score"), F.asc("video_idx"))
    return filtered.withColumn("rank", F.row_number().over(window)).where(F.col("rank") <= k)


def als_recommendations(model, users_to_eval: DataFrame, train_history: DataFrame, k: int, over_generate: int = 300) -> DataFrame:
    raw = model.recommendForUserSubset(users_to_eval.select("user_idx").distinct(), max(k, over_generate))
    exploded = raw.select("user_idx", F.posexplode("recommendations").alias("pos", "rec")).select(
        "user_idx",
        F.col("rec.video_idx").alias("video_idx"),
        F.col("rec.rating").alias("score"),
    )
    filtered = exploded.join(train_history.select("user_idx", "video_idx"), on=["user_idx", "video_idx"], how="left_anti")
    window = Window.partitionBy("user_idx").orderBy(F.desc("score"), F.asc("video_idx"))
    return filtered.withColumn("rank", F.row_number().over(window)).where(F.col("rank") <= k)


def train_als(train: DataFrame, params: AlsParams):
    als = ALS(
        userCol="user_idx",
        itemCol="video_idx",
        ratingCol="interaction_strength",
        implicitPrefs=True,
        coldStartStrategy="drop",
        nonnegative=False,
        rank=params.rank,
        regParam=params.reg_param,
        alpha=params.alpha,
        maxIter=params.max_iter,
        seed=42,
    )
    return als.fit(train.select("user_idx", "video_idx", "interaction_strength"))


def tune_strength_weights(
    train_events: DataFrame,
    validation_relevance: DataFrame,
    users: DataFrame,
    items: DataFrame,
    trials: int,
    k: int,
) -> dict[str, float]:
    if trials <= 0:
        return DEFAULT_WEIGHTS

    def evaluate(weights: dict[str, float]) -> float:
        weighted = add_event_strength(train_events, weights)
        train_interactions = apply_mappings(build_train_interactions(weighted), users, items).cache()
        recs = popularity_recommendations(
            train_interactions,
            validation_relevance.where(F.col("is_warm_user")).select("user_idx").distinct(),
            train_interactions.select("user_idx", "video_idx"),
            k=k,
        )
        score = ranking_metrics(recs, validation_relevance, k=k)[f"ndcg@{k}"]
        train_interactions.unpersist()
        return score

    try:
        import optuna  # type: ignore

        def objective(trial):
            weights = {
                "watch_ratio": trial.suggest_float("watch_ratio", 0.1, 1.0),
                "long_view": trial.suggest_float("long_view", 0.5, 2.0),
                "is_like": trial.suggest_float("is_like", 0.5, 3.0),
                "is_comment": trial.suggest_float("is_comment", 0.5, 3.0),
                "is_forward": trial.suggest_float("is_forward", 0.5, 3.0),
                "is_follow": trial.suggest_float("is_follow", 1.0, 4.0),
            }
            return evaluate(weights)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=trials)
        return {k: float(v) for k, v in study.best_params.items()}
    except Exception:
        rng = random.Random(42)
        best_weights = DEFAULT_WEIGHTS
        best_score = evaluate(best_weights)
        for _ in range(trials):
            weights = {
                "watch_ratio": rng.uniform(0.1, 1.0),
                "long_view": rng.uniform(0.5, 2.0),
                "is_like": rng.uniform(0.5, 3.0),
                "is_comment": rng.uniform(0.5, 3.0),
                "is_forward": rng.uniform(0.5, 3.0),
                "is_follow": rng.uniform(1.0, 4.0),
            }
            score = evaluate(weights)
            if score > best_score:
                best_score = score
                best_weights = weights
        return best_weights


def write_parquet(df: DataFrame, path: Path, overwrite: bool) -> None:
    mode = "overwrite" if overwrite else "errorifexists"
    df.write.mode(mode).parquet(str(path))


def write_json(data: dict, path: Path) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def run_pipeline(args: argparse.Namespace) -> dict:
    spark = get_spark("kuairand-als-baseline", reset=True)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_events = read_interactions(spark, args.silver_dir)
    split, split_meta = split_events(raw_events)
    split = split.cache()
    base_summary = split_summary(split)

    weighted_default = add_event_strength(split, DEFAULT_WEIGHTS)
    train_events = weighted_default.where(F.col("split") == "train").cache()
    train_interactions_original = build_train_interactions(train_events).cache()
    user_mapping, item_mapping = build_mappings(train_interactions_original)
    user_mapping = user_mapping.cache()
    item_mapping = item_mapping.cache()

    validation_relevance_default = build_relevance(weighted_default, user_mapping, item_mapping, "validation", DEFAULT_WEIGHTS).cache()
    selected_weights = tune_strength_weights(
        split.where(F.col("split") == "train"),
        validation_relevance_default,
        user_mapping,
        item_mapping,
        args.strength_tuning_trials,
        args.k,
    )
    weighted = add_event_strength(split, selected_weights)
    train_events = weighted.where(F.col("split") == "train").cache()
    train_interactions_original = build_train_interactions(train_events).cache()
    user_mapping, item_mapping = build_mappings(train_interactions_original)
    user_mapping = user_mapping.cache()
    item_mapping = item_mapping.cache()
    train_interactions = apply_mappings(train_interactions_original, user_mapping, item_mapping).cache()

    validation_relevance = build_relevance(weighted, user_mapping, item_mapping, "validation", selected_weights).cache()
    test_relevance = build_relevance(weighted, user_mapping, item_mapping, "test", selected_weights).cache()
    train_history = train_interactions.select("user_idx", "video_idx").cache()

    write_parquet(train_interactions.select("user_idx", "video_idx", "interaction_strength", "user_id", "video_id"), output_dir / "train_interactions", args.overwrite)
    write_parquet(validation_relevance, output_dir / "validation_relevance", args.overwrite)
    write_parquet(test_relevance, output_dir / "test_relevance", args.overwrite)
    write_parquet(user_mapping, output_dir / "user_mapping", args.overwrite)
    write_parquet(item_mapping, output_dir / "item_mapping", args.overwrite)

    validation_users = validation_relevance.where(F.col("is_warm_user")).select("user_idx").distinct()
    test_users = test_relevance.where(F.col("is_warm_user")).select("user_idx").distinct()

    pop_val_recs = popularity_recommendations(train_interactions, validation_users, train_history, args.k).cache()
    pop_test_recs = popularity_recommendations(train_interactions, test_users, train_history, args.k).cache()
    pop_validation = ranking_metrics(pop_val_recs, validation_relevance, k=args.k)
    pop_test = ranking_metrics(pop_test_recs, test_relevance, k=args.k)

    grid = [
        AlsParams(rank=rank, reg_param=reg, alpha=alpha, max_iter=args.max_iter)
        for rank, reg, alpha in itertools.product(args.ranks, args.reg_params, args.alphas)
    ]
    if args.max_grid_models:
        grid = grid[: args.max_grid_models]

    als_results = []
    best = None
    for params in grid:
        model = train_als(train_interactions, params)
        recs = als_recommendations(model, validation_users, train_history, args.k, over_generate=args.als_over_generate).cache()
        metrics = ranking_metrics(recs, validation_relevance, k=args.k)
        result = {"params": asdict(params), "validation": metrics}
        als_results.append(result)
        score = metrics[f"ndcg@{args.k}"]
        if best is None or score > best["score"]:
            best = {"score": score, "params": params}
        recs.unpersist()

    if best is None:
        raise RuntimeError("ALS grid was empty")

    selected_params: AlsParams = best["params"]
    train_validation_events = weighted.where(F.col("split").isin("train", "validation")).cache()
    final_original = build_train_interactions(train_validation_events).cache()
    final_users, final_items = build_mappings(final_original)
    final_users = final_users.cache()
    final_items = final_items.cache()
    final_train = apply_mappings(final_original, final_users, final_items).cache()
    final_history = final_train.select("user_idx", "video_idx").cache()
    final_test_relevance = build_relevance(weighted, final_users, final_items, "test", selected_weights).cache()
    final_test_users = final_test_relevance.where(F.col("is_warm_user")).select("user_idx").distinct()

    final_model = train_als(final_train, selected_params)
    final_test_recs = als_recommendations(final_model, final_test_users, final_history, args.k, over_generate=args.als_over_generate).cache()
    final_test_metrics = ranking_metrics(final_test_recs, final_test_relevance, k=args.k)

    write_parquet(final_train.select("user_idx", "video_idx", "interaction_strength", "user_id", "video_id"), output_dir / "train_validation_interactions", args.overwrite)
    write_parquet(final_users, output_dir / "final_user_mapping", args.overwrite)
    write_parquet(final_items, output_dir / "final_item_mapping", args.overwrite)
    write_parquet(final_model.userFactors.withColumnRenamed("id", "user_idx"), output_dir / "user_factors", args.overwrite)
    write_parquet(final_model.itemFactors.withColumnRenamed("id", "video_idx"), output_dir / "item_factors", args.overwrite)

    manifest = {
        "gold_dataset_version": args.version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_silver_path": str(args.silver_dir),
        "output_path": str(output_dir),
        "code_commit": git_commit(),
        "interaction_strength": {
            "formula_version": FORMULA_VERSION,
            "formula": "log1p(sum(0.5*clip(watch_ratio,0,3)+1.0*long_view+1.5*like+1.5*comment+1.5*forward+2.0*follow)) with selected weights if tuning is enabled",
            "weights": selected_weights,
            "negative_feedback_policy": "is_hate is retained in silver but not encoded as negative ALS rating in v1",
        },
        "temporal_split": split_meta,
        "split_summary": base_summary,
        "mapping_statistics_train_only": {
            "users": user_mapping.count(),
            "items": item_mapping.count(),
            "train_user_item_rows": train_interactions.count(),
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
        "popularity_baseline": {"validation": pop_validation, "test": pop_test},
        "als_grid_results": als_results,
        "selected_als_hyperparameters": asdict(selected_params),
        "final_als_test_metrics": final_test_metrics,
        "artifact_paths": {
            "train_interactions": str(output_dir / "train_interactions"),
            "validation_relevance": str(output_dir / "validation_relevance"),
            "test_relevance": str(output_dir / "test_relevance"),
            "user_mapping": str(output_dir / "user_mapping"),
            "item_mapping": str(output_dir / "item_mapping"),
            "user_factors": str(output_dir / "user_factors"),
            "item_factors": str(output_dir / "item_factors"),
        },
    }
    write_json(manifest, output_dir / "manifest.json")
    write_json(
        {
            "popularity_validation": pop_validation,
            "popularity_test": pop_test,
            "als_grid_results": als_results,
            "final_als_test": final_test_metrics,
        },
        output_dir / "evaluation_summary.json",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    spark.stop()
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build KuaiRand ALS gold data and train popularity/Spark ALS baselines.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--silver-dir", type=Path, default=Path("data/silver/kuairand"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/gold/als/v1"))
    parser.add_argument("--version", default="als_v1")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--ranks", type=parse_int_list, default=[32, 64])
    parser.add_argument("--reg-params", type=parse_float_list, default=[0.05, 0.1])
    parser.add_argument("--alphas", type=parse_float_list, default=[10.0, 20.0])
    parser.add_argument("--max-iter", type=int, default=8)
    parser.add_argument("--max-grid-models", type=int, default=0, help="Optional cap for local smoke runs; 0 means all grid combinations.")
    parser.add_argument("--als-over-generate", type=int, default=300)
    parser.add_argument("--strength-tuning-trials", type=int, default=0, help="Optional Optuna/random-search trials for interaction-strength weights.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    run_pipeline(parse_args())


if __name__ == "__main__":
    main()
