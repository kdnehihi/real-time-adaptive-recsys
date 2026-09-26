from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
import pyarrow.dataset as ds

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from recommender.models.two_tower.config import DEFAULT_CONFIG, embedding_table_configs
from recommender.models.two_tower.encoders import ItemFeatureEncoder, UserFeatureEncoder
from recommender.models.two_tower.inspection import embedding_parameter_reports, total_embedding_parameters
from recommender.models.two_tower.preprocessing import FeatureBatchPreprocessor, NumericalPreprocessor
from recommender.models.two_tower.vocab import Vocabulary, load_parquet_vocabulary


FULL_TRAIN_CARDINALITIES = {
    "user_id": 994,
    "video_id": 3_215_507,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect Two-Tower feature encoding on a small Gold batch.")
    parser.add_argument("--gold-root", type=Path, default=Path("data/gold/two_tower/v1_ready"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--use-full-video-table-size",
        action="store_true",
        help="Allocate the default full train video_id embedding table size for memory-realistic inspection.",
    )
    return parser.parse_args()


def load_small_batch(gold_root: Path, batch_size: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    core = read_parquet_head(gold_root / "train", batch_size)
    user_state = read_parquet_head(gold_root / "user_state" / "train", batch_size)
    item_features = read_parquet_head(gold_root / "item_features" / "point_in_time" / "train", batch_size)
    targets = read_parquet_head(gold_root / "targets" / "train", batch_size)
    return core, user_state, item_features, targets


def read_parquet_head(path: Path, batch_size: int) -> pd.DataFrame:
    return ds.dataset(path, format="parquet").head(batch_size).to_pandas()


def load_demo_vocabularies(gold_root: Path, user_state: pd.DataFrame, item_features: pd.DataFrame) -> dict[str, Vocabulary]:
    vocabularies = {
        "user_active_degree": load_parquet_vocabulary(gold_root / "vocabularies" / "user_active_degree", "user_active_degree"),
        "video_type": load_parquet_vocabulary(gold_root / "vocabularies" / "video_type", "video_type"),
        "upload_type": load_parquet_vocabulary(gold_root / "vocabularies" / "upload_type", "upload_type"),
    }
    vocabularies["user_id"] = Vocabulary.from_values("user_id", user_state["user_id"].tolist())
    vocabularies["video_id"] = Vocabulary.from_values("video_id", item_features["video_id"].tolist())
    return vocabularies


def main() -> None:
    args = parse_args()
    config = DEFAULT_CONFIG
    core, user_state, item_features, targets = load_small_batch(args.gold_root, args.batch_size)
    vocabularies = load_demo_vocabularies(args.gold_root, user_state, item_features)
    numeric = NumericalPreprocessor.from_json(args.gold_root / "transforms" / "numeric_stats.json", config.numerical)
    preprocessor = FeatureBatchPreprocessor(config=config, vocabularies=vocabularies, numerical_preprocessor=numeric)

    user_vocab_sizes = {name: vocabularies[name].size for name in config.user_id_features + config.user_categorical_features}
    item_vocab_sizes = {name: vocabularies[name].size for name in config.item_categorical_features}
    if args.use_full_video_table_size:
        item_vocab_sizes["video_id"] = FULL_TRAIN_CARDINALITIES["video_id"] + 1
    else:
        item_vocab_sizes["video_id"] = vocabularies["video_id"].size

    user_tables = embedding_table_configs(config.user_id_features + config.user_categorical_features, user_vocab_sizes, config)
    item_tables = embedding_table_configs(config.item_id_features + config.item_categorical_features, item_vocab_sizes, config)
    user_encoder = UserFeatureEncoder(user_tables, config.user_numeric_features)
    item_encoder = ItemFeatureEncoder(item_tables, config.item_numeric_features)

    batch = preprocessor.encode_batch(user_state, item_features)
    user_parts = user_encoder.forward_with_parts(batch.user_categorical, batch.user_numeric)
    item_parts = item_encoder.forward_with_parts(batch.item_categorical, batch.item_numeric)

    print("Raw example columns:")
    print(core.iloc[0].to_dict())
    print("\nTarget-only display, not encoder input:")
    print(targets.iloc[0][["target_class", "is_positive", "engagement_strength", "watch_ratio"]].to_dict())
    print("\nCategorical raw -> encoded IDs:")
    for name, values in {**batch.user_categorical, **batch.item_categorical}.items():
        print(f"{name}: {values[:5].tolist()}")

    unseen_item = item_features.head(1).copy()
    unseen_item["video_id"] = -999_999_999
    cold_item_categorical, cold_item_numeric = preprocessor.encode_item(unseen_item)
    cold_vector = item_encoder(cold_item_categorical, cold_item_numeric)
    print("\nArtificial cold video_id maps to:", cold_item_categorical["video_id"].tolist())
    print("Cold item vector shape:", list(cold_vector.shape))

    print("\nShapes:")
    print("User categorical embeddings:", list(user_parts.categorical_embeddings.shape))
    print("User numerical features:", list(user_parts.numerical_features.shape))
    print("UserFeatureEncoder output:", list(user_parts.vector.shape))
    print("Item categorical embeddings:", list(item_parts.categorical_embeddings.shape))
    print("Item numerical features:", list(item_parts.numerical_features.shape))
    print("ItemFeatureEncoder output:", list(item_parts.vector.shape))

    print("\nEmbedding parameter report:")
    all_tables = {**user_tables, **item_tables}
    for report in embedding_parameter_reports(all_tables):
        print(
            f"{report.feature_name}: num_embeddings={report.num_embeddings:,}, "
            f"dim={report.embedding_dim}, params={report.parameters:,}, fp32={report.fp32_memory_mb:.2f} MB"
        )
    print(f"Total embedding params: {total_embedding_parameters(all_tables):,}")
    print(
        "Full train video_id dim=16 estimate: "
        f"{(FULL_TRAIN_CARDINALITIES['video_id'] + 1) * config.embedding_dims['video_id'] * 4 / (1024 * 1024):.2f} MB"
    )


if __name__ == "__main__":
    main()
