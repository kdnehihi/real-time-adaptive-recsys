from __future__ import annotations

from dataclasses import dataclass, field


OOV_INDEX = 0

TARGET_ONLY_COLUMNS = frozenset(
    {
        "target_class",
        "is_positive",
        "is_observed_negative",
        "is_ambiguous",
        "engagement_strength",
        "watch_ratio_clipped",
        "long_view",
        "is_like",
        "is_comment",
        "is_forward",
        "is_follow",
        "is_hate",
        "is_click",
        "is_profile_enter",
        "play_time_ms",
        "duration_ms",
        "watch_ratio",
    }
)

USER_CATEGORICAL_FEATURES = ("user_active_degree",)
ITEM_CATEGORICAL_FEATURES = ("video_type", "upload_type")

USER_ID_FEATURES = ("user_id",)
ITEM_ID_FEATURES = ("video_id",)

USER_NUMERIC_FEATURES = (
    "is_lowactive_period",
    "is_live_streamer",
    "is_video_author",
    "follow_user_num",
    "fans_user_num",
    "friend_user_num",
    "register_days",
    "onehot_feat0",
    "onehot_feat1",
    "onehot_feat2",
    "onehot_feat3",
    "onehot_feat4",
    "onehot_feat5",
    "onehot_feat6",
    "onehot_feat7",
    "onehot_feat8",
    "onehot_feat9",
    "onehot_feat10",
    "onehot_feat11",
    "onehot_feat12",
    "onehot_feat13",
    "onehot_feat14",
    "onehot_feat15",
    "onehot_feat16",
    "onehot_feat17",
    "user_hist_events",
    "user_hist_long_view_rate",
    "user_hist_like_rate",
    "user_hist_comment_rate",
    "user_hist_forward_rate",
    "user_hist_follow_rate",
    "user_hist_hate_rate",
    "user_hist_avg_watch_ratio",
    "user_hist_avg_play_time_sec",
    "session_event_index",
    "session_elapsed_sec",
    "session_prior_long_view_rate",
    "session_prior_like_rate",
    "session_prior_hate_rate",
    "session_prior_avg_watch_ratio",
    "session_vs_user_long_view_delta",
    "session_vs_user_watch_ratio_delta",
    "event_hour",
    "event_dayofweek",
    "tab",
    "is_rand",
)

ITEM_NUMERIC_FEATURES = (
    "video_duration_sec",
    "aspect_ratio",
    "visible_status",
    "music_type",
    "upload_age_days_at_event",
    "item_hist_events",
    "item_hist_long_view_rate",
    "item_hist_like_rate",
    "item_hist_hate_rate",
    "item_hist_avg_watch_ratio",
)


@dataclass(frozen=True)
class EmbeddingTableConfig:
    """Config for one trainable embedding table."""

    feature_name: str
    num_embeddings: int
    embedding_dim: int
    oov_index: int = OOV_INDEX

    def __post_init__(self) -> None:
        if self.num_embeddings <= self.oov_index:
            raise ValueError(f"{self.feature_name} num_embeddings must include an OOV row")
        if self.embedding_dim <= 0:
            raise ValueError(f"{self.feature_name} embedding_dim must be positive")


@dataclass(frozen=True)
class NumericalTransformConfig:
    """How train-only numerical stats are applied."""

    clip: bool = True
    normalize: bool = True
    impute_strategy: str = "mean"
    eps: float = 1e-6

    def __post_init__(self) -> None:
        if self.impute_strategy not in {"mean", "median", "zero"}:
            raise ValueError("impute_strategy must be one of: mean, median, zero")


@dataclass(frozen=True)
class TwoTowerFeatureConfig:
    """Feature contract for the current Two-Tower feature encoders."""

    user_categorical_features: tuple[str, ...] = USER_CATEGORICAL_FEATURES
    item_categorical_features: tuple[str, ...] = ITEM_CATEGORICAL_FEATURES
    user_id_features: tuple[str, ...] = USER_ID_FEATURES
    item_id_features: tuple[str, ...] = ITEM_ID_FEATURES
    user_numeric_features: tuple[str, ...] = USER_NUMERIC_FEATURES
    item_numeric_features: tuple[str, ...] = ITEM_NUMERIC_FEATURES
    embedding_dims: dict[str, int] = field(
        default_factory=lambda: {
            "user_active_degree": 4,
            "video_type": 3,
            "upload_type": 8,
            "user_id": 8,
            "video_id": 16,
            "author_id": 8,
        }
    )
    numerical: NumericalTransformConfig = field(default_factory=NumericalTransformConfig)
    target_only_columns: frozenset[str] = TARGET_ONLY_COLUMNS

    @property
    def user_input_features(self) -> tuple[str, ...]:
        return self.user_id_features + self.user_categorical_features + self.user_numeric_features

    @property
    def item_input_features(self) -> tuple[str, ...]:
        return self.item_id_features + self.item_categorical_features + self.item_numeric_features


DEFAULT_CONFIG = TwoTowerFeatureConfig()


def embedding_table_configs(
    feature_names: tuple[str, ...],
    vocabulary_sizes: dict[str, int],
    config: TwoTowerFeatureConfig = DEFAULT_CONFIG,
) -> dict[str, EmbeddingTableConfig]:
    """Build embedding table configs for the requested features."""

    tables: dict[str, EmbeddingTableConfig] = {}
    for feature_name in feature_names:
        if feature_name not in vocabulary_sizes:
            raise ValueError(f"Missing vocabulary size for {feature_name}")
        if feature_name not in config.embedding_dims:
            raise ValueError(f"Missing embedding dimension for {feature_name}")
        tables[feature_name] = EmbeddingTableConfig(
            feature_name=feature_name,
            num_embeddings=vocabulary_sizes[feature_name],
            embedding_dim=config.embedding_dims[feature_name],
        )
    return tables
