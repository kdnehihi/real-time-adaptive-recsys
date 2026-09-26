import math

import pandas as pd
import pytest
import torch

from recommender.models.two_tower.config import DEFAULT_CONFIG, embedding_table_configs
from recommender.models.two_tower.encoders import ItemFeatureEncoder, UserFeatureEncoder
from recommender.models.two_tower.preprocessing import FeatureBatchPreprocessor, NumericalPreprocessor
from recommender.models.two_tower.vocab import Vocabulary


def _stats(features):
    return {name: {"mean": 1.0, "median": 1.0, "stddev": 2.0, "p01": 0.0, "p99": 10.0, "min": 0.0, "max": 10.0} for name in features}


def _frame(features, rows=3):
    return pd.DataFrame({name: [1.0, None, 100.0][:rows] for name in features})


def _preprocessor(config=DEFAULT_CONFIG):
    vocabs = {
        "user_id": Vocabulary.from_values("user_id", [1, 2]),
        "video_id": Vocabulary.from_values("video_id", [10, 20]),
        "user_active_degree": Vocabulary("user_active_degree", {"low_active": 1}),
        "video_type": Vocabulary("video_type", {"NORMAL": 1}),
        "upload_type": Vocabulary("upload_type", {"ShortCamera": 1}),
    }
    stats = _stats(config.user_numeric_features + config.item_numeric_features)
    return FeatureBatchPreprocessor(config, vocabs, NumericalPreprocessor(stats, config.numerical))


def test_target_columns_are_rejected_as_inputs():
    preprocessor = _preprocessor()
    with pytest.raises(ValueError, match="Target/current-event"):
        preprocessor.check_no_target_leakage(["user_id", "watch_ratio", "long_view"])


def test_oov_categorical_mapping_uses_zero():
    vocab = Vocabulary("video_type", {"NORMAL": 1})
    assert vocab.encode_many(["NORMAL", "AD", None]) == [1, 0, 0]


def test_numerical_preprocessing_clips_normalizes_and_has_no_nan_inf():
    config = DEFAULT_CONFIG
    preprocessor = NumericalPreprocessor(_stats(("x",)), config.numerical)
    out = preprocessor.transform_tensor(pd.DataFrame({"x": [None, -5.0, 1000.0, math.inf]}), ("x",))

    assert out.shape == (4, 1)
    assert torch.isfinite(out).all()
    assert out[1].item() == pytest.approx((0.0 - 1.0) / 2.0)
    assert out[2].item() == pytest.approx((10.0 - 1.0) / 2.0)


def test_user_and_item_encoder_output_dimensions_are_stable():
    config = DEFAULT_CONFIG
    preprocessor = _preprocessor(config)
    user_frame = _frame(config.user_numeric_features).assign(user_id=[1, 2, 999], user_active_degree=["low_active", "missing", None])
    item_frame = _frame(config.item_numeric_features).assign(
        video_id=[10, 20, 999],
        video_type=["NORMAL", "AD", None],
        upload_type=["ShortCamera", "Other", None],
    )
    batch = preprocessor.encode_batch(user_frame, item_frame)

    user_tables = embedding_table_configs(
        config.user_id_features + config.user_categorical_features,
        {"user_id": 3, "user_active_degree": 2},
        config,
    )
    item_tables = embedding_table_configs(
        config.item_id_features + config.item_categorical_features,
        {"video_id": 3, "video_type": 2, "upload_type": 2},
        config,
    )
    user_encoder = UserFeatureEncoder(user_tables, config.user_numeric_features)
    item_encoder = ItemFeatureEncoder(item_tables, config.item_numeric_features)

    user_out = user_encoder(batch.user_categorical, batch.user_numeric)
    item_out = item_encoder(batch.item_categorical, batch.item_numeric)

    assert user_out.shape == (3, user_encoder.output_dim)
    assert item_out.shape == (3, item_encoder.output_dim)
    assert user_encoder.output_dim == 8 + 4 + len(config.user_numeric_features)
    assert item_encoder.output_dim == 16 + 3 + 8 + len(config.item_numeric_features)


def test_cold_video_id_uses_oov_but_item_vector_still_valid():
    config = DEFAULT_CONFIG
    preprocessor = _preprocessor(config)
    item_frame = _frame(config.item_numeric_features, rows=1).assign(
        video_id=[123456789],
        video_type=["NORMAL"],
        upload_type=["ShortCamera"],
    )
    categorical, numeric = preprocessor.encode_item(item_frame)
    tables = embedding_table_configs(
        config.item_id_features + config.item_categorical_features,
        {"video_id": 3, "video_type": 2, "upload_type": 2},
        config,
    )
    encoder = ItemFeatureEncoder(tables, config.item_numeric_features)
    out = encoder(categorical, numeric)

    assert categorical["video_id"].tolist() == [0]
    assert out.shape == (1, encoder.output_dim)
    assert torch.isfinite(out).all()


def test_embedding_indices_out_of_bounds_raise_error():
    config = DEFAULT_CONFIG
    tables = embedding_table_configs(("video_id",), {"video_id": 2}, config)
    encoder = ItemFeatureEncoder(tables, ())

    with pytest.raises(ValueError, match="outside embedding table bounds"):
        encoder({"video_id": torch.tensor([2])}, torch.empty((1, 0)))
