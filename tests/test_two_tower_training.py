import torch

from recommender.models.two_tower.config import DEFAULT_CONFIG, embedding_table_configs
from recommender.models.two_tower.evaluation import in_batch_retrieval_metrics
from recommender.models.two_tower.model import TwoTowerRetrievalModel


def test_two_tower_model_forward_shapes():
    config = DEFAULT_CONFIG
    user_tables = embedding_table_configs(
        config.user_id_features + config.user_categorical_features,
        {"user_id": 4, "user_active_degree": 3},
        config,
    )
    item_tables = embedding_table_configs(
        config.item_id_features + config.item_categorical_features,
        {"video_id": 5, "video_type": 3, "upload_type": 4},
        config,
    )
    model = TwoTowerRetrievalModel(
        user_tables=user_tables,
        item_tables=item_tables,
        user_numeric_features=config.user_numeric_features,
        item_numeric_features=config.item_numeric_features,
        retrieval_dim=16,
        user_hidden_dims=(32,),
        item_hidden_dims=(32,),
    )
    batch_size = 4
    output = model(
        user_categorical={
            "user_id": torch.tensor([1, 2, 3, 0]),
            "user_active_degree": torch.tensor([1, 2, 0, 1]),
        },
        user_numeric=torch.zeros((batch_size, len(config.user_numeric_features))),
        item_categorical={
            "video_id": torch.tensor([1, 2, 3, 0]),
            "video_type": torch.tensor([1, 2, 0, 1]),
            "upload_type": torch.tensor([1, 2, 3, 0]),
        },
        item_numeric=torch.zeros((batch_size, len(config.item_numeric_features))),
    )

    assert output.user_embedding.shape == (batch_size, 16)
    assert output.item_embedding.shape == (batch_size, 16)
    assert output.logits.shape == (batch_size, batch_size)
    assert torch.isfinite(output.logits).all()


def test_in_batch_retrieval_metrics_detect_perfect_ranking():
    logits = torch.eye(4) * 10.0
    metrics = in_batch_retrieval_metrics(logits, (1, 2))

    assert metrics["hitrate@1"] == 1.0
    assert metrics["recall@1"] == 1.0
    assert metrics["ndcg@1"] == 1.0
    assert metrics["mrr"] == 1.0
