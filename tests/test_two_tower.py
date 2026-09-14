"""
Unit Tests for Phase 28: Two-Tower Deep Learning Recommender System (PyTorch).
"""

import os
from unittest.mock import patch
import pytest
import numpy as np
import torch
from fastapi.testclient import TestClient

from src.two_tower_recommender import (
    UserTower,
    ItemTower,
    TwoTowerModel,
    USER_FEATURE_DIM,
    ITEM_FEATURE_DIM,
    EMBEDDING_DIM,
    train_two_tower_model
)
from app.two_tower_service import TwoTowerRecommenderService
from app.main import app


@pytest.fixture
def sample_user_batch():
    # Batch of 4 users, each with 6 behavioral features
    return torch.randn(4, USER_FEATURE_DIM)


@pytest.fixture
def sample_item_batch():
    # Batch of 4 items, each with 4 metadata features
    return torch.randn(4, ITEM_FEATURE_DIM)


def test_user_tower_forward_shape_and_normalization(sample_user_batch):
    """Verifies that UserTower projects features to unit-norm 64d latent vectors."""
    user_tower = UserTower(input_dim=USER_FEATURE_DIM, embedding_dim=EMBEDDING_DIM)
    user_tower.eval()
    
    with torch.no_grad():
        embeddings = user_tower(sample_user_batch)

    assert embeddings.shape == (4, EMBEDDING_DIM)
    
    # Check L2 normalization (norm should be 1.0 for each vector)
    norms = torch.norm(embeddings, p=2, dim=-1)
    for norm in norms:
        assert pytest.approx(norm.item(), rel=1e-4) == 1.0


def test_item_tower_forward_shape_and_normalization(sample_item_batch):
    """Verifies that ItemTower projects features to unit-norm 64d latent vectors."""
    item_tower = ItemTower(input_dim=ITEM_FEATURE_DIM, embedding_dim=EMBEDDING_DIM)
    item_tower.eval()
    
    with torch.no_grad():
        embeddings = item_tower(sample_item_batch)

    assert embeddings.shape == (4, EMBEDDING_DIM)
    
    norms = torch.norm(embeddings, p=2, dim=-1)
    for norm in norms:
        assert pytest.approx(norm.item(), rel=1e-4) == 1.0


def test_two_tower_forward_affinity(sample_user_batch, sample_item_batch):
    """Verifies that TwoTowerModel computes cosine affinity within [-1.0, 1.0]."""
    model = TwoTowerModel(
        user_dim=USER_FEATURE_DIM,
        item_dim=ITEM_FEATURE_DIM,
        embedding_dim=EMBEDDING_DIM
    )
    model.eval()

    with torch.no_grad():
        affinities = model(sample_user_batch, sample_item_batch)

    assert affinities.shape == (4,)
    for val in affinities:
        assert -1.05 <= val.item() <= 1.05


def test_two_tower_training_pipeline(tmp_path):
    """Verifies that the training routine runs and saves model weights."""
    model_path = str(tmp_path / "test_two_tower.pt")
    model = train_two_tower_model(
        epochs=2,
        batch_size=16,
        lr=0.01,
        output_model_path=model_path
    )

    assert model is not None
    assert os.path.exists(model_path)


@patch("app.db_postgres.get_online_features")
def test_two_tower_recommender_service(mock_features):
    """Verifies that TwoTowerRecommenderService generates ranked product recommendations."""
    mock_features.return_value = {
        "recency": 10.0,
        "frequency": 4,
        "avg_order_value": 50.0,
        "spending_velocity": 1.0,
        "cancellation_rate": 0.0,
        "preferred_shopping_hour": 14
    }
    service = TwoTowerRecommenderService()
    result = service.recommend_for_customer("17850", top_k=3)

    assert result["customer_id"] == "17850"
    assert result["embedding_dimension"] == EMBEDDING_DIM
    assert len(result["recommendations"]) == 3
    assert "stock_code" in result["recommendations"][0]
    assert "affinity_score" in result["recommendations"][0]
    assert 0.0 <= result["recommendations"][0]["affinity_score"] <= 1.0


@patch("app.db_postgres.get_online_features")
def test_two_tower_fastapi_endpoints(mock_features):
    """Verifies FastAPI GET and POST /recommend/two-tower endpoints."""
    mock_features.return_value = {
        "recency": 10.0,
        "frequency": 4,
        "avg_order_value": 50.0,
        "spending_velocity": 1.0,
        "cancellation_rate": 0.0,
        "preferred_shopping_hour": 14
    }
    client = TestClient(app)

    # 1. Test GET /recommend/two-tower/{customer_id}
    res_get = client.get("/recommend/two-tower/17850?top_k=2")
    assert res_get.status_code == 200
    data_get = res_get.json()
    assert data_get["customer_id"] == "17850"
    assert len(data_get["recommendations"]) == 2

    # 2. Test POST /recommend/two-tower with custom profile
    res_post = client.post("/recommend/two-tower", json={
        "customer_id": "99999",
        "recency": 5.0,
        "frequency": 8,
        "avg_order_value": 120.0,
        "top_k": 3
    })
    assert res_post.status_code == 200
    data_post = res_post.json()
    assert data_post["customer_id"] == "99999"
    assert len(data_post["recommendations"]) == 3
