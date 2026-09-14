"""
Inference Service for Two-Tower Deep Learning Recommendations.

Generates real-time 64-dimensional user embeddings from customer RFM & behavioral features,
queries catalog candidates, and returns personalized recommendations with similarity scores.
"""

import os
import logging
from typing import Dict, Any, List, Optional
import numpy as np
import torch

from src.two_tower_recommender import (
    TwoTowerModel,
    USER_FEATURE_DIM,
    ITEM_FEATURE_DIM,
    EMBEDDING_DIM
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("two_tower_service")


class TwoTowerRecommenderService:
    """Production serving wrapper for PyTorch Two-Tower Dual-Encoder inference."""

    def __init__(self, model_path: str = "models/two_tower_model.pt"):
        self.model_path = model_path
        self.model = self._load_or_initialize_model()
        self.catalog_cache = self._build_catalog_cache()

    def _load_or_initialize_model(self) -> TwoTowerModel:
        """Loads trained weights or initializes model architecture in evaluation mode."""
        model = TwoTowerModel(
            user_dim=USER_FEATURE_DIM,
            item_dim=ITEM_FEATURE_DIM,
            embedding_dim=EMBEDDING_DIM
        )
        if os.path.exists(self.model_path):
            try:
                checkpoint = torch.load(self.model_path, map_location=torch.device("cpu"))
                if "model_state_dict" in checkpoint:
                    model.load_state_dict(checkpoint["model_state_dict"])
                else:
                    model.load_state_dict(checkpoint)
                logger.info("Loaded pre-trained Two-Tower weights from %s", self.model_path)
            except Exception as e:
                logger.warning("Could not load weights from %s: %s. Using initialized model.", self.model_path, e)
        else:
            logger.info("No checkpoint found at %s. Initialized base Two-Tower model.", self.model_path)

        model.eval()
        return model

    def _build_catalog_cache(self) -> List[Dict[str, Any]]:
        """Pre-computes and caches catalog item features and embeddings for fast retrieval."""
        default_catalog = [
            {"stock_code": "85123A", "description": "WHITE HANGING HEART T-LIGHT HOLDER", "category": "Home Decor & Lighting", "unit_price": 2.55, "category_id": 1},
            {"stock_code": "22423", "description": "REGENCY CAKESTAND 3 TIER", "category": "Kitchen & Dining", "unit_price": 12.75, "category_id": 0},
            {"stock_code": "47566", "description": "PARTY BUNTING", "category": "Party & Celebration", "unit_price": 4.95, "category_id": 5},
            {"stock_code": "84879", "description": "ASSORTED COLOUR BIRD ORNAMENT", "category": "Home Decor & Lighting", "unit_price": 1.69, "category_id": 1},
            {"stock_code": "20725", "description": "LUNCH BAG RED RETROSPOT", "category": "Storage & Accessories", "unit_price": 2.45, "category_id": 4},
            {"stock_code": "22086", "description": "PAPER CHAIN KIT 50'S CHRISTMAS", "category": "Holiday & Seasonal", "unit_price": 2.95, "category_id": 2},
            {"stock_code": "22197", "description": "SMALL POPCORN HOLDER", "category": "Kitchen & Dining", "unit_price": 0.85, "category_id": 0},
            {"stock_code": "22960", "description": "JAM MAKING SET WITH JARS", "category": "Kitchen & Dining", "unit_price": 3.75, "category_id": 0},
        ]

        cached = []
        for item in default_catalog:
            price = float(item["unit_price"])
            cat_id = float(item["category_id"])
            item_feat = np.array([
                price,
                cat_id,
                float(np.log1p(max(price, 0.1))),
                1.0
            ], dtype=np.float32)

            with torch.no_grad():
                tensor_feat = torch.tensor(item_feat, dtype=torch.float32).unsqueeze(0)
                item_emb = self.model.item_tower(tensor_feat).cpu().numpy()[0]

            cached.append({
                **item,
                "item_features": item_feat,
                "embedding": item_emb
            })

        return cached

    def recommend_for_customer(
        self,
        customer_id: str,
        custom_features: Optional[Dict[str, Any]] = None,
        top_k: int = 4
    ) -> Dict[str, Any]:
        """
        Generates personalized product recommendations for a customer using Two-Tower embeddings.
        """
        # 1. Fetch features from Feature Store or fallback
        features = custom_features
        if not features:
            try:
                from app.db_postgres import get_online_features
                features = get_online_features(customer_id)
            except Exception:
                features = None

        if not features:
            features = {
                "recency": 15.0,
                "frequency": 3,
                "avg_order_value": 45.0,
                "spending_velocity": 1.0,
                "cancellation_rate": 0.0,
                "preferred_shopping_hour": 14
            }

        # 2. Build User Feature Tensor
        user_feat = np.array([
            float(features.get("recency", 15.0)),
            float(features.get("frequency", 3)),
            float(features.get("avg_order_value", 45.0)),
            float(features.get("spending_velocity", 1.0)),
            float(features.get("cancellation_rate", 0.0)),
            float(features.get("preferred_shopping_hour", 14))
        ], dtype=np.float32)

        # 3. Compute 64d User Embedding with User Tower
        with torch.no_grad():
            user_tensor = torch.tensor(user_feat, dtype=torch.float32).unsqueeze(0)
            user_emb = self.model.user_tower(user_tensor).cpu().numpy()[0]

        # 4. Rank catalog items by cosine similarity (dot product on normalized vectors)
        scored_candidates = []
        for item in self.catalog_cache:
            item_emb = item["embedding"]
            score = float(np.dot(user_emb, item_emb))
            scored_candidates.append({
                "stock_code": item["stock_code"],
                "description": item["description"],
                "category": item["category"],
                "unit_price": item["unit_price"],
                "affinity_score": round((score + 1.0) / 2.0, 4) # Rescale [-1, 1] to [0, 1]
            })

        # Sort descending by affinity score
        scored_candidates.sort(key=lambda x: x["affinity_score"], reverse=True)
        top_recs = scored_candidates[:top_k]

        return {
            "customer_id": str(customer_id),
            "engine": "Two-Tower Deep Learning (PyTorch)",
            "embedding_dimension": EMBEDDING_DIM,
            "user_embedding_norm": float(np.linalg.norm(user_emb)),
            "recommendations": top_recs
        }
