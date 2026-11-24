"""
Unified Online Store Synchronization Worker (Reverse ETL).
Orchestrates the sequential replication of BigQuery analytical layers to operational stores:
1. Online Customer Feature Store (Gold RFM -> Firestore / PostgreSQL)
2. Vector Product Catalog (BigQuery ML embeddings -> PostgreSQL pgvector)
"""

import os
import sys
import time
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.sync_feature_store import sync_feature_store
from scripts.sync_product_vectors import sync_product_vectors

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("sync_all")


def sync_all() -> dict:
    """
    Executes full synchronization of customer features and product catalog vectors.
    Fails fast if any component encounters an unhandled error.
    """
    start_total = time.perf_counter()

    logger.info("==================================================================")
    logger.info("Starting Unified Online Store Synchronization Worker (Full)")
    logger.info("Environment: %s", "Cloud (BigQuery)" if os.getenv("USE_BIGQUERY", "false").lower() == "true" else "Local")
    logger.info("==================================================================")

    # 1. Sync Customer Features (Feature Store)
    t0 = time.perf_counter()
    logger.info("[1/2] Synchronizing Online Customer Feature Store...")
    try:
        sync_feature_store()
        elapsed_feat = round(time.perf_counter() - t0, 2)
        logger.info("✅ Customer features synchronized successfully in %ss.", elapsed_feat)
    except Exception as e:
        logger.error("❌ Customer feature store synchronization failed: %s", e)
        raise RuntimeError(f"Feature store synchronization failed: {e}") from e

    # 2. Sync Product Catalog Vectors (Vector Store)
    t0 = time.perf_counter()
    logger.info("[2/2] Synchronizing Product Catalog Vectors (pgvector)...")
    try:
        sync_product_vectors()
        elapsed_vec = round(time.perf_counter() - t0, 2)
        logger.info("✅ Product catalog vectors synchronized successfully in %ss.", elapsed_vec)
    except Exception as e:
        logger.error("❌ Product catalog vector synchronization failed: %s", e)
        raise RuntimeError(f"Product vector synchronization failed: {e}") from e

    total_elapsed = round(time.perf_counter() - start_total, 2)
    logger.info("==================================================================")
    logger.info("🎉 Unified Synchronization Completed Successfully in %ss!", total_elapsed)
    logger.info("==================================================================")
    return {
        "features": {"status": "success", "duration_seconds": elapsed_feat},
        "vectors": {"status": "success", "duration_seconds": elapsed_vec},
        "total_duration_seconds": total_elapsed,
    }


def main():
    try:
        sync_all()
    except Exception as e:
        logger.error("Sync worker failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
