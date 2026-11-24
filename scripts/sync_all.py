"""
Unified Online Store Synchronization Worker (Reverse ETL).
Orchestrates the replication of BigQuery analytical layers to operational stores:
1. Online Customer Feature Store (Gold RFM -> Firestore / PostgreSQL)
2. Vector Product Catalog (BigQuery ML embeddings -> PostgreSQL pgvector)
"""

import os
import sys
import time
import argparse
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.sync_feature_store import sync_feature_store
from scripts.sync_product_vectors import sync_product_vectors

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("sync_all")


def sync_all(features: bool = True, vectors: bool = True) -> dict:
    """
    Executes unified synchronization of customer features and product catalog vectors.
    Fails fast if any component encounters an unhandled error.
    """
    start_total = time.perf_counter()
    results = {}

    logger.info("==================================================================")
    logger.info("Starting Unified Online Store Synchronization Worker")
    logger.info("Environment: %s", "Cloud (BigQuery)" if os.getenv("USE_BIGQUERY", "false").lower() == "true" else "Local")
    logger.info("Targets: features=%s, vectors=%s", features, vectors)
    logger.info("==================================================================")

    # 1. Sync Customer Features (Feature Store)
    if features:
        t0 = time.perf_counter()
        logger.info("[1/2] Synchronizing Online Customer Feature Store...")
        try:
            sync_feature_store()
            elapsed = round(time.perf_counter() - t0, 2)
            results["features"] = {"status": "success", "duration_seconds": elapsed}
            logger.info("✅ Customer features synchronized successfully in %ss.", elapsed)
        except Exception as e:
            elapsed = round(time.perf_counter() - t0, 2)
            results["features"] = {"status": "failed", "error": str(e), "duration_seconds": elapsed}
            logger.error("❌ Customer feature store synchronization failed: %s", e)
            raise RuntimeError(f"Feature store synchronization failed: {e}") from e

    # 2. Sync Product Catalog Vectors (Vector Store)
    if vectors:
        t0 = time.perf_counter()
        logger.info("[2/2] Synchronizing Product Catalog Vectors (pgvector)...")
        try:
            sync_product_vectors()
            elapsed = round(time.perf_counter() - t0, 2)
            results["vectors"] = {"status": "success", "duration_seconds": elapsed}
            logger.info("✅ Product catalog vectors synchronized successfully in %ss.", elapsed)
        except Exception as e:
            elapsed = round(time.perf_counter() - t0, 2)
            results["vectors"] = {"status": "failed", "error": str(e), "duration_seconds": elapsed}
            logger.error("❌ Product catalog vector synchronization failed: %s", e)
            raise RuntimeError(f"Product vector synchronization failed: {e}") from e

    total_elapsed = round(time.perf_counter() - start_total, 2)
    results["total_duration_seconds"] = total_elapsed
    logger.info("==================================================================")
    logger.info("🎉 Unified Synchronization Completed Successfully in %ss!", total_elapsed)
    logger.info("==================================================================")
    return results


def main():
    parser = argparse.ArgumentParser(description="Unified Reverse ETL Sync Worker")
    parser.add_argument("--features-only", action="store_true", help="Sync only customer features to Feature Store")
    parser.add_argument("--vectors-only", action="store_true", help="Sync only product catalog vectors to pgvector")
    args = parser.parse_args()

    if args.features_only and args.vectors_only:
        logger.error("Cannot specify both --features-only and --vectors-only simultaneously.")
        sys.exit(1)

    sync_features = not args.vectors_only
    sync_vecs = not args.features_only

    try:
        sync_all(features=sync_features, vectors=sync_vecs)
    except Exception as e:
        logger.error("Sync worker failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
