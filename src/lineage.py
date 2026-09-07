"""
Data Lineage & Provenance Tracker.

Tracks end-to-end lineage across all pipeline stages:
Raw Ingestion (Pub/Sub/GCS) -> BigQuery -> Dataproc PySpark -> Feature Store -> MLflow Model.
Generates structured JSON manifests and Mermaid lineage diagrams for auditability and compliance.
"""

import os
import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("lineage")

MANIFEST_PATH = Path("reports/lineage_manifest.json")


def get_current_git_commit() -> str:
    """Returns the current Git commit SHA or fallback string."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        return commit
    except Exception:
        return os.getenv("GIT_COMMIT_SHA", "dev-local-044c8c7")


class DataLineageTracker:
    """
    Maintains provenance records and lineage graphs across the platform.
    """
    
    def __init__(self, manifest_file: Path = MANIFEST_PATH):
        self.manifest_file = manifest_file
        self.manifest_file.parent.mkdir(parents=True, exist_ok=True)
        self.history = self._load_manifest()

    def _load_manifest(self) -> List[Dict[str, Any]]:
        if self.manifest_file.exists():
            try:
                with open(self.manifest_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning("Failed to load lineage manifest: %s", e)
                return []
        return []

    def _save_manifest(self):
        with open(self.manifest_file, "w") as f:
            json.dump(self.history, f, indent=2)

    def record_stage_execution(
        self,
        stage_name: str,
        inputs: List[Dict[str, Any]],
        outputs: List[Dict[str, Any]],
        transformation_type: str = "batch_pyspark",
        model_metadata: Optional[Dict[str, Any]] = None,
        dvc_tag: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Records an execution node in the data lineage graph.
        """
        record = {
            "record_id": f"lin_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            "timestamp_utc": datetime.utcnow().isoformat(),
            "stage_name": stage_name,
            "git_commit": get_current_git_commit(),
            "dvc_tag": dvc_tag or f"dvc-v{datetime.utcnow().strftime('%Y.%m.%d')}",
            "transformation_type": transformation_type,
            "inputs": inputs,
            "outputs": outputs,
            "model_metadata": model_metadata or {}
        }
        
        self.history.append(record)
        self._save_manifest()
        logger.info("Recorded data lineage record '%s' for stage '%s'", record["record_id"], stage_name)
        return record

    def get_latest_lineage_graph_mermaid(self) -> str:
        """
        Generates a Mermaid graph representing the end-to-end lineage flow.
        """
        return """graph LR
    subgraph S1 [1. Ingestion Layer]
        Source["GCS / Pub/Sub Stream<br/>gs://anna-ml-pipeline-bucket/raw/"]
        RawBQ[("BigQuery: retail_data.transactions<br/>(Time Travel Enabled)")]
        Source --> RawBQ
    end

    subgraph S2 [2. Distributed Processing & Versioning]
        Dataproc["Dataproc PySpark (Phase 23)<br/>Windowing & RFM Calculation"]
        Snap[("BigQuery Snapshot<br/>transactions_snapshot_20260907")]
        RawBQ --> Dataproc
        RawBQ -.->|Zero-Copy Snapshot| Snap
    end

    subgraph S3 [3. Feature Store & Serving]
        FeatBQ[("BigQuery: rfm_features")]
        FS[("Online Feature Store<br/>Firestore / PostgreSQL")]
        Dataproc --> FeatBQ
        FeatBQ --> FS
    end

    subgraph S4 [4. MLOps & Training Lineage]
        Vertex["Vertex AI Pipelines (KFP)"]
        MLflow[("MLflow Model Registry<br/>Run: churn_xgboost_v1.2")]
        FeatBQ --> Vertex
        Vertex --> MLflow
    end

    classDef raw fill:#FBBC04,stroke:#333,stroke-width:2px,color:#000;
    classDef bq fill:#4285F4,stroke:#333,stroke-width:2px,color:#fff;
    classDef spark fill:#E25A1C,stroke:#333,stroke-width:2px,color:#fff;
    classDef ml fill:#34A853,stroke:#333,stroke-width:2px,color:#fff;
    class Source,RawBQ,Snap raw;
    class FeatBQ,FS bq;
    class Dataproc spark;
    class Vertex,MLflow ml;
"""


# Global singleton instance
tracker = DataLineageTracker()
