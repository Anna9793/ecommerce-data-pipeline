"""
Programmatic dbt Model Compiler & Lineage DAG Inspector.

This module parses, validates, and compiles dbt Jinja-SQL models and YAML schema tests
into BigQuery SQL execution plans and dependency graphs.
"""

import os
import re
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Any, Set

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("dbt_runner")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DBT_PROJECT_FILE = PROJECT_ROOT / "dbt_project.yml"
MODELS_DIR = PROJECT_ROOT / "dbt_models"


class DbtProjectInspector:
    """
    Parses and compiles dbt models, resolves refs, and evaluates schema tests.
    """

    def __init__(self, project_dir: Path = PROJECT_ROOT, models_dir: Path = MODELS_DIR):
        self.project_dir = project_dir
        self.models_dir = models_dir
        self.config = self._load_project_config()
        self.models = self._discover_models()
        self.tests = self._discover_schema_tests()

    def _load_project_config(self) -> Dict[str, Any]:
        config_path = self.project_dir / "dbt_project.yml"
        if not config_path.exists():
            return {"name": "ecommerce_dbt_pipeline", "version": "1.0.0"}
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def _discover_models(self) -> Dict[str, Dict[str, Any]]:
        models = {}
        for sql_file in self.models_dir.rglob("*.sql"):
            model_name = sql_file.stem
            with open(sql_file, "r") as f:
                content = f.read()

            # Extract ref(...) dependencies using regex
            refs = re.findall(r"\{\{\s*ref\(['\"]([a-zA-Z0-9_]+)['\"]\)\s*\}\}", content)
            
            # Extract config blocks
            config_matches = re.findall(r"\{\{\s*config\((.*?)\)\s*\}\}", content, re.DOTALL)
            materialization = "view"
            if config_matches:
                mat_match = re.search(r"materialized\s*=\s*['\"]([a-zA-Z0-9_]+)['\"]", config_matches[0])
                if mat_match:
                    materialization = mat_match.group(1)

            models[model_name] = {
                "name": model_name,
                "file_path": str(sql_file.relative_to(self.project_dir)),
                "raw_sql": content,
                "dependencies": refs,
                "materialization": materialization,
                "layer": sql_file.parent.name
            }
        return models

    def _discover_schema_tests(self) -> List[Dict[str, Any]]:
        tests = []
        for yml_file in self.models_dir.rglob("*.yml"):
            with open(yml_file, "r") as f:
                try:
                    data = yaml.safe_load(f)
                    if not data or "models" not in data:
                        continue
                    for model_def in data["models"]:
                        model_name = model_def.get("name")
                        for col in model_def.get("columns", []):
                            col_name = col.get("name")
                            for test in col.get("tests", []):
                                if isinstance(test, str):
                                    test_type = test
                                    test_config = {}
                                elif isinstance(test, dict):
                                    test_type = list(test.keys())[0]
                                    test_config = test[test_type]
                                else:
                                    continue
                                tests.append({
                                    "model": model_name,
                                    "column": col_name,
                                    "test_type": test_type,
                                    "test_config": test_config
                                })
                except Exception as e:
                    logger.warning("Error parsing YAML tests in %s: %s", yml_file, e)
        return tests

    def get_lineage_graph(self) -> Dict[str, List[str]]:
        """Returns the DAG mapping model -> upstream dependencies."""
        return {name: info["dependencies"] for name, info in self.models.items()}

    def compile_model_sql(self, model_name: str, target_schema: str = "retail_data") -> str:
        """
        Compiles dbt Jinja SQL into executable BigQuery SQL by resolving {{ ref(...) }}
        and {{ config(...) }} blocks.
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found in dbt project.")

        raw_sql = self.models[model_name]["raw_sql"]

        # 1. Remove config(...) macro
        compiled_sql = re.sub(r"\{\{\s*config\(.*?\)\s*\}\}", "", raw_sql, flags=re.DOTALL)

        # 2. Replace {{ ref('model_name') }} with target table identifier
        def ref_replacer(match):
            ref_name = match.group(1)
            # If the upstream is ephemeral, resolve CTE; else reference dataset.table
            return f"`anna-ml-pipeline.{target_schema}.{ref_name}`"

        compiled_sql = re.sub(r"\{\{\s*ref\(['\"]([a-zA-Z0-9_]+)['\"]\)\s*\}\}", ref_replacer, compiled_sql)
        
        # Clean leading/trailing whitespace
        return compiled_sql.strip()


def run_dbt_inspection():
    """CLI helper to inspect dbt project status."""
    inspector = DbtProjectInspector()
    print(f"\n==========================================")
    print(f"📦 dbt Project: {inspector.config.get('name')}")
    print(f"==========================================")
    print(f"Discovered {len(inspector.models)} SQL models:")
    for name, info in inspector.models.items():
        print(f"  • [{info['layer'].upper()}] {name} (Materialization: {info['materialization']}) -> Refs: {info['dependencies']}")
        
    print(f"\nDiscovered {len(inspector.tests)} Schema & Data Contract Tests:")
    for t in inspector.tests:
        print(f"  • {t['model']}.{t['column']} -> Test: {t['test_type']}")
    print(f"==========================================\n")


if __name__ == "__main__":
    run_dbt_inspection()
