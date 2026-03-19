import mlflow
from pathlib import Path

mlflow.set_tracking_uri(f"file:{Path('mlruns').resolve()}")
mlflow.set_experiment("debug_test")

with mlflow.start_run():
    print("MLFLOW TEST RUN")
    mlflow.log_param("test_param", 123)
    mlflow.log_metric("test_metric", 0.99)