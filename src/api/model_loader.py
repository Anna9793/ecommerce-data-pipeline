import os
import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri(
    os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5050")
)

MODEL_URI = "models:/CustomerSegmentationModel/140"

def load_model():
    model = mlflow.sklearn.load_model(MODEL_URI)
    return model
