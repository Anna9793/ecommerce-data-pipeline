import mlflow
import mlflow.pyfunc

MODEL_NAME = "CustomerSegmentationModel"
MODEL_URI = f"models:/{MODEL_NAME}/latest"

current_model = None
current_version = None

def load_model():
    global current_model, current_version

    model = mlflow.pyfunc.load_model(MODEL_URI)

    #Extract version from metada
    version = model.metadata.run_id

    if version != current_version:
        print(f"🔄Loading new model version: {version}")
        current_model = model
        current_version = version

    return current_modelmodel
