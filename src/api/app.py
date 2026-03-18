from fastapi import FastAPI
import pandas as pd
import joblib
import mlflow
from config.paths import BEST_MODEL_PATH
from src.api.schemas import PredictionRequest, PredictionResponse
from src.utils.config_loader import load_config

app = FastAPI()

model = joblib.load(BEST_MODEL_PATH)

config = load_config("config/experiment.yaml")
feature_columns = config.features.columns

@app.get("/")
def health():
    return {"status":"ok"}

@app.post("/predict",response_model=PredictionResponse)
def predict(request: PredictionRequest):

    #Convert input to Dataframe
    df = pd.DataFrame([request.features])[feature_columns]
    
    #MLflow logging
    with mlflow.start_run(run_name="prediction_run"):

        mlflow.log_param("customer_id", request.customer_id)
        
        #Predict cluster
        cluster = int(model.predict(df)[0])

        mlflow.log_metric("predicted_cluster", cluster)

    return PredictionResponse(
        customer_id=request.customer_id,
        cluster=cluster
    )