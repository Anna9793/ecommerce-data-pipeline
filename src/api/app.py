from fastapi import FastAPI
import pandas as pd
import joblib
import mlflow
from config.paths import MODEL_PATH
from src.api.schemas import PredictionRequest, PredictionResponse
from src.utils.config_loader import load_config

config = load_config("config/experiment.yaml")
feature_columns = config.features.columns

app = FastAPI()

@app.get("/")
def health():
    return {"status":"ok"}

@app.post("/predict",response_model=PredictionResponse)
def predict(request: PredictionRequest):

    #Convert input to Dataframe
    df = pd.DataFrame([request.features])[feature_columns]

    model = joblib.load(MODEL_PATH)
    
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