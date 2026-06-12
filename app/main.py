import uuid
import time
import logging
from fastapi import FastAPI, HTTPException
from app.schemas import PredictionRequest
from app.service import predict_cluster, MODEL_VERSION
from app.db_postgres import insert_prediction

app = FastAPI()

@app.get("/")
def health_check():
    return {"status": "healthy"}

@app.post("/predict")
def predict(request: PredictionRequest):
    start = time.time()

    try:

        features_dict = request.model_dump(exclude={"customer_id"})

        cluster,label = predict_cluster(features_dict)

        record = {
            "request_id": str(uuid.uuid4()),
            "customer_id": request.customer_id,
            **features_dict,
            "cluster": cluster,
            "label": label,
            "model_version": str(MODEL_VERSION),
            "feature_version": "rfm_v1",
            "response_time_ms": (time.time() - start) * 1000
        }

        insert_prediction(record)

        return {
            "customer_id": request.customer_id or "unknown",
            "cluster": cluster,
            "label": label        
        }
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception:
        logging.exception("Unexpected error")
        
        raise HTTPException(
            status_code=500, 
            detail="Internal server error"
        )