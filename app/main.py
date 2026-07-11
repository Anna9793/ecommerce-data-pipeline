import uuid
import time
import logging
from fastapi import FastAPI, HTTPException
from app.schemas import PredictionRequest, ChurnPredictionRequest, ChurnPredictionResponse
from app.service import predict_cluster, MODEL_VERSION, predict_churn_service, CHURN_MODEL_VERSION
from app.db_postgres import insert_prediction, insert_churn_prediction


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

@app.post("/predict/churn", response_model=ChurnPredictionResponse)
def predict_churn_endpoint(request: ChurnPredictionRequest):
    start = time.time()

    try:
        features_dict = request.model_dump(exclude={"customer_id"})

        is_churn, churn_probability = predict_churn_service(features_dict)

        record = {
            "request_id": str(uuid.uuid4()),
            "customer_id": request.customer_id,
            **features_dict,
            "churn_probability": churn_probability,
            "is_churn": is_churn,
            "model_version": str(CHURN_MODEL_VERSION),
            "feature_version": "rfm_v1",
            "response_time_ms": (time.time() - start) * 1000
        }

        insert_churn_prediction(record)

        return ChurnPredictionResponse(
            customer_id=request.customer_id,
            churn_probability=churn_probability,
            is_churn=is_churn
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception:
        logging.exception("Unexpected error during churn prediction")
        raise HTTPException(
            status_code=500, 
            detail="Internal server error"
        )