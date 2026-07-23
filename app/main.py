import os
import uuid
import time
import logging
from fastapi import FastAPI, HTTPException, BackgroundTasks
from app.schemas import PredictionRequest, ChurnPredictionRequest, ChurnPredictionResponse
from app.service import predict_cluster, MODEL_VERSION, predict_churn_service, CHURN_MODEL_VERSION
from app.db_postgres import insert_prediction, insert_churn_prediction
from scripts.train_on_vertex import submit_vertex_training_job


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

@app.get("/predict/campaign/{customer_id}")
def generate_campaign_endpoint(customer_id: str):
    try:
        from app.agent_service import MarketingAgentService
        agent_service = MarketingAgentService()
        campaign = agent_service.generate_marketing_campaign(customer_id)
        return campaign
    except Exception as e:
        logging.exception("Error generating campaign")
        raise HTTPException(
            status_code=500,
            detail=f"Campaign generation failed: {str(e)}"
        )

@app.post("/train/churn")
def trigger_churn_retraining():
    try:
        job_name = submit_vertex_training_job()
        project_id = os.getenv("GCP_PROJECT", "anna-ml-pipeline")
        location = os.getenv("GCP_LOCATION", "us-central1")
        console_url = f"https://console.cloud.google.com/vertex-ai/pipelines/locations/{location}/runs/{job_name}?project={project_id}"
        return {
            "status": "success",
            "message": "Vertex AI pipeline run submitted successfully.",
            "job_name": job_name,
            "console_url": console_url
        }
    except Exception as e:
        logging.exception("Error triggering Vertex AI training job")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to submit training job: {str(e)}"
        )

@app.post("/reload-models")
def reload_models():
    try:
        from app.service import reload_production_models
        reload_production_models()
        return {"status": "success", "message": "Production models reloaded successfully."}
    except Exception as e:
        logging.exception("Error reloading production models")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to reload models: {str(e)}"
        )

@app.post("/simulate")
def simulate_stream_endpoint(mode: str = "standard", num_records: int = 50):
    try:
        from scripts.simulate_stream import generate_mock_transactions, insert_transactions_to_bq
        project_id = os.getenv("GCP_PROJECT", "anna-ml-pipeline")
        
        if os.getenv("USE_BIGQUERY", "false").lower() != "true":
            return {"status": "success", "message": f"Local simulation mode active (mocked {num_records} records)."}
            
        rows = generate_mock_transactions(mode=mode, num_records=num_records)
        num_inserted = insert_transactions_to_bq(rows, project_id=project_id)
        return {
            "status": "success",
            "message": f"Successfully streamed {num_inserted} transactions to BigQuery in {mode} mode."
        }
    except Exception as e:
        logging.exception("Error during transaction streaming simulation")
        raise HTTPException(
            status_code=500,
            detail=f"Streaming simulation failed: {str(e)}"
        )