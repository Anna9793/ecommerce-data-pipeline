import os
import uuid
import time
import logging
from fastapi import FastAPI, HTTPException, BackgroundTasks
from app.schemas import PredictionRequest, ChurnPredictionRequest, ChurnPredictionResponse, ProductAdvisorRequest
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
        customer_id = request.customer_id
        
        # If any features are None, load from Online Feature Store
        if request.recency is None or request.frequency is None or request.avg_order_value is None:
            if not customer_id:
                raise HTTPException(status_code=400, detail="Missing required features and no customer_id provided.")
            
            from app.db_postgres import get_online_features
            features = get_online_features(customer_id)
            if not features:
                raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found in the Feature Store.")
            
            recency = features["recency"]
            frequency = features["frequency"]
            avg_order_value = features["avg_order_value"]
        else:
            recency = request.recency
            frequency = request.frequency
            avg_order_value = request.avg_order_value

        features_dict = {
            "recency": recency,
            "frequency": frequency,
            "avg_order_value": avg_order_value
        }

        cluster, label = predict_cluster(features_dict)

        record = {
            "request_id": str(uuid.uuid4()),
            "customer_id": customer_id,
            **features_dict,
            "cluster": cluster,
            "label": label,
            "model_version": str(MODEL_VERSION),
            "feature_version": "rfm_v1",
            "response_time_ms": (time.time() - start) * 1000
        }

        insert_prediction(record)

        return {
            "customer_id": customer_id or "unknown",
            "cluster": cluster,
            "label": label        
        }
    
    except HTTPException:
        raise
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
        customer_id = request.customer_id
        
        # If any features are None, load from Online Feature Store
        feature_fields = [
            request.recency, request.frequency, request.avg_order_value, 
            request.spending_velocity, request.cancellation_rate, request.preferred_shopping_hour
        ]
        if any(f is None for f in feature_fields):
            if not customer_id:
                raise HTTPException(status_code=400, detail="Missing required features and no customer_id provided.")
            
            from app.db_postgres import get_online_features
            features = get_online_features(customer_id)
            if not features:
                raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found in the Feature Store.")
            
            recency = features["recency"]
            frequency = features["frequency"]
            avg_order_value = features["avg_order_value"]
            spending_velocity = features["spending_velocity"]
            cancellation_rate = features["cancellation_rate"]
            preferred_shopping_hour = features["preferred_shopping_hour"]
        else:
            recency = request.recency
            frequency = request.frequency
            avg_order_value = request.avg_order_value
            spending_velocity = request.spending_velocity
            cancellation_rate = request.cancellation_rate
            preferred_shopping_hour = request.preferred_shopping_hour

        features_dict = {
            "recency": recency,
            "frequency": frequency,
            "avg_order_value": avg_order_value,
            "spending_velocity": spending_velocity,
            "cancellation_rate": cancellation_rate,
            "preferred_shopping_hour": preferred_shopping_hour
        }

        is_churn, churn_probability = predict_churn_service(features_dict)

        record = {
            "request_id": str(uuid.uuid4()),
            "customer_id": customer_id,
            **features_dict,
            "churn_probability": churn_probability,
            "is_churn": is_churn,
            "model_version": str(CHURN_MODEL_VERSION),
            "feature_version": "rfm_v1",
            "response_time_ms": (time.time() - start) * 1000
        }

        insert_churn_prediction(record)

        return ChurnPredictionResponse(
            customer_id=customer_id,
            churn_probability=churn_probability,
            is_churn=is_churn
        )
    
    except HTTPException:
        raise
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

@app.get("/predict/campaign-graph/{customer_id}")
def generate_campaign_graph_endpoint(customer_id: str):
    try:
        from app.agent_graph import MarketingGraphOrchestrator
        orchestrator = MarketingGraphOrchestrator()
        campaign = orchestrator.run(customer_id)
        return campaign
    except Exception as e:
        logging.exception("Error generating campaign via LangGraph")
        raise HTTPException(
            status_code=500,
            detail=f"LangGraph campaign generation failed: {str(e)}"
        )

@app.post("/rag/advisor")
def product_advisor_endpoint(request: ProductAdvisorRequest):
    try:
        from app.rag_service import ProductAdvisorService
        advisor_service = ProductAdvisorService()
        result = advisor_service.advise(
            query_text=request.query,
            budget_max=request.budget_max,
            top_k=request.top_k or 4
        )
        return result
    except Exception as e:
        logging.exception("Error executing product advisor RAG")
        raise HTTPException(
            status_code=500,
            detail=f"Product advisor failed: {str(e)}"
        )

@app.post("/train/churn")
def trigger_churn_retraining():
    try:
        from scripts.train_on_vertex import submit_vertex_training_job
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
        import traceback
        logging.exception("Error during transaction streaming simulation")
        return {
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
        }

@app.get("/monitoring/drift")
def get_drift_report():
    try:
        from src.monitoring import calculate_feature_drift
        report = calculate_feature_drift()
        return report
    except Exception as e:
        logging.exception("Error calculating feature drift")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to calculate feature drift: {str(e)}"
        )

@app.post("/monitoring/check-and-retrain")
def check_and_retrain():
    try:
        from src.monitoring import calculate_feature_drift
        report = calculate_feature_drift()
        
        if report.get("drift_detected", False):
            logging.info("Data drift detected! Launching automated retraining pipeline...")
            from scripts.train_on_vertex import submit_vertex_training_job
            job_name = submit_vertex_training_job()
            project_id = os.getenv("GCP_PROJECT", "anna-ml-pipeline")
            location = os.getenv("GCP_LOCATION", "us-central1")
            console_url = f"https://console.cloud.google.com/vertex-ai/pipelines/locations/{location}/runs/{job_name}?project={project_id}"
            return {
                "status": "drift_detected",
                "message": "Data drift detected! Retraining pipeline submitted successfully.",
                "job_name": job_name,
                "console_url": console_url
            }
        else:
            logging.info("Features are healthy. No retraining triggered.")
            return {
                "status": "healthy",
                "message": "Features are healthy. Retraining is not required."
            }
    except Exception as e:
        logging.exception("Error checking drift and retraining")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to check and retrain: {str(e)}"
        )