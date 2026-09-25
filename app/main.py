import os
import uuid
import time
import logging
from fastapi import FastAPI, HTTPException, BackgroundTasks
from app.schemas import (
    PredictionRequest,
    ChurnPredictionRequest,
    ChurnPredictionResponse,
    ProductAdvisorRequest,
    ProductAdvisorResponse,
    TwoTowerRecommendationRequest,
)
from app.service import (
    predict_cluster,
    MODEL_VERSION,
    predict_churn_service,
    CHURN_MODEL_VERSION,
    RFM_FEATURE_VERSION,
    CHURN_FEATURE_VERSION,
)
from contextlib import asynccontextmanager
from fastapi import Request
from config.logging_config import setup_logging
from app.db_postgres import insert_prediction, insert_churn_prediction, close_pool

# Initialize 12-Factor Event Stream Logger (Factor XI)
logger = setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI Lifespan Context Manager implementing 12-Factor App Factor IX (Disposability).
    Handles graceful resource initialization and clean connection teardown upon SIGTERM/shutdown.
    """
    logger.info("Starting up E-Commerce ML & AI Platform (12-Factor App Factor IX: Fast Startup)...")
    yield
    logger.info("Shutting down gracefully: closing PostgreSQL connection pools...")
    close_pool()


app = FastAPI(
    title="E-Commerce ML & Agentic GenAI Platform",
    description="Enterprise Multi-Tenant Data, MLOps and Agentic GenAI Recommendation Platform",
    version="1.0.0",
    lifespan=lifespan
)


@app.middleware("http")
async def log_requests_middleware(request: Request, call_next):
    """
    Structured HTTP Request Logging Middleware (12-Factor App Factor XI: Logs as Event Streams).
    Records method, path, response status, and processing duration for Cloud Logging observability.
    """
    start_time = time.time()
    response = await call_next(request)
    duration_ms = (time.time() - start_time) * 1000.0

    logger.info(
        f"{request.method} {request.url.path} completed with status {response.status_code} in {duration_ms:.2f}ms",
        extra={
            "http_method": request.method,
            "http_path": request.url.path,
            "status_code": response.status_code,
            "latency_ms": round(duration_ms, 2),
            "client_ip": request.client.host if request.client else "unknown"
        }
    )
    return response


@app.get("/")
def health_check():
    return {"status": "healthy"}

@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        customer_id = request.customer_id
        
        # 1. Resolve features: Direct inputs vs Online Feature Store lookup
        if request.recency is not None and request.frequency is not None and request.avg_order_value is not None:
            features = {
                "recency": request.recency,
                "frequency": request.frequency,
                "avg_order_value": request.avg_order_value
            }
        elif customer_id:
            from app.db_postgres import get_online_features
            features = get_online_features(customer_id)
            if not features:
                raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found in the Feature Store.")
        else:
            raise HTTPException(status_code=400, detail="Missing required features and no customer_id provided.")

        cluster, label = predict_cluster(features)

        record = {
            "request_id": str(uuid.uuid4()),
            "customer_id": customer_id,
            "recency": features["recency"],
            "frequency": features["frequency"],
            "avg_order_value": features["avg_order_value"],
            "cluster": cluster,
            "label": label,
            "model_version": str(MODEL_VERSION),
            "feature_version": str(RFM_FEATURE_VERSION)
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
    try:
        customer_id = request.customer_id
        
        # 1. Resolve features: Direct inputs vs Online Feature Store lookup
        feature_fields = [
            request.recency, request.frequency, request.avg_order_value, 
            request.spending_velocity, request.cancellation_rate, request.preferred_shopping_hour
        ]
        if all(f is not None for f in feature_fields):
            features = {
                "recency": request.recency,
                "frequency": request.frequency,
                "avg_order_value": request.avg_order_value,
                "spending_velocity": request.spending_velocity,
                "cancellation_rate": request.cancellation_rate,
                "preferred_shopping_hour": request.preferred_shopping_hour
            }
        elif customer_id:
            from app.db_postgres import get_online_features
            features = get_online_features(customer_id)
            if not features:
                raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found in the Feature Store.")
        else:
            raise HTTPException(status_code=400, detail="Missing required features and no customer_id provided.")

        is_churn, churn_probability = predict_churn_service(features)

        record = {
            "request_id": str(uuid.uuid4()),
            "customer_id": customer_id,
            "recency": features["recency"],
            "frequency": features["frequency"],
            "avg_order_value": features["avg_order_value"],
            "spending_velocity": features.get("spending_velocity", 1.0),
            "cancellation_rate": features.get("cancellation_rate", 0.0),
            "preferred_shopping_hour": features.get("preferred_shopping_hour", 12),
            "churn_probability": churn_probability,
            "is_churn": is_churn,
            "model_version": str(CHURN_MODEL_VERSION),
            "feature_version": str(CHURN_FEATURE_VERSION)
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

@app.post("/rag/advisor", response_model=ProductAdvisorResponse)
def product_advisor_endpoint(request: ProductAdvisorRequest):
    try:
        from app.rag_service import ProductAdvisorService
        advisor_service = ProductAdvisorService()
        result = advisor_service.advise(
            query_text=request.query,
            budget_max=request.budget_max,
            top_k=request.top_k or 4,
            tenant_id=request.tenant_id or "giftshop_uk"
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

@app.get("/recommend/two-tower/{customer_id}")
def recommend_two_tower_get(customer_id: str, top_k: int = 4):
    try:
        from app.two_tower_service import TwoTowerRecommenderService
        service = TwoTowerRecommenderService()
        return service.recommend_for_customer(customer_id=customer_id, top_k=top_k)
    except Exception as e:
        logging.exception("Error generating Two-Tower recommendations")
        raise HTTPException(
            status_code=500,
            detail=f"Two-Tower recommendation failed: {str(e)}"
        )

@app.post("/recommend/two-tower")
def recommend_two_tower_post(request: dict):
    try:
        from app.two_tower_service import TwoTowerRecommenderService
        service = TwoTowerRecommenderService()
        customer_id = request.get("customer_id", "custom_user")
        top_k = int(request.get("top_k", 4))
        return service.recommend_for_customer(
            customer_id=customer_id,
            custom_features=request,
            top_k=top_k
        )
    except Exception as e:
        logging.exception("Error generating Two-Tower custom recommendations")
        raise HTTPException(
            status_code=500,
            detail=f"Two-Tower custom recommendation failed: {str(e)}"
        )