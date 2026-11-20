import os
import uuid
import time
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from pydantic import ValidationError
from app.schemas import (
    ClusterPredictionRequest,
    ClusterPredictionResponse,
    ClusterFeatures,
    ChurnFeatures,
    ChurnPredictionRequest,
    ChurnPredictionResponse,
    ProductAdvisorRequest,
    ProductAdvisorResponse,
)
from app.service import (
    predict_clustering_service,
    CLUSTERING_MODEL_VERSION,
    predict_churn_service,
    CHURN_MODEL_VERSION,
    CLUSTERING_FEATURE_VERSION,
    CHURN_FEATURE_VERSION,
)
import sys
from app.db_postgres import get_pool, insert_prediction, insert_churn_prediction, close_pool

# Configure application-wide logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI Lifespan Context Manager implementing 12-Factor App Factor IX (Disposability).
    Handles pre-warming connection pools on startup and clean connection teardown upon SIGTERM/shutdown.
    """
    logger.info("Starting up E-Commerce ML & AI Platform (12-Factor App Factor IX: Fast Startup)...")
    db_pool = get_pool()
    if db_pool:
        logger.info("PostgreSQL connection pool pre-warmed and ready.")
    else:
        logger.warning("PostgreSQL connection pool unavailable; operating in BigQuery/Firestore or direct fallback mode.")
    yield
    logger.info("Shutting down gracefully: closing PostgreSQL connection pools...")
    close_pool()


app = FastAPI(
    title="E-Commerce Cloud ML & Vector Search Platform",
    description="Production Cloud ML, Feature Store, and pgvector Semantic Search Platform",
    version="1.0.0",
    lifespan=lifespan
)


# ==============================================================================
# Global Exception Handlers & Request Telemetry Middleware
# ==============================================================================

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


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Translates mathematical or data validation ValueErrors into HTTP 400 Bad Request."""
    logger.warning("Validation error during %s %s: %s", request.method, request.url.path, exc)
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc)}
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catches unhandled errors across all endpoints, logs traceback, and returns HTTP 500."""
    logger.exception("Unhandled error during %s %s: %s", request.method, request.url.path, exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


# ==============================================================================
# Health & Machine Learning Inference Endpoints
# ==============================================================================

@app.get("/")
def health_check():
    return {"status": "healthy"}


@app.post("/predict/clustering", response_model=ClusterPredictionResponse, tags=["Inference"])
def predict_clustering_endpoint(request: ClusterPredictionRequest):
    customer_id = request.customer_id
    
    # 1. Resolve raw data: Online Feature Store lookup vs Direct inputs
    if customer_id:
        from app.db_postgres import get_online_features
        raw_data = get_online_features(customer_id)
        if not raw_data:
            raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found in the Feature Store.")
    else:
        raw_data = request.model_dump(exclude={"customer_id"})

    # 2. Strict Pydantic validation: ensures all 3 Cluster features exist with correct types
    try:
        features = ClusterFeatures(**raw_data)
    except ValidationError as err:
        raise HTTPException(
            status_code=422,
            detail=f"Incomplete Cluster feature set in Feature Store or payload: {err.errors()}"
        )

    # 3. Model inference with latency profiling
    start_time = time.perf_counter()
    cluster, label = predict_clustering_service(features)
    duration_ms = (time.perf_counter() - start_time) * 1000.0

    # 4. Audit persistence
    record = {
        "request_id": str(uuid.uuid4()),
        "customer_id": customer_id,
        **features.model_dump(),
        "cluster": cluster,
        "label": label,
        "model_version": str(CLUSTERING_MODEL_VERSION),
        "feature_version": str(CLUSTERING_FEATURE_VERSION),
        "response_time_ms": round(duration_ms, 2)
    }

    insert_prediction(record)

    return ClusterPredictionResponse(
        customer_id=customer_id or "unknown",
        cluster=cluster,
        label=label        
    )


@app.post("/predict/churn", response_model=ChurnPredictionResponse, tags=["Inference"])
def predict_churn_endpoint(request: ChurnPredictionRequest):
    customer_id = request.customer_id
    
    # 1. Resolve raw data: Online Feature Store lookup vs Direct inputs
    if customer_id:
        from app.db_postgres import get_online_features
        raw_data = get_online_features(customer_id)
        if not raw_data:
            raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found in the Feature Store.")
    else:
        raw_data = request.model_dump(exclude={"customer_id", "threshold"})

    # 2. Strict Pydantic validation: ensures all 6 features exist with correct types
    try:
        features = ChurnFeatures(**raw_data)
    except ValidationError as err:
        raise HTTPException(
            status_code=422,
            detail=f"Incomplete feature set in Feature Store or payload: {err.errors()}"
        )

    # 3. Model inference with latency profiling
    start_time = time.perf_counter()
    is_churn, churn_probability, applied_threshold = predict_churn_service(
        features, threshold=request.threshold
    )
    duration_ms = (time.perf_counter() - start_time) * 1000.0

    # 4. Audit persistence
    record = {
        "request_id": str(uuid.uuid4()),
        "customer_id": customer_id,
        **features.model_dump(),
        "churn_probability": churn_probability,
        "is_churn": is_churn,
        "model_version": str(CHURN_MODEL_VERSION),
        "feature_version": str(CHURN_FEATURE_VERSION),
        "response_time_ms": round(duration_ms, 2)
    }

    insert_churn_prediction(record)

    return ChurnPredictionResponse(
        customer_id=customer_id,
        churn_probability=churn_probability,
        is_churn=is_churn,
        threshold_applied=applied_threshold
    )


# ==============================================================================
# Vector Search & Semantic Product Advisor (pgvector + RAG)
# ==============================================================================


@app.post("/rag/advisor", response_model=ProductAdvisorResponse)
def product_advisor_endpoint(request: ProductAdvisorRequest):
    from app.rag_service import ProductAdvisorService
    advisor_service = ProductAdvisorService()
    return advisor_service.advise(
        query_text=request.query,
        budget_max=request.budget_max,
        top_k=request.top_k or 4
    )





# ==============================================================================
# MLOps Training, Drift Monitoring & Administration Endpoints
# ==============================================================================

@app.post("/train/churn")
def trigger_churn_retraining():
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


@app.post("/reload-models")
def reload_models():
    from app.service import reload_production_models
    reload_production_models()
    return {"status": "success", "message": "Production models reloaded successfully."}


@app.post("/simulate")
def simulate_stream_endpoint(mode: str = "standard", num_records: int = 50):
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


@app.get("/monitoring/drift")
def get_drift_report():
    from src.monitoring import calculate_feature_drift
    return calculate_feature_drift()


@app.post("/monitoring/check-and-retrain")
def check_and_retrain():
    from src.monitoring import calculate_feature_drift
    report = calculate_feature_drift()
    
    if report.get("drift_detected", False):
        logger.info("Data drift detected! Launching automated retraining pipeline...")
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
        logger.info("Features are healthy. No retraining triggered.")
        return {
            "status": "healthy",
            "message": "Features are healthy. Retraining is not required."
        }
