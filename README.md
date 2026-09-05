# 🛒 Enterprise E-Commerce Data & MLOps Platform

[![CI/CD Pipeline](https://github.com/Anna9793/ecommerce-data-pipeline/actions/workflows/deploy.yml/badge.svg)](https://github.com/Anna9793/ecommerce-data-pipeline/actions)
[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![Unit Tests](https://img.shields.io/badge/tests-60%2F60%20passing-brightgreen.svg)]()
[![Cloud](https://img.shields.io/badge/GCP-Cloud%20Run%20%7C%20BigQuery%20%7C%20Vertex%20AI%20%7C%20Dataproc-orange.svg)](https://cloud.google.com/)
[![IaC](https://img.shields.io/badge/IaC-Terraform-623CE4.svg)](https://www.terraform.io/)
[![Big Data](https://img.shields.io/badge/Big%20Data-PySpark%20%7C%20Dataproc-E25A1C.svg)](https://spark.apache.org/)
[![Streaming](https://img.shields.io/badge/Streaming-Pub%2FSub%20%7C%20Dataflow%20(Beam)-FF6F00.svg)](https://cloud.google.com/dataflow)
[![Orchestration](https://img.shields.io/badge/Orchestration-Airflow%20%7C%20Composer-017CEE.svg)](https://airflow.apache.org/)
[![Packaging](https://img.shields.io/badge/Packaging-pyproject.toml%20%7C%20uv-DE5FE9.svg)](https://github.com/astral-sh/uv)
[![API Gateway](https://img.shields.io/badge/Ingress-Cloud%20API%20Gateway%20%7C%20OpenAPI-009688.svg)](https://cloud.google.com/api-gateway)
[![Multi-Tenant](https://img.shields.io/badge/Architecture-Multi--Tenant%20%7C%20Shopify%20Adapter-8A2BE2.svg)]()
[![AI Agents](https://img.shields.io/badge/GenAI-LangGraph%20%7C%20pgvector%20%7C%20Gemini-4285F4.svg)](https://cloud.google.com/vertex-ai)

A production-grade, end-to-end **Data Engineering, MLOps, and Agentic GenAI Platform** built on Google Cloud Platform (GCP). The platform automates customer segmentation, predictive churn scoring, statistical drift detection, closed-loop model retraining, event-driven streaming ingestion, multi-agent marketing campaign generation, and hybrid semantic product search.

---

## 🏛️ System Architecture

```mermaid
graph TD
    %% Ingress & API Gateway Layer
    subgraph Ingress_Layer [1. Edge Security & Ingress Layer]
        Clients[Multi-Tenant Clients: Shopify / Mobile / Web] -->|HTTPS with API Key| Gateway["Google Cloud API Gateway<br/>(OpenAPI 3.0 Contract & Rate Limiting)"]
    end

    %% Schema Normalization & Adapter Layer
    subgraph Adapter_Layer [2. Schema Normalization & Adapter Factory]
        Gateway --> Factory["SchemaAdapterFactory<br/>(ShopifyAdapter, UciRetailAdapter, OlistAdapter)"]
        Factory --> Canonical["CanonicalTransaction (Pydantic Universal Contract)"]
    end

    %% Streaming Ingestion & Processing Layer
    subgraph Ingestion_Stream [3. Streaming Ingestion & ETL Layer]
        Canonical --> Topic[("GCP Pub/Sub: retail-transactions-topic")]
        Topic -->|DLQ Policy: 5 Retries| PubSub_DLQ[("Pub/Sub DLQ: dead-letter-topic")]
        Topic -->|Native Subscription| BQ_Raw[("BigQuery: retail_data.transactions")]
        Topic -->|Stream Pull| Beam["Apache Beam on Dataflow<br/>(5-Min Windowing & Velocity)"]
        Beam -->|Tagged Output: DLQ| Beam_DLQ[("Dataflow DLQ Logs")]
        Beam -->|Windowed Features| BQ_Agg[("BigQuery: streaming_customer_aggregates")]
    end

    %% Master Orchestration Layer
    subgraph Orchestration_Layer [4. Master Enterprise Orchestrator (Cloud Composer / Airflow)]
        Airflow["Airflow Master DAG (Daily @ 00:00 UTC)<br/>1. Sensors → 2. Data Quality → 3. PySpark on Dataproc<br/>4. Sync Feature Store & pgvector → 5. K-S Drift Check"]
    end

    %% Distributed Big Data Feature Engineering
    subgraph BigData_Engine [5. Distributed PySpark Batch Engine (GCP Dataproc)]
        Airflow --> Dataproc["Dataproc Ephemeral Cluster<br/>(PySpark Windowing, RFM, 30d/90d Velocity, Spot VMs)"]
        Dataproc -->|Parquet & BigQuery Connector| BQ_RFM[("BigQuery: rfm_features View")]
    end

    %% Serving & Storage Layer
    subgraph Serving_Layer [6. Low-Latency Serving & Feature Store]
        Gateway -->|Reverse Proxy /v1/*| API[FastAPI on Cloud Run]
        UI[Streamlit Dashboard UI] <-->|REST API| API
        API <-->|Sub-15ms Key-Value Lookup| FS[("Online Feature Store: Firestore / PostgreSQL")]
        API -->|Analytical Queries| BQ_RFM
    end

    %% Agentic GenAI & Hybrid RAG
    subgraph GenAI_Engine [7. LangGraph Autonomous Multi-Agent & RAG]
        API --> LangGraph["LangGraph StateMachine<br/>(Analyst → Strategist → Copywriter → Critic)"]
        LangGraph -->|Rejection Feedback Loop| LangGraph
        LangGraph -->|Approved Campaign| UI
        
        UI -->|Natural Language Query| RAG_API["POST /rag/advisor"]
        RAG_API --> Embed["Vertex AI text-embedding-004"]
        Embed -->|768d Vector Query| PG_Vec[("PostgreSQL pgvector: HNSW Index")]
        PG_Vec -->|Budget & Distance Filter| Gemini["Gemini 1.5 Flash (RAG)"]
        Gemini -->|Product Justifications| UI
    end

    %% Closed-Loop MLOps & Retraining
    subgraph MLOps_Retraining [8. Closed-Loop MLOps & Retraining]
        Airflow -->|If Drift Detected p < 0.05| Vertex["Vertex AI Pipelines (Kubeflow/KFP)"]
        Vertex -->|Parallel Tasks| Train["Train XGBoost & KMeans"]
        Train --> Gate{"F1 Evaluation Gate"}
        Gate -->|Approved| GCS[("GCS Model Registry Bucket")]
        GCS -->|In-Memory Hot Reload| API
    end

    %% Infrastructure as Code
    subgraph IaC_Layer [9. Infrastructure as Code & CI/CD]
        TF["Terraform (IaC Modules: Dataproc, BigQuery, GCS, Cloud Run, Pub/Sub, Dataflow, Composer, API Gateway)"] --> GCP_Cloud["Google Cloud Infrastructure"]
        GHA["GitHub Actions CI/CD (OIDC Workload Identity Federation + 60 Tests)"] --> CloudRun_Deploy["Zero-Downtime Cloud Run Deployment"]
    end

    classDef stream fill:#FF6F00,stroke:#333,stroke-width:2px,color:#fff;
    classDef pubsub fill:#FBBC04,stroke:#333,stroke-width:2px,color:#000;
    classDef airflow fill:#017CEE,stroke:#333,stroke-width:2px,color:#fff;
    classDef spark fill:#E25A1C,stroke:#333,stroke-width:2px,color:#fff;
    classDef gw fill:#009688,stroke:#333,stroke-width:2px,color:#fff;
    classDef gcp fill:#4285F4,stroke:#333,stroke-width:2px,color:#fff;
    classDef ai fill:#34A853,stroke:#333,stroke-width:2px,color:#fff;
    classDef tf fill:#623CE4,stroke:#333,stroke-width:2px,color:#fff;
    class Clients,Gateway gw;
    class Factory,Canonical gw;
    class Topic,PubSub_DLQ pubsub;
    class Beam stream;
    class Airflow airflow;
    class Dataproc spark;
    class BQ_Raw,BQ_Agg,FS,BQ_RFM,GCS,API,UI gcp;
    class LangGraph,Gemini,RAG_API,Embed,PG_Vec ai;
    class TF,GHA tf;
```

---

## 🗺️ 23-Phase Architectural Roadmap

| Phase | Category | Description | Key Technologies |
| :---: | :--- | :--- | :--- |
| **01** | **Data Engineering** | Data exploration, outlier cleaning, and RFM feature calculation on UCI Online Retail. | `Pandas`, `NumPy`, `Scipy` |
| **02** | **Machine Learning** | Unsupervised customer segmentation with K-Means, LogTransforms & StandardScaler. | `Scikit-Learn`, `KMeans` |
| **03** | **Machine Learning** | Predictive Churn modeling with calibrated probabilities and feature interpretability. | `LogisticRegression`, `XGBoost` |
| **04** | **Software Eng.** | Microservices containerization and interactive Streamlit UI dashboard. | `FastAPI`, `Streamlit`, `Docker` |
| **05** | **MLOps** | Centralized experiment tracking, hyperparameter logging, and Model Registry. | `MLflow`, `SQLite` |
| **06** | **Cloud Data** | Cloud migration to Google BigQuery data warehouse and Google Cloud Storage. | `BigQuery`, `GCS`, `GCP` |
| **07** | **Cloud Serving** | Serverless API serving on Cloud Run with zero downtime in-RAM model hot-reloading. | `Google Cloud Run`, `FastAPI` |
| **08** | **DevOps & CI/CD** | Automated CI/CD pipeline using Google Workload Identity Federation (keyless OIDC). | `GitHub Actions`, `Artifact Registry` |
| **09** | **Orchestration** | Distributed model training and evaluation gates in Kubeflow / Vertex AI Pipelines. | `Vertex AI Pipelines`, `KFP` |
| **10** | **Monitoring** | Statistical Data & Feature Drift detection via 2-sample Kolmogorov-Smirnov tests. | `Scipy.stats`, `Plotly`, `BigQuery` |
| **11** | **Automation** | Closed-loop automated model evaluation cron job triggering serverless retraining. | `Cloud Scheduler`, `OIDC` |
| **12** | **Feature Store** | Dual-backend low-latency Online Feature Store (<15ms key-value queries). | `Firestore (NoSQL)`, `PostgreSQL` |
| **13** | **Multi-Agent GenAI**| 4-Agent collaborative marketing assembly line with Pydantic JSON schema constraints. | `Vertex AI Gemini`, `Pydantic` |
| **14** | **Vector Search / RAG**| Contextual multi-attribute semantic search and conversational advisor with HNSW index. | `pgvector`, `HNSW`, `text-embedding-004` |
| **15** | **Agentic Workflows** | Stateful multi-agent graph with automated self-correcting Critic feedback loops. | `LangGraph`, `StateGraph` |
| **16** | **IaC (Terraform)** | Full declarative provisioning of all GCP datasets, buckets, Cloud Run, and IAM roles. | `Terraform (IaC)`, `HCL` |
| **17** | **Event Streaming** | Decoupled event-driven streaming ingestion with fan-out Pub/Sub subscriptions. | `Google Cloud Pub/Sub` |
| **18** | **Streaming ETL** | Real-time sliding window aggregations and Dual-Level Dead Letter Queues (DLQ). | `Apache Beam`, `Google Dataflow` |
| **19** | **Master Orchestrator** | Master enterprise workflow orchestration, Data Quality gates, and drift triggers. | `Apache Airflow`, `Cloud Composer` |
| **20** | **API Ingress** | Secure edge ingress, OpenAPI contract, rate limiting, and API key authorization. | `Google Cloud API Gateway`, `OpenAPI` |
| **21** | **Multi-Tenancy** | Universal Canonical Data Model, Shopify/Olist adapters, and multi-store UI. | `Pydantic`, `Schema Adapters` |
| **22** | **Modern Packaging & Fast Dependencies** | Modern packaging standard with `pyproject.toml` (PEP 517/621) and Rust-powered `uv` package resolver. | `uv`, `pyproject.toml`, `Docker Multi-Stage` |
| **23** | **Distributed Big Data Feature Engineering** | Scalable batch customer feature computation and windowing on Google Cloud Dataproc. | `Apache Spark`, `PySpark`, `GCP Dataproc` |

---

## 💻 Interactive Streamlit Dashboard Features

The application provides a unified UI in Streamlit ([streamlit_app.py](file:///Users/Anna/ecommerce-data-pipeline/streamlit_app.py)) with 5 specialized tabs:

1.  📊 **Customer Segmentation**: Interactive 2D/3D PCA visualization of customer clusters with dynamic spending and frequency sliders.
2.  🔮 **Churn Prediction**: Real-time churn probability scoring with customer ID lookup directly from the Online Feature Store.
3.  🤖 **AI Marketing Copilot**: Multi-agent marketing generator with live engine selection between **LangGraph (with self-correction loops)** and **Linear Assembly Line**.
4.  📈 **Model Monitoring & Drift**: Visual histograms of live feature distributions vs. baseline with automated **Kolmogorov-Smirnov p-value evaluation** and one-click retraining triggers.
5.  🛍️ **Product Advisor (RAG)**: Conversational shopping assistant powered by **PostgreSQL `pgvector`** semantic search and Gemini 1.5.

---

## 🛠️ Tech Stack & Tooling

*   **Cloud Platform (GCP)**: BigQuery, Cloud Run, Cloud Storage, Cloud Firestore, Cloud Scheduler, Google Pub/Sub, Google Dataflow, Vertex AI, Artifact Registry.
*   **Machine Learning & MLOps**: Scikit-Learn, XGBoost, MLflow, Kubeflow Pipelines (KFP), Scipy (Kolmogorov-Smirnov).
*   **Generative AI & LLMOps**: LangGraph, LangChain, Google Vertex AI (Gemini 1.5 Flash, `text-embedding-004`), Pydantic v2.
*   **Databases & Vector Engines**: PostgreSQL 15, `pgvector` (HNSW indexing), Google Cloud Firestore.
*   **Big Data & Batch Processing**: Apache Spark (PySpark), Google Cloud Dataproc (Ephemeral & Autoscaling Clusters).
*   **Streaming & Processing**: Apache Beam (Python SDK), Google Cloud Dataflow, Google Cloud Pub/Sub.
*   **Packaging & Dependency Management**: `pyproject.toml` (PEP 517/621), `uv` (Astral Rust package resolver).
*   **Infrastructure as Code & CI/CD**: Terraform (`>= 1.5.0`), GitHub Actions (OIDC authentication), Docker Multi-Stage.

---

## 🚀 Quickstart & Local Execution

### 1. Prerequisites
*   Python 3.9+
*   Docker & Docker Compose
*   Google Cloud SDK (`gcloud`)

### 2. Local Environment Setup
```bash
# Clone repository
git clone https://github.com/Anna9793/ecommerce-data-pipeline.git
cd ecommerce-data-pipeline

# Option A: Modern setup with uv (Recommended - 10x-100x faster)
uv venv
uv pip install -e ".[dev,ml]"

# Option B: Traditional pip setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Launch local PostgreSQL (pgvector), MLflow, and services
docker compose up -d
```

### 3. Run Automated Unit Tests (60 Passing)
```bash
PYTHONPATH=. pytest
```

### 4. Launch Application
```bash
# Terminal 1: FastAPI Backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2: Streamlit Dashboard
streamlit run streamlit_app.py --server.port=8501
```

---

## 📖 Deep-Dive Documentation

For detailed technical specifications, benchmark comparisons, and design rationales (*Pydantic vs. Regex, FinOps Scale-to-Zero, HNSW Indexing, Dual-Level DLQ*), please refer to:
👉 **[ARCHITECTURE.md](file:///Users/Anna/ecommerce-data-pipeline/ARCHITECTURE.md)**
