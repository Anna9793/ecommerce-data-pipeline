# 🛒 Enterprise E-Commerce Cloud ML & Vector Search Platform

[![CI/CD Pipeline](https://github.com/Anna9793/ecommerce-data-pipeline/actions/workflows/deploy.yml/badge.svg)](https://github.com/Anna9793/ecommerce-data-pipeline/actions)
[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![Unit Tests](https://img.shields.io/badge/tests-43%2F43%20passing-brightgreen.svg)]()
[![Cloud](https://img.shields.io/badge/GCP-Cloud%20Run%20%7C%20BigQuery-orange.svg)](https://cloud.google.com/)
[![Vector Search](https://img.shields.io/badge/Vector%20Search-PostgreSQL%20%7C%20pgvector-34A853.svg)](https://github.com/pgvector/pgvector)
[![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn%20%7C%20XGBoost-EE4C2C.svg)](https://scikit-learn.org/)
[![Container](https://img.shields.io/badge/Container-Docker-2496ED.svg)](https://www.docker.com/)

A production-grade, end-to-end **Cloud Data, Machine Learning, and Semantic Vector Search Platform** built on Google Cloud Platform (GCP). The platform automates customer segmentation, predictive churn scoring, low-latency online feature serving, statistical drift monitoring, and AI-powered product discovery with pgvector.

---

## 🏛️ System Architecture

```mermaid
graph TD
    %% Client & Interface Layer
    subgraph UI_Layer["1. Client and Interface Layer"]
        User["End User / Business Analyst"] --> UI["Streamlit Dashboard (Cloud Run)"]
        ClientApp["E-Commerce Clients"] --> API["FastAPI Serving Engine (Cloud Run)"]
        UI -->|"REST API Requests"| API
    end

    %% Storage and Feature Layer
    subgraph Data_Storage["2. Cloud Storage and Feature Store"]
        API -->|"Analytical Queries"| BigQuery["Google BigQuery (Data Warehouse)"]
        API -->|"Sub-15ms Online Lookup"| FeatureStore["Online Feature Store (PostgreSQL)"]
        API -->|"Vector Similarity Search"| PGVector["pgvector (Catalog Embeddings)"]
    end

    %% Machine Learning & AI
    subgraph ML_AI["3. Machine Learning and Semantic AI"]
        API --> KMeans["K-Means Segmentation (Customer Personas)"]
        API --> Churn["XGBoost / Logistic Regression (Churn Risk)"]
        API --> Gemini["Vertex AI / Gemini (RAG Product Advisor)"]
        PGVector -->|"Semantic Top-K Context"| Gemini
    end

    %% CI/CD and Containerization
    subgraph DevOps_Layer["4. DevOps and CI/CD Automation"]
        GHA["GitHub Actions CI/CD"] -->|"Automated PyTests"| DockerBuild["Docker Multi-Stage Build"]
        DockerBuild -->|"Zero-Downtime Deployment"| API
        DockerBuild -->|"Zero-Downtime Deployment"| UI
    end

    classDef gcp fill:#4285F4,stroke:#333,stroke-width:2px,color:#fff;
    classDef ai fill:#34A853,stroke:#333,stroke-width:2px,color:#fff;
    classDef devops fill:#623CE4,stroke:#333,stroke-width:2px,color:#fff;
    classDef ui fill:#FF6F00,stroke:#333,stroke-width:2px,color:#fff;

    class UI,API ui;
    class BigQuery,FeatureStore,PGVector gcp;
    class KMeans,Churn,Gemini ai;
    class GHA,DockerBuild devops;
```

---

## 🌟 Core Platform Pillars

| Pillar | Category | Description | Key Technologies |
| :---: | :--- | :--- | :--- |
| **01** | **Data Engineering & RFM** | Data preprocessing, outlier detection, and Recency-Frequency-Monetary (RFM) feature calculation. | `Pandas`, `NumPy`, `Scipy` |
| **02** | **Customer Segmentation** | Unsupervised customer persona clustering with K-Means, LogTransforms, and StandardScaler. | `Scikit-Learn`, `KMeans` |
| **03** | **Predictive Churn ML** | Supervised churn prediction modeling with calibrated probabilities and risk scoring. | `LogisticRegression`, `XGBoost` |
| **04** | **Data Warehouse** | Scalable analytical storage for millions of e-commerce transactions and point-in-time time-travel queries. | `Google BigQuery` |
| **05** | **Online Feature Store** | Low-latency (<15ms) key-value feature serving for real-time customer inference. | `PostgreSQL`, `psycopg2.pool` |
| **06** | **Vector Search & RAG** | Conversational product advisor using pgvector embeddings and Google Gemini reasoning. | `pgvector`, `Vertex AI`, `Gemini` |
| **07** | **MLOps & Drift Monitoring** | Automated statistical data drift evaluation using two-sample Kolmogorov-Smirnov (K-S) tests. | `Scipy.stats`, `Plotly` |
| **08** | **Serverless Serving** | Containerized microservices on Cloud Run with zero-downtime model hot-reloading. | `FastAPI`, `Cloud Run`, `Docker` |
| **09** | **CI/CD Quality Gates** | Keyless automated testing and deployment workflow using Workload Identity Federation. | `GitHub Actions`, `Docker Buildx` |

---

## 💻 Interactive Streamlit Dashboard

The platform includes a modern Streamlit web application ([streamlit_app.py](file:///Users/Anna/ecommerce-data-pipeline/streamlit_app.py)) featuring 4 dedicated analytical modules:

1. 📊 **Customer Segmentation**: Interactive 2D/3D PCA visualization of customer clusters with dynamic spending and frequency sliders.
2. 🔮 **Churn Prediction**: Real-time customer churn probability scoring with instant profile lookup from the Online Feature Store.
3. 🛍️ **Product Advisor (RAG & pgvector)**: Conversational shopping assistant with real-time budget filtering and similarity matching.
4. 📈 **Model Monitoring & Drift**: Statistical drift diagnostics comparing live BigQuery distributions against training baselines with BigQuery Time Travel explorer.

---

## 🛠️ Tech Stack

* **Cloud Infrastructure (GCP)**: Google BigQuery, Cloud Run, Cloud Storage, Vertex AI, Artifact Registry.
* **Machine Learning & MLOps**: Scikit-Learn, XGBoost, Scipy (Kolmogorov-Smirnov), MLflow.
* **Vector Database & Feature Store**: PostgreSQL 15, `pgvector` (HNSW indexing), `psycopg2` Threaded Connection Pools.
* **Serving & API**: FastAPI, Uvicorn, Pydantic v2, Streamlit, Plotly.
* **Packaging & DevOps**: `requirements.txt`, Docker Multi-Stage, GitHub Actions CI/CD.

---

## 🚀 Quickstart & Local Execution

### 1. Prerequisites
* Python 3.9+
* Docker & Docker Compose
* Google Cloud SDK (`gcloud`)

### 2. Local Environment Setup
```bash
# Clone repository
git clone https://github.com/Anna9793/ecommerce-data-pipeline.git
cd ecommerce-data-pipeline

# Virtual environment setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Launch local PostgreSQL (with pgvector) and local services
docker compose up -d
```

### 3. Run Automated Unit Tests (43 Passing)
```bash
pytest tests/
```

### 4. Launch Application
```bash
# Terminal 1: FastAPI Backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2: Streamlit Dashboard
streamlit run streamlit_app.py --server.port=8501
```
