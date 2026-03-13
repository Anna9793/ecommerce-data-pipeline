import pandas as pd
import logging
from pathlib import Path

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from config.paths import (RFM_CUSTOMERS, TRAIN_CLUSTERS, CUSTOMER_CLUSTERS, 
PREDICTED_CLUSTERS, CLUSTER_PROFILE, CLUSTER_PLOT, BEST_MODEL_PATH)
from src.utils.config_loader import load_config

config = load_config("config/experiment.yaml")
cluster_range = config.clustering.cluster_range
random_state = config.clustering.random_state
feature_columns = config.features.columns

def load_rfm_data(path):
    logging.info("Loading RFM dataset from %s", path)
    df = pd.read_csv(path)
    return df

def split_dataset(df, random_state):
    logging.info("Splitting dataset into training and prediction sets")
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=random_state)
    return train_df, test_df

def build_pipeline(k, random_state):

    logging.info("Building ML pipeline")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", KMeans(n_clusters=k, random_state=random_state))
    ])

    return pipeline

def train_pipeline(pipeline, df):

    logging.info("Training clustering pipeline")

    clusters = pipeline.fit_predict(df[feature_columns])

    return pipeline, clusters

def assign_clusters(df, clusters):

    logging.info("Assigning cluster labels")

    df["cluster"] = clusters

    return df

def profile_clusters(df):

    logging.info("Generating cluster profile")

    profile = (
        df.groupby("cluster")
        .agg(
            customers=("customer_id", "count"),
            avg_recency=("recency", "mean"),
            avg_frequency=("frequency","mean"),
            avg_monetary=("monetary","mean")
        )
        .round(2)
    )

    return profile

def label_clusters(profile):

    logging.info("Assigning human-readable cluster labels")

    profile = profile.copy()

    # sort clusters by monetary value only - temporary solution
    sorted_clusters = profile.sort_values("avg_monetary")

    labels = {
        sorted_clusters.index[0]: "Inactive customers",
        sorted_clusters.index[1]: "Occassional buyers",
        sorted_clusters.index[2]: "Loyal customers",
        sorted_clusters.index[3]: "VIP customers"
    }

    profile["segment"] = profile.index.map(labels)

    return profile

def visualize_clusters(df, pipeline, path):

    logging.info("Generating PCA cluster visualization")

    scaler = pipeline.named_steps["scaler"]

    scaled_features = scaler.transform(df[feature_columns])

    pca = PCA(n_components=2)

    components = pca.fit_transform(scaled_features)

    plt.figure(figsize=(8,6))

    scatter = plt.scatter(
        components[:,0],
        components[:,1],
        c=df["cluster"],
        cmap="viridis",
        alpha=0.5
    )

    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title("Customer Clusters (PCA Projection)")

    plt.colorbar(scatter, label="Cluster")

    path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(path)

    plt.close()

def save_clustered_data(df, path):

    logging.info("Saving clustered dataset to %s", path)

    df.to_csv(path, index=False)


def save_cluster_profile(profile, path):

    logging.info("Saving cluster profile to %s", path)

    path.parent.mkdir(parents=True, exist_ok=True)

    profile.to_csv(path)


def save_model(pipeline, path):

    logging.info("Saving model artifact to %s", path)

    path.parent.mkdir(parents=True, exist_ok=True)

    artifact = {
        "pipeline": pipeline,
        "features": ["recency", "frequency", "monetary"],
        "model_type": "k_means_customer_segmentation",
        "version": "1.0"
    }

    joblib.dump(artifact, path)

def run_clustering(config, config_path=None):
    
    config_name = Path(config_path).stem if config_path else "default"
    
    df = load_rfm_data(RFM_CUSTOMERS)

    train_df, test_df = split_dataset(df, random_state)

    best_score = -1
    best_k = None

    for k in cluster_range:
        
        pipeline = build_pipeline(k, random_state)

        run_name = f"kmeans_k{k}_{config_name}"

        config_name = config_name.replace(".yaml","")

        with mlflow.start_run(run_name=run_name):

            if config_path:
                mlflow.set_tag("experiment_config", config_path)
                mlflow.log_artifact(config_path, artifact_path="config")
        
            mlflow.log_param("cluster_range", str(config.clustering.cluster_range))
            mlflow.log_param("random_state", config.clustering.random_state)

            mlflow.log_param("n_clusters", k)

            pipeline, clusters = train_pipeline(pipeline, train_df)

            score = silhouette_score(train_df[feature_columns], clusters)

            mlflow.log_metric("silhouette_score", float(score))

            mlflow.sklearn.log_model(
            pipeline, 
            artifact_path="kmeans_pipeline",
            input_example=train_df[feature_columns].head(5),
            registered_model_name="CustomerSegmentationModel"
            )

            if score > best_score:
                
                best_score = score
                best_k = k
    
    logging .info(
        "Best model uses %s clusters with score %.4f",
        best_k,
        best_score
    )   

    joblib.dump(pipeline, BEST_MODEL_PATH)

    train_df = assign_clusters(train_df, clusters)

    profile = profile_clusters(train_df)

    profile = label_clusters(profile)

    train_df["segment"] = train_df["cluster"].map(profile["segment"])

    save_clustered_data(train_df, TRAIN_CLUSTERS)
    
    save_cluster_profile(profile, CLUSTER_PROFILE)

    visualize_clusters(train_df, pipeline, CLUSTER_PLOT)

    features = test_df[feature_columns]

    predictions = pipeline.predict(features)

    test_df = assign_clusters(test_df, predictions)

    save_clustered_data(test_df, PREDICTED_CLUSTERS)

if __name__ == "__main__":
    run_clustering(config)
