import logging
from pathlib import Path

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from config.paths import (
    RFM_CUSTOMERS,
    TRAIN_CLUSTERS,
    CUSTOMER_CLUSTERS,
    CLUSTER_PROFILE,
    BEST_MODEL_PATH,
    REPORTS_DIR
)

# BASIC STEPS

def load_rfm_data(path):
    df = pd.read_csv(path)
    logging.info("Loading RFM dataset from %s with shape %s", path, df.shape)
    return df

def split_dataset(df, random_state):
    logging.info("Splitting dataset")
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=random_state)
    return train_df, test_df

def build_pipeline(k, random_state):
    logging.info("Building pipeline")
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", KMeans(n_clusters=k, random_state=random_state))
    ])

# MODEL SELECTION

def find_best_kmeans(train_df, feature_columns, cluster_range, random_state):
    logging.info("Searching best k in range %s", cluster_range)
    best_score = -1
    best_k = None
    best_pipeline = None

    scores = []

    for k in cluster_range:
        pipeline = build_pipeline(k, random_state)

        clusters = pipeline.fit_predict(train_df[feature_columns])
        
        score = silhouette_score(train_df[feature_columns], clusters)

        scores.append((k, score))
        logging.info("k=%d → silhouette score=%.4f", k, score)

        if score > best_score:
                
                best_score = score
                best_k = k
                best_pipeline = pipeline

        logging.info("Best model: k=%d → silhouette score=%.4f",
                best_k,
                best_score)

    return best_pipeline, best_k, best_score, scores

# APPLY THE MODEL 

def assign_clusters(df, pipeline, feature_columns):
    logging.info("Assigning clusters (%d rows)", len(df))

    df = df.copy()

    df["cluster"] = pipeline.predict(df[feature_columns])

    return df

# SCORE VISUALIZATION

def plot_silhouette_scores(scores, path):
    logging.info("Visualizing silhouette scores distribution")

    ks, vals = zip(*scores)

    plt.figure()

    plt.plot(ks, vals, marker="o")

    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Silhouette score")
    plt.title("Silhouette Score vs Number of Clusters")

    plt.grid(True)

    #plt.show()

    path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(path)

    logging.info("Saved silhouette plot to %s", path)

    plt.close()

# BUSINESS INTERPRETATION 

def profile_clusters(df):
    logging.info("Generating cluster profile")

    profile = (
        df.groupby("cluster")
        .agg(
            customers=("customer_id", "count"),
            avg_recency=("recency", "mean"),
            avg_frequency=("frequency","mean"),
            avg_monetary=("monetary","mean"),
            avg_order_value=("avg_order_value","mean"),
        )
        .round(2)
    )

    return profile

def label_clusters(profile):
    logging.info("Assigning cluster labels")

    profile = profile.copy()

    # sort clusters by monetary value (ascending)
    sorted_clusters= profile.sort_values("avg_monetary").index
    n = len(sorted_clusters)

    if n == 3:
        labels = {
            sorted_clusters[0]: "Inactive",
            sorted_clusters[1]: "High ticket occassional",
            sorted_clusters[2]: "Loyal customers",
        }

    elif n == 4:
        labels = { 
            sorted_clusters[0]: "Low value",
            sorted_clusters[1]: "High ticket",
            sorted_clusters[2]: "Medium value",
            sorted_clusters[3]: "High frequency",
       }

    else:
        labels = {c: f"Segment {i}" for i, c in enumerate(sorted_clusters)}

    profile["segment"] = profile.index.map(labels)

    return profile

def build_cluster_profile(df):
    logging.info("Building cluster profile and assigning segments")

    profile = profile_clusters(df)
    profile = label_clusters(profile)

    df = df.copy()
    df["segment"] = df["cluster"].map(profile["segment"])

    return df, profile

# PCA VISUALIZATION

def visualize_clusters(df, pipeline, feature_columns, path):
    logging.info("Generating PCA cluster visualization")

    scaler = pipeline.named_steps["scaler"]
    scaled = scaler.transform(df[feature_columns])

    pca = PCA(n_components=2)
    components = pca.fit_transform(scaled)

    logging.info(
        "PCA explained variance ratio: %s",
        pca.explained_variance_ratio_
    )

    plt.figure(figsize=(8,6))

    scatter = plt.scatter(
        components[:,0],
        components[:,1],
        c=df["cluster"],
        cmap="viridis",
        alpha=0.5
    )

    plt.colorbar(scatter, label="Cluster")

    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title("Customer Clusters (PCA Projection)")

    #plt.show()

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    logging.info("Saved PCA plot to %s", path)
    plt.close()

# SAVE FUNCTIONS

def save_outputs(
    train_df, 
    test_df, 
    profile, 
    pipeline):
    
    logging.info("Saving model to %s", BEST_MODEL_PATH)
    joblib.dump(pipeline, BEST_MODEL_PATH)

    logging.info("Saving training clusters to %s", TRAIN_CLUSTERS)
    train_df.to_csv(TRAIN_CLUSTERS, index=False)

    logging.info("Saving predicted clusters to %s", CUSTOMER_CLUSTERS)
    test_df.to_csv(CUSTOMER_CLUSTERS, index=False)

    logging.info("Saving cluster profile to %s", CLUSTER_PROFILE)
    profile.to_csv(CLUSTER_PROFILE)


# MAIN PIPELINE FUNCTION

def run_clustering(config):

    logging.info("Starting clustering pipeline")

    experiment_name = "v5_remove_monetary"
    experiment_dir = REPORTS_DIR / experiment_name
    pca_path = experiment_dir / "pca_clusters.png"
    silhouette_path = experiment_dir / "silhouette_plot.png"

    feature_columns = config.features.columns
    cluster_range = config.clustering.cluster_range
    random_state = config.clustering.random_state

    df = load_rfm_data(RFM_CUSTOMERS)
    
    df = df[df["monetary"] >= 0]

    df["monetary"] = np.log1p(df["monetary"])
    df["frequency"] = np.log1p(df["frequency"])

    df["avg_order_value"] = df["monetary"] / df["frequency"].replace(0, 1)

    train_df, test_df = split_dataset(df, random_state)

    # Find best model
    best_pipeline, best_k, best_score, scores = find_best_kmeans(
        train_df,
        feature_columns,
        cluster_range,
        random_state
    )
    
    train_df = assign_clusters(train_df, best_pipeline, feature_columns)
    logging.info(
        "Cluster sizes:\n%s", 
        train_df["cluster"].value_counts()
    )

    # Plot score distribution
    plot_silhouette_scores(scores, silhouette_path)


    # Build profile
    train_df, profile = build_cluster_profile(train_df)
    logging.info ("Cluster profile:\n%s", profile)

    # Visualize clusters
    visualize_clusters(train_df, best_pipeline, feature_columns, pca_path)

    # Predict on test
    test_df = assign_clusters(test_df, best_pipeline, feature_columns)

    # Save
    save_outputs( train_df, test_df, profile, best_pipeline)

    logging.info("Clustering pipeline completed")
