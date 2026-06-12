import logging
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from src.features.transformers import LogTransformer
from src.features.selection import ColumnSelector
from src.features.to_numpy import ToNumpy

from config.paths import (
    RFM_CUSTOMERS,
    TRAIN_CLUSTERS,
    CUSTOMER_CLUSTERS,
    CLUSTER_PROFILE,
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

def build_pipeline(k, random_state, feature_columns):
    logging.info("Building pipeline")
    return Pipeline([
        ("selector", ColumnSelector(columns=feature_columns)),
        ("log", LogTransformer(columns=["frequency", "avg_order_value"])),
        ("to_numpy", ToNumpy()),
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
        pipeline = build_pipeline(k, random_state, feature_columns)

        clusters = pipeline.fit_predict(train_df)

        X_transformed = pipeline[:-1].transform(train_df)
        score = silhouette_score(X_transformed, clusters)
        
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

def plot_silhouette_scores(scores):
    logging.info("Visualizing silhouette scores distribution")

    ks, vals = zip(*scores)

    fig, ax = plt.subplots()

    ax.plot(ks, vals, marker="o")

    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Scores")
    ax.set_title("Silhouette Scores")

    return fig

# BUSINESS INTERPRETATION 

def profile_clusters(df):
    logging.info("Generating cluster profile")

    profile = (
        df.groupby("cluster")
        .agg(
            customers=("customer_id", "count"),
            avg_recency=("recency", "mean"),
            avg_frequency=("frequency","mean"),
            avg_order_value=("avg_order_value","mean"),
        )
        .round(2)
    )

    return profile

def label_clusters(profile):
    logging.info("Assigning cluster labels")

    profile = profile.copy()

    labels = {}

    for cluster_id, row in profile.iterrows():

        recency = row["avg_recency"]
        frequency = row["avg_frequency"]
        value = row ["avg_order_value"]

        if recency > 150:
            labels[cluster_id] = "Inactive Customers"

        elif frequency > 10:
            labels[cluster_id] = "Frequent Buyers"

        elif value > 500:
            labels[cluster_id] = "Occasional High Value Buyers"

        else:
            labels[cluster_id] = "Medium Customers"

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

def visualize_clusters(df, pipeline, feature_columns):
    logging.info("Generating PCA cluster visualization")

    if len(feature_columns) < 2:
        logging.info("Skipping PCA: only one feature")
        return None

    X_transformed = pipeline[:-1].transform(df)

    pca = PCA(n_components=2)
    components = pca.fit_transform(X_transformed)

    logging.info(
        "PCA explained variance ratio: %s",
        pca.explained_variance_ratio_
    )

    fig, ax = plt.subplots(figsize=(8,6))

    scatter = ax.scatter(
        components[:,0],
        components[:,1],
        c=df["cluster"],
        cmap="viridis",
        alpha=0.5
    )

    fig.colorbar(scatter, ax=ax, label="Cluster")

    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.set_title("Customer Clusters (PCA Projection)")

    return fig

# SAVE FUNCTIONS

def save_outputs(
    train_df, 
    test_df, 
    profile, 
    pipeline):
    

    logging.info("Saving training clusters to %s", TRAIN_CLUSTERS)
    train_df.to_csv(TRAIN_CLUSTERS, index=False)

    logging.info("Saving predicted clusters to %s", CUSTOMER_CLUSTERS)
    test_df.to_csv(CUSTOMER_CLUSTERS, index=False)

    logging.info("Saving cluster profile to %s", CLUSTER_PROFILE)
    profile.to_csv(CLUSTER_PROFILE)


# MAIN PIPELINE FUNCTION

def run_clustering(
    config, 
    df=None,
    feature_columns=None,
    cluster_range=None):

    logging.info("Starting clustering pipeline")

    if cluster_range is None: 
        cluster_range = config.clustering.cluster_range
    random_state = config.clustering.random_state

    if df is None:
        df = load_rfm_data(RFM_CUSTOMERS)

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
    fig = plot_silhouette_scores(scores)


    # Build profile
    train_df, profile = build_cluster_profile(train_df)
    logging.info ("Cluster profile:\n%s", profile)

    # Visualize clusters
    pca_fig = visualize_clusters(train_df, best_pipeline, feature_columns)

    # Predict on test
    test_df = assign_clusters(test_df, best_pipeline, feature_columns)

    # Save
    save_outputs( train_df, test_df, profile, best_pipeline)

    logging.info("Clustering pipeline completed")

    return best_pipeline, best_k, {"silhouette": best_score}, fig, pca_fig
