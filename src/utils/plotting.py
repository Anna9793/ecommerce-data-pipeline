import logging
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def plot_silhouette_scores(scores):
    """
    Plots Silhouette Score across different values of k.

    Args:
        scores: List of tuples (k, silhouette_score).

    Returns:
        matplotlib.figure.Figure: The rendered silhouette curve figure.
    """
    logging.info("Generating silhouette score curve")
    ks, vals = zip(*scores)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ks, vals, marker="o", linewidth=2, color="#1f77b4")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Silhouette Score")
    ax.set_title("Silhouette Score vs. Number of Clusters")
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()
    return fig


def plot_pca_clusters(df, pipeline, feature_columns):
    """
    Projects multidimensional customer features into a 2D space using PCA
    and displays the cluster separation.

    Args:
        df: DataFrame containing customer features and assigned 'cluster'.
        pipeline: Trained Scikit-Learn Pipeline.
        feature_columns: List of feature names used during training.

    Returns:
        matplotlib.figure.Figure or None: The rendered PCA scatter plot,
        or None if fewer than 2 features were used.
    """
    logging.info("Generating PCA 2D cluster visualization")
    if len(feature_columns) < 2:
        logging.warning("Skipping PCA visualization: at least 2 features required")
        return None

    # Transform features through pipeline preprocessing (log1p + scaling)
    X_transformed = pipeline[:-1].transform(df)

    pca = PCA(n_components=2)
    components = pca.fit_transform(X_transformed)
    var_ratio = pca.explained_variance_ratio_

    logging.info("PCA explained variance ratio: PC1=%.2f%%, PC2=%.2f%%", var_ratio[0] * 100, var_ratio[1] * 100)

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        components[:, 0],
        components[:, 1],
        c=df["cluster"],
        cmap="viridis",
        alpha=0.6,
        edgecolors="none"
    )

    fig.colorbar(scatter, ax=ax, label="Cluster")
    ax.set_xlabel(f"PCA Component 1 ({var_ratio[0] * 100:.1f}% var)")
    ax.set_ylabel(f"PCA Component 2 ({var_ratio[1] * 100:.1f}% var)")
    ax.set_title("Customer Clusters (PCA 2D Projection)")
    fig.tight_layout()
    return fig
