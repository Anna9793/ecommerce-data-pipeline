import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc


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


def plot_feature_importance(model, feature_names):
    """
    Plots horizontal bar chart of feature coefficients (linear models) or importances (tree models).

    Args:
        model: Trained scikit-learn estimator or Pipeline with named step 'model'.
        feature_names: List of feature names.

    Returns:
        matplotlib.figure.Figure or None: The rendered bar chart.
    """
    logging.info("Generating feature importance plot")
    fitted = model.named_steps["model"] if hasattr(model, "named_steps") and "model" in model.named_steps else model

    if hasattr(fitted, "coef_"):
        importances = fitted.coef_[0]
        title = "Feature Coefficients (Logistic Regression)"
    elif hasattr(fitted, "feature_importances_"):
        importances = fitted.feature_importances_
        title = "Feature Importances"
    else:
        logging.warning("Model does not expose coef_ or feature_importances_")
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    y_pos = range(len(feature_names))
    ax.barh(y_pos, importances, align="center", color="#2ca02c", alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_names)
    ax.invert_yaxis()
    ax.set_xlabel("Weight / Importance")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()
    return fig


def plot_confusion_matrix(y_true, y_pred, display_labels=("Active (0)", "Churn (1)")):
    """
    Plots confusion matrix for binary churn classification.

    Args:
        y_true: True binary labels (0 or 1).
        y_pred: Predicted binary labels (0 or 1).
        display_labels: Tuple of class display labels.

    Returns:
        matplotlib.figure.Figure: The rendered confusion matrix heatmap.
    """
    logging.info("Generating confusion matrix plot")
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)

    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title("Customer Churn Confusion Matrix")
    fig.tight_layout()
    return fig


def plot_roc_curve(y_true, y_prob):
    """
    Plots Receiver Operating Characteristic (ROC) curve with AUC score.

    Args:
        y_true: True binary labels (0 or 1).
        y_prob: Predicted churn probabilities (scores for class 1).

    Returns:
        matplotlib.figure.Figure or None: The rendered ROC curve figure,
        or None if y_true contains fewer than 2 distinct classes.
    """
    logging.info("Generating ROC curve plot")
    if len(np.unique(y_true)) < 2:
        logging.warning("Skipping ROC curve plot: at least 2 distinct classes required in y_true")
        return None

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color="#1f77b4", lw=2, label=f"ROC Curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="#7f7f7f", lw=1.5, linestyle="--", label="Random Chance (AUC = 0.500)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Recall / Sensitivity)")
    ax.set_title("Receiver Operating Characteristic (ROC) Curve")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()
    return fig

