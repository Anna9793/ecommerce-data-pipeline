import logging
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

from src.cleaning import run_cleaning
from src.transformation import run_transformation
from src.rfm_features import (
    run_rfm_features,
    load_rfm
)
from src.clustering import run_clustering

def run_training_pipeline(config):

    logging.info("Start training pipeline")

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("customer_segmentation")

    for exp_name, exp_config in config.experiments.items():
        
        with mlflow.start_run():
            
            print(f"\n Running {exp_name}")

            feature_columns = exp_config.features

            run_cleaning()
            run_transformation()
            run_rfm_features()

            df = load_rfm()

            model, best_k, metrics, fig, pca_fig = run_clustering(
                config, 
                df=df,
                feature_columns=feature_columns,
                cluster_range=config.clustering.cluster_range
                )

            input_example = df[feature_columns].head(5)
            predictions = model.predict(input_example)
            signature = infer_signature(
                input_example,
                predictions
            )

            mlflow.sklearn.log_model(
                sk_model= model, 
                name="model",
                input_example=input_example,
                signature=signature)

            mlflow.log_metric("silhouette_score", metrics["silhouette"])

            mlflow.log_figure(fig, "silhouette_plot.png")
            if pca_fig is not None:
                mlflow.log_figure(pca_fig, "pca_clusters.png")
            
        
            mlflow.log_param("feature_set", ",".join(feature_columns))
            mlflow.log_param("n_clusters", best_k)

            mlflow.register_model(
            model_uri=f"runs:/{mlflow.active_run().info.run_id}/model",
            name="customer_segmentation_model"
        )

    logging.info("Training pipeline completed")