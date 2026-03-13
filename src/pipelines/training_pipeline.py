import logging

from src.cleaning import run_cleaning
from src.transformation import run_transformation
from src.rfm_features import run_rfm_features
from src.clustering import run_clustering

def run_training_pipeline(config):

    logging.info("Start training pipeline")

    run_cleaning()
    run_transformation()
    run_rfm_features()
    run_clustering(config)

    logging.info("Training pipeline completed")