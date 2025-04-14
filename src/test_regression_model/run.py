#!/usr/bin/env python
"""
Load the promoted model and evaluate it on the hold-out test set
"""
import argparse
import logging
import os
import pandas as pd
import wandb
import mlflow.sklearn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def go(args):
    run = wandb.init(job_type="test_regression_model")
    run.config.update(vars(args))

    logger.info("Downloading model export artifact...")
    model_path = run.use_artifact(args.model_export).download()
    model = mlflow.sklearn.load_model(model_path)

    logger.info("Downloading test set artifact...")
    test_data_dir = run.use_artifact(args.test_set).download()
    test_df = pd.read_csv(os.path.join(test_data_dir, "test.csv"))

    logger.info("Running model predictions on test set...")
    X_test = test_df.copy()
    y_test = X_test.pop("price")
    y_pred = model.predict(X_test)

    logger.info("Calculating metrics...")
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    logger.info(f"R2: {r2}")
    logger.info(f"MAE: {mae}")
    logger.info(f"RMSE: {rmse}")

    logger.info("Logging metrics to W&B")
    run.summary["r2"] = r2
    run.summary["mae"] = mae
    run.summary["rmse"] = rmse

    run.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the promoted model on the test set")

    parser.add_argument(
        "--model_export",
        type=str,
        help="Fully qualified name of the exported model artifact (e.g. random_forest_export:latest)",
        required=True
    )

    parser.add_argument(
        "--test_set",
        type=str,
        help="Fully qualified name of the test dataset artifact (e.g. split_data.csv:latest)",
        required=True
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="Name of the output artifact to log results (optional)",
        required=False
    )

    args = parser.parse_args()
    go(args)