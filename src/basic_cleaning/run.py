#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning,
exporting the result to a new artifact.
"""

import argparse
import logging
import pandas as pd
import wandb

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def go(args):
    run = wandb.init(job_type="basic_cleaning")
    run.config.update(vars(args))

    logger.info("Downloading artifact from W&B...")
    artifact_local_path = run.use_artifact(args.input_artifact).file()

    logger.info("Reading CSV file...")
    df = pd.read_csv(artifact_local_path)

    logger.info(f"Initial data shape: {df.shape}")

    # Remove outliers
    idx = df['price'].between(args.min_price, args.max_price)
    df = df[idx].copy()

    # Convert last_review to datetime
    df['last_review'] = pd.to_datetime(df['last_review'], errors='coerce')

    # Remove rows outside NYC bounds
    idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)
    df = df[idx].copy()

    # Save cleaned data
    output_file = "clean_sample.csv"
    df.to_csv(output_file, index=False)
    logger.info(f"Saved cleaned data to {output_file}, shape: {df.shape}")

    # Log artifact to W&B
    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file(output_file)
    run.log_artifact(artifact)
    logger.info("Artifact logged to W&B")

    run.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This step cleans the data")

    parser.add_argument("--input_artifact", type=str, help="Input artifact name (sample.csv:latest)", required=True)
    parser.add_argument("--output_artifact", type=str, help="Name for the cleaned dataset artifact", required=True)
    parser.add_argument("--output_type", type=str, help="Type of the output artifact", required=True)
    parser.add_argument("--output_description", type=str, help="Description for the output artifact", required=True)
    parser.add_argument("--min_price", type=float, help="Minimum acceptable price", required=True)
    parser.add_argument("--max_price", type=float, help="Maximum acceptable price", required=True)

    args = parser.parse_args()
    go(args)