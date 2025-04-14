#!/usr/bin/env python
"""
This step splits the cleaned dataset into training, validation, and test sets.
"""

import os
import argparse
import logging
import pandas as pd
import wandb
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def go(args):
    run = wandb.init(job_type="data_split")
    run.config.update(vars(args))

    logger.info("Downloading cleaned dataset from W&B...")
    artifact_path = run.use_artifact(args.input_artifact).file()
    df = pd.read_csv(artifact_path)

    logger.info("Splitting dataset...")
    stratify_col = df[args.stratify_by] if args.stratify_by.lower() != 'none' else None

    # First split: train_val vs test
    train_val, test = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.random_seed,
        stratify=stratify_col
    )

    # Second split: train vs val
    train, val = train_test_split(
        train_val,
        test_size=0.2,  # 20% of train_val
        random_state=args.random_seed,
        stratify=stratify_col.loc[train_val.index] if stratify_col is not None else None
    )

    # Save to files
    os.makedirs("split_data", exist_ok=True)
    train.to_csv("split_data/train.csv", index=False)
    val.to_csv("split_data/val.csv", index=False)
    test.to_csv("split_data/test.csv", index=False)

    logger.info("Uploading split dataset to W&B...")
    artifact = wandb.Artifact(
        args.output_artifact,
        type="split_data",
        description="Training, validation, and test split"
    )
    artifact.add_dir("split_data")
    run.log_artifact(artifact)

    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split data into train, val, and test sets")

    parser.add_argument("--input_artifact", type=str, required=True, help="Name of the cleaned data artifact")
    parser.add_argument("--test_size", type=float, required=True, help="Fraction of data to use for test set")
    parser.add_argument("--random_seed", type=int, required=True, help="Random seed for splitting")
    parser.add_argument("--stratify_by", type=str, required=True, help="Column name to use for stratification")
    parser.add_argument("--output_artifact", type=str, required=True, help="Name for the output split artifact")

    args = parser.parse_args()
    go(args)