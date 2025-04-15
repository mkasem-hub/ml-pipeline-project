# 🏡 Build an ML Pipeline for Short-Term Rental Prices

This project is part of the **Udacity Machine Learning DevOps Engineer Nanodegree**. The objective is to build a complete ML pipeline using industry-standard tools and best practices to predict short-term rental prices in New York City using Airbnb data.

## 📦 Project Structure

```bash
.
├── main.py                         # Pipeline orchestrator using MLflow & Hydra
├── config/                         # Configuration files
├── src/                            # ML pipeline components
│   ├── basic_cleaning/
│   ├── data_check/
│   ├── data_split/
│   ├── train_random_forest/
│   └── test_regression_model/
├── outputs/                        # Output artifacts and reports
└── README.md

Tools & Frameworks
Python 3.10

MLflow – For experiment tracking and running pipeline steps

Weights & Biases (W&B) – For logging artifacts, metrics, and visualizations

Hydra – For managing configuration and parameters

Scikit-learn – For modeling and preprocessing

Pandas / NumPy / Matplotlib

🧪 ML Pipeline Steps
Download: Fetch the raw dataset from W&B

Basic Cleaning: Remove outliers and handle missing values

Data Check: Validate data quality and statistical distribution

Data Split: Split into training, validation, and test sets

Train Random Forest: Train and evaluate a regression model

Test Model: Evaluate the promoted model on a hold-out test set

## 🔗 Links

- 📁 **GitHub Repository**:  
  [https://github.com/mkasem-hub/ml-pipeline-project]

- 📊 **W&B Project Workspace**:  
  [https://wandb.ai/mostafa-kasem-a-/nyc_airbnb]

Author: Mostafa Kasem
Course: Udacity ML DevOps Engineer Nanodegree
