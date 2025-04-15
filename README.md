# 🏡 Build an ML Pipeline for Short-Term Rental Prices in NYC

This project is part of the **Udacity Machine Learning DevOps Engineer Nanodegree**.

The goal is to develop a **reproducible, scalable machine learning pipeline** that predicts Airbnb rental prices in New York City. The pipeline covers the full ML lifecycle: from data ingestion to model training, testing, and evaluation, using **MLflow**, **Hydra**, and **Weights & Biases (W&B)**.

---

## 🚀 Project Overview

This ML pipeline emulates a production-grade environment using MLOps principles:

✅ **Modularized Components**: Each step in the pipeline is a standalone MLflow project.  
✅ **Trackable Experiments**: Leveraging W&B for artifact and metric logging.  
✅ **Reproducibility & Versioning**: GitHub releases and MLflow parameters ensure consistency.  
✅ **Scalable Configurations**: Using Hydra to dynamically manage parameters.

---

## 🧪 Pipeline Workflow

| Step                  | Description                                                                 |
|-----------------------|-----------------------------------------------------------------------------|
| `download`            | Downloads the raw Airbnb dataset from W&B                                  |
| `basic_cleaning`      | Cleans data by removing outliers, converting dates, and filtering location  |
| `data_check`          | Runs data validation tests (price ranges, row counts, etc.)                 |
| `data_split`          | Splits cleaned data into training, validation, and test sets                |
| `train_random_forest` | Trains a Random Forest model using Scikit-learn pipeline                    |
| `test_regression_model` | Evaluates model on test data, calculating R², MAE, and RMSE                |

---

## 📦 Project Structure

```bash
.
├── main.py                         # Orchestrates the ML pipeline using Hydra + MLflow
├── config/                         # Experiment and pipeline configuration files
├── src/                            # Component-specific logic
│   ├── basic_cleaning/
│   ├── data_check/
│   ├── data_split/
│   ├── train_random_forest/
│   └── test_regression_model/
├── outputs/                        # Model artifacts, plots, reports
└── README.md

##🛠️ Tools & Technologies

Python 3.10
MLflow – Experiment tracking and pipeline orchestration
Weights & Biases (W&B) – Artifact logging and performance visualization
Hydra – Dynamic configuration management
Scikit-learn – Machine learning models and preprocessing pipelines
Pandas / NumPy / Matplotlib

##📌 GitHub Releases

Two official versions were tagged and released to demonstrate pipeline versioning:
🔖 v1.0.0 – Initial pipeline without NYC bounds filtering
🔖 v1.0.1 – Enhanced cleaning: added min/max price filter and NYC geo-bounds

##📊 Results Summary

The pipeline achieves:
✅ Proper preprocessing and outlier removal
✅ Effective stratified data splitting
✅ Tracked MAE, RMSE, and R² metrics in W&B
✅ Feature importance plots logged for interpretability

##🔗 Project Links

📁 GitHub Repository:
https://github.com/mkasem-hub/ml-pipeline-project
📊 Weights & Biases Dashboard:
https://wandb.ai/mostafa-kasem-a-/nyc_airbnb

##👤 Author
Mostafa Kasem

