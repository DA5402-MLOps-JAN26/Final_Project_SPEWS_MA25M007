#!/usr/bin/env python3
"""
Reproducible training of the production XGBoost model using all weeks 4-10.
Run with: python scripts/retrain_robust.py
"""
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score
import xgboost as xgb
import os

# Configuration – change these if needed
TRACKING_URI = "file:///Users/aswinis_mac/Documents/1.IIT_Madras_IMSC/2Sem/MLOps/Project/Final_Project_SPEWS_MA25M007/mlruns"
EXPERIMENT_NAME = "student_dropout_risk"
MODEL_NAME = "student_dropout_best"
ALIAS = "production"
WEEKS = [4, 5, 6, 7, 8, 9, 10]

def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Load and combine data
    dfs = []
    for w in WEEKS:
        df = pd.read_csv(f'data/features/features_week_{w}.csv')
        dfs.append(df)
    X_all = pd.concat(dfs, ignore_index=True)
    y = X_all['at_risk']

    # Drop id_student and target
    feature_cols = [c for c in X_all.columns if c not in ['id_student', 'at_risk']]
    X = X_all[feature_cols]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # Handle class imbalance
    scale_pos = float((y_train == 0).sum() / (y_train == 1).sum())

    # Train model with regularization
    model = xgb.XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        scale_pos_weight=scale_pos, subsample=0.8, colsample_bytree=0.8,
        reg_lambda=2.0, reg_alpha=0.5,
        random_state=42, verbosity=0, eval_metric='logloss'
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred, pos_label=1)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
    print(f'Model F1: {f1:.4f}, AUC: {auc:.4f}')

    # Log and register
    with mlflow.start_run(run_name='Robust_Generalized_Model') as run:
        mlflow.log_params({'weeks': WEEKS, 'features': feature_cols})
        mlflow.log_metrics({'f1_at_risk': f1, 'auc_roc': auc})
        mlflow.sklearn.log_model(model, 'model', registered_model_name=MODEL_NAME)

        client = MlflowClient()
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        latest = max(versions, key=lambda v: int(v.version))
        client.set_registered_model_alias(MODEL_NAME, ALIAS, latest.version)
        print(f'Alias @{ALIAS} set to version {latest.version}')

if __name__ == '__main__':
    main()
