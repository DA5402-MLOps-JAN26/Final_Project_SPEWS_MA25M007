import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb
import joblib
import os
import json
from typing import Tuple, Dict, Any
from datetime import datetime

FEATURE_COLS = [
    'weekly_clicks_current', 'cumulative_clicks', 'weeks_since_active', 'click_trend_slope',
    'latest_score', 'avg_weighted_score', 'missed_assessments', 'avg_days_late',
    'gender_enc', 'disability_enc', 'education_level', 'imd_score',
    'num_of_prev_attempts', 'studied_credits', 'early_unreg'
]

class Trainer:
    def __init__(self, experiment_name: str = None):
        if experiment_name:
            mlflow.set_experiment(experiment_name)
        self.experiment_name = experiment_name

    def load_feature_matrix(self, path: str) -> Tuple[pd.DataFrame, pd.Series]:
        df = pd.read_csv(path)
        available = [c for c in FEATURE_COLS if c in df.columns]
        X = df[available].fillna(0)
        y = df['at_risk']
        return X, y

    def evaluate(self, model, X_test, y_test) -> Dict[str, float]:
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        return {
            'f1_score': float(f1_score(y_test, y_pred, average='weighted')),
            'f1_at_risk': float(f1_score(y_test, y_pred, pos_label=1)),
            'auc_roc': float(roc_auc_score(y_test, y_proba)),
            'precision': float(precision_score(y_test, y_pred, pos_label=1, zero_division=0)),
            'recall': float(recall_score(y_test, y_pred, pos_label=1, zero_division=0)),
        }

    def train_logistic_regression(self, X_train, X_test, y_train, y_test, week_num: int) -> str:
        params = {'C': 1.0, 'class_weight': 'balanced', 'max_iter': 1000,
                  'solver': 'lbfgs', 'random_state': 42}
        with mlflow.start_run(run_name=f"LogReg_week{week_num}", nested=True) as run:
            mlflow.log_params({**params, 'week_number': week_num, 'model_type': 'LogisticRegression'})
            pipe = Pipeline([('scaler', StandardScaler()),
                             ('model', LogisticRegression(**params))])
            pipe.fit(X_train, y_train)
            metrics = self.evaluate(pipe, X_test, y_test)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(pipe, "model",
                                     registered_model_name="student_dropout_logreg")
            print(f"LogReg F1={metrics['f1_at_risk']:.3f} AUC={metrics['auc_roc']:.3f}")
            return run.info.run_id

    def train_random_forest(self, X_train, X_test, y_train, y_test, week_num: int) -> str:
        params = {'n_estimators': 200, 'max_depth': 12, 'min_samples_leaf': 4,
                  'class_weight': 'balanced', 'random_state': 42, 'n_jobs': -1}
        with mlflow.start_run(run_name=f"RandomForest_week{week_num}", nested=True) as run:
            mlflow.log_params({**params, 'week_number': week_num, 'model_type': 'RandomForest'})
            model = RandomForestClassifier(**params)
            model.fit(X_train, y_train)
            metrics = self.evaluate(model, X_test, y_test)
            mlflow.log_metrics(metrics)

            # Feature importance
            fi = pd.Series(model.feature_importances_, index=X_train.columns)
            fi.sort_values(ascending=False).to_csv('/tmp/feature_importance.csv')
            mlflow.log_artifact('/tmp/feature_importance.csv')

            mlflow.sklearn.log_model(model, "model",
                                     registered_model_name="student_dropout_rf")
            print(f"RF F1={metrics['f1_at_risk']:.3f} AUC={metrics['auc_roc']:.3f}")
            return run.info.run_id

    def train_xgboost(self, X_train, X_test, y_train, y_test, week_num: int) -> str:
        scale_pos = float((y_train == 0).sum() / (y_train == 1).sum())
        params = {'n_estimators': 300, 'max_depth': 6, 'learning_rate': 0.05,
                  'scale_pos_weight': scale_pos, 'subsample': 0.8,
                  'colsample_bytree': 0.8, 'random_state': 42, 'eval_metric': 'logloss'}
        with mlflow.start_run(run_name=f"XGBoost_week{week_num}", nested=True) as run:
            mlflow.log_params({**params, 'week_number': week_num, 'model_type': 'XGBoost'})
            model = xgb.XGBClassifier(**params, verbosity=0)
            model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
            metrics = self.evaluate(model, X_test, y_test)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, "model",
                                     registered_model_name="student_dropout_xgb")
            print(f"XGBoost F1={metrics['f1_at_risk']:.3f} AUC={metrics['auc_roc']:.3f}")
            return run.info.run_id

    def train_all_models(self, feature_path: str, week_num: int) -> Dict[str, Any]:
        X, y = self.load_feature_matrix(feature_path)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)
        print(f"Training on week {week_num} | Train: {len(X_train)} | Test: {len(X_test)}")
        print(f"At-risk rate: {y.mean():.2%}")
        results = {}
        results['logreg'] = self.train_logistic_regression(X_train, X_test, y_train, y_test, week_num)
        results['rf'] = self.train_random_forest(X_train, X_test, y_train, y_test, week_num)
        results['xgb'] = self.train_xgboost(X_train, X_test, y_train, y_test, week_num)
        return results
