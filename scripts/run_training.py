import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
from data.features import build_feature_matrix
from models.trainer import Trainer

parser = argparse.ArgumentParser()
parser.add_argument("--week", type=int, default=4)
parser.add_argument("--train-xgb", type=int, default=0)
args = parser.parse_args()

WEEK_NUM = args.week
TRAIN_XGB = bool(args.train_xgb)

# Do NOT set tracking URI here; let mlflow run handle it.
# Do NOT call set_experiment; rely on the outer run's experiment.

# Load raw data
print("Loading data...")
info = pd.read_csv('data/raw/studentInfo.csv')
info = info[info['code_module'] == 'BBB'].copy()
labels = pd.read_csv('data/processed/student_labels.csv')
info = info.merge(labels[['id_student', 'at_risk']], on='id_student', how='inner')

vle = pd.read_csv('data/raw/studentVle.csv')
vle = vle[vle['code_module'] == 'BBB']
assess = pd.read_csv('data/raw/studentAssessment.csv')
assess_meta = pd.read_csv('data/raw/assessments.csv')
reg = pd.read_csv('data/raw/studentRegistration.csv')

print(f"Building features for week {WEEK_NUM}...")
X = build_feature_matrix(info, vle, assess, assess_meta, reg, WEEK_NUM)

os.makedirs('data/features', exist_ok=True)
feature_path = f'data/features/features_week_{WEEK_NUM}.csv'
X.to_csv(feature_path, index=False)
print(f"Saved features: {feature_path}")

# Train models
trainer = Trainer(experiment_name=None)  # Don't set experiment inside
X_data, y = trainer.load_feature_matrix(feature_path)
X_train, X_test, y_train, y_test = train_test_split(
    X_data, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training on week {WEEK_NUM} | Train: {len(X_train)} | Test: {len(X_test)}")
print(f"At-risk rate: {y.mean():.2%}")

# Nested runs will automatically be children of the active run created by 'mlflow run'
logreg_run_id = trainer.train_logistic_regression(X_train, X_test, y_train, y_test, WEEK_NUM)
rf_run_id = trainer.train_random_forest(X_train, X_test, y_train, y_test, WEEK_NUM)

if TRAIN_XGB:
    xgb_run_id = trainer.train_xgboost(X_train, X_test, y_train, y_test, WEEK_NUM)
    print(f"XGBoost run: {xgb_run_id}")

print(f"Training complete. LogReg run: {logreg_run_id}, RF run: {rf_run_id}")
