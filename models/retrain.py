# models/retrain.py 
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score
import xgboost as xgb
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.features import build_feature_matrix

MLFLOW_URI = "http://127.0.0.1:5000"
MODEL_NAME = "student_dropout_best"
ALIAS = "production"

FEATURE_COLS = [
    'weekly_clicks_current', 'cumulative_clicks', 'weeks_since_active', 'click_trend_slope',
    'latest_score', 'avg_weighted_score', 'missed_assessments', 'avg_days_late',
    'gender_enc', 'disability_enc', 'education_level', 'imd_score',
    'num_of_prev_attempts', 'studied_credits', 'early_unreg'
]

def get_production_metrics() -> dict:
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = MlflowClient()
    try:
        version = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
        run = client.get_run(version.run_id)
        return {
            'f1': run.data.metrics.get('f1_at_risk', 0.0),
            'auc': run.data.metrics.get('auc_roc', 0.0),
            'version': version.version
        }
    except Exception as e:
        print(f"Could not fetch production metrics: {e}")
        return {'f1': 0.0, 'auc': 0.0, 'version': '0'}

def incremental_retrain(week_num: int, force: bool = False) -> dict:
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("student_dropout_risk")
    mlflow.set_experiment("student_dropout_risk")
    client = MlflowClient()

    print(f"\n=== Incremental Retrain - Week {week_num} ===")

    # Load feature matrix (already built in Day 4 for weeks 4-10)
    feat_path = f"data/features/features_week_{week_num}.csv"
    if not os.path.exists(feat_path):
        print(f"Building feature matrix for week {week_num}...")
        info = pd.read_csv('data/processed/student_labels.csv')
        vle = pd.read_csv('data/raw/studentVle.csv')
        vle = vle[vle['code_module'] == 'BBB']
        assess = pd.read_csv('data/raw/studentAssessment.csv')
        assess_meta = pd.read_csv('data/raw/assessments.csv')
        reg = pd.read_csv('data/raw/studentRegistration.csv')
        X = build_feature_matrix(info, vle, assess, assess_meta, reg, week_num)
        os.makedirs('data/features', exist_ok=True)
        X.to_csv(feat_path, index=False)
    else:
        X = pd.read_csv(feat_path)

    available = [c for c in FEATURE_COLS if c in X.columns]
    Xf = X[available].fillna(0)
    y = X['at_risk']

    X_train, X_test, y_train, y_test = train_test_split(
        Xf, y, test_size=0.2, random_state=42, stratify=y)

    current = get_production_metrics()
    print(f"Current @production: F1={current['f1']:.4f} v{current['version']}")

    # Train XGBoost (best performer from Day 4)
    scale_pos = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        scale_pos_weight=scale_pos, subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbosity=0, eval_metric='logloss'
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    new_f1 = float(f1_score(y_test, y_pred, pos_label=1))
    new_auc = float(roc_auc_score(y_test, y_proba))
    print(f"New model: F1={new_f1:.4f} AUC={new_auc:.4f}")

    with mlflow.start_run(run_name=f"Incremental_week{week_num}") as run:
        mlflow.log_params({'week_number': week_num, 'model_type': 'XGBoost_incremental',
                           'train_size': len(X_train), 'n_features': len(available)})
        mlflow.log_metrics({'f1_at_risk': new_f1, 'auc_roc': new_auc,
                            'f1_improvement': new_f1 - current['f1']})
        mlflow.sklearn.log_model(model, 'model', registered_model_name=MODEL_NAME)

        # Promote if improvement OR forced
        if new_f1 > current['f1'] or force:
            versions = client.search_model_versions(f"name='{MODEL_NAME}'")
            latest = max(versions, key=lambda v: int(v.version))
            client.set_registered_model_alias(MODEL_NAME, ALIAS, latest.version)
            print(f"Promoted v{latest.version} to @{ALIAS}")
        else:
            print(f"No improvement - keeping existing @{ALIAS}")

    return {'week': week_num, 'f1': new_f1, 'auc': new_auc, 'improved': new_f1 > current['f1']}

if __name__ == '__main__':
    week = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    result = incremental_retrain(week, force=(len(sys.argv) > 2))
    print("Result:", result)