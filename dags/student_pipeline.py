# dags/student_pipeline.py 
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import json
import os
import sys
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

default_args = {
    'owner': 'mlops_student',
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
}

def ingest_data(**context):
    week_num = context['dag_run'].conf.get('week_num', 4)
    vle = pd.read_csv('data/raw/studentVle.csv')
    vle = vle[vle['code_module'] == 'BBB']
    vle['week'] = vle['date'] // 7
    current = vle[vle['week'] <= week_num]
    os.makedirs('data/processed', exist_ok=True)
    current.to_csv('data/processed/current_vle.csv', index=False)
    print(f'Ingested {len(current)} rows for week {week_num}')
    return week_num

def validate_schema(**context):
    df = pd.read_csv('data/processed/current_vle.csv')
    required_cols = ['id_student', 'id_site', 'date', 'sum_click', 'week']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f'Missing columns: {missing}')
    null_pct = df.isnull().sum() / len(df)
    high_null = null_pct[null_pct > 0.3]
    if len(high_null) > 0:
        print(f'WARNING: High null rate in columns: {high_null.to_dict()}')
    print(f'Schema valid. Rows: {len(df)}, Cols: {df.shape[1]}')

def clean_data(**context):
    df = pd.read_csv('data/processed/current_vle.csv')
    orig_len = len(df)
    df = df.dropna(subset=['id_student', 'sum_click'])
    p99 = df['sum_click'].quantile(0.99)
    df = df[df['sum_click'] <= p99]
    df.to_csv('data/processed/cleaned_vle.csv', index=False)
    print(f'Cleaned: {orig_len} -> {len(df)} rows (removed {orig_len - len(df)})')

def compute_drift(**context):
    with open('data/processed/baseline_stats.json') as f:
        baselines = json.load(f)
    df = pd.read_csv('data/processed/cleaned_vle.csv')
    def psi(expected_mean, expected_std, actual_vals):
        actual_mean = actual_vals.mean()
        if expected_std == 0: return 0.0
        z_shift = abs(actual_mean - expected_mean) / expected_std
        return min(z_shift / 5.0, 1.0)
    psi_clicks = psi(baselines['vle_sum_click']['mean'], baselines['vle_sum_click']['std'], df['sum_click'])
    context['task_instance'].xcom_push(key='psi_score', value=psi_clicks)
    print(f"PSI score for sum_click: {psi_clicks:.4f}")
    return psi_clicks

def should_retrain(**context):
    psi = context['task_instance'].xcom_pull(task_ids='compute_drift', key='psi_score') or 0.0
    print(f"PSI={psi:.4f} — threshold=0.2")
    return 'trigger_retraining' if float(psi) > 0.2 else 'skip_retraining'

def trigger_retraining(**context):
    print("Drift detected — triggering retraining pipeline")

def log_pipeline_metrics(**context):
    psi = context['task_instance'].xcom_pull(task_ids='compute_drift', key='psi_score') or 0.0
    metrics = {'psi_score': psi, 'pipeline_status': 'success', 'timestamp': datetime.now().isoformat()}
    os.makedirs('data/processed', exist_ok=True)
    with open('data/processed/pipeline_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Pipeline metrics logged: {metrics}")
    psi = context['task_instance'].xcom_pull(task_ids='compute_drift', key='psi_score') or 0.0
    # Update Prometheus metrics via Pushgateway
    registry = CollectorRegistry()
    pipeline_success = Gauge('spews_pipeline_success', 'Last Airflow run success', registry=registry)
    pipeline_success.set(1)  # Task only runs on success path
    psi_gauge = Gauge('spews_drift_psi_score', 'PSI drift score', ['feature'], registry=registry)
    psi_gauge.labels(feature='sum_click').set(psi)
    try:
        push_to_gateway('pushgateway:9091', job='airflow_pipeline', registry=registry)
        print("Metrics pushed to Pushgateway")
    except Exception as e:
        print(f"Failed to push metrics: {e}")

with DAG(
    'student_weekly_pipeline',
    default_args=default_args,
    start_date=datetime(2026, 4, 12),
    schedule='@weekly',
    catchup=False,
    description='Student dropout risk weekly pipeline',
) as dag:

    t1 = PythonOperator(task_id='ingest_data', python_callable=ingest_data)
    t2 = PythonOperator(task_id='validate_schema', python_callable=validate_schema)
    t3 = PythonOperator(task_id='clean_data', python_callable=clean_data)
    t4 = PythonOperator(task_id='compute_drift', python_callable=compute_drift)
    t5 = BranchPythonOperator(task_id='should_retrain', python_callable=should_retrain)
    t6 = PythonOperator(task_id='trigger_retraining', python_callable=trigger_retraining)
    t7 = EmptyOperator(task_id='skip_retraining')
    t8 = PythonOperator(task_id='log_pipeline_metrics', python_callable=log_pipeline_metrics,
                        trigger_rule='none_failed_min_one_success')

    t1 >> t2 >> t3 >> t4 >> t5 >> [t6, t7] >> t8