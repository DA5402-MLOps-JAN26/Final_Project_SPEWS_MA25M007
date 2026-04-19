import mlflow
from mlflow.tracking import MlflowClient
import os

# Set tracking URI from environment or default
tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "file://./mlruns")
mlflow.set_tracking_uri(tracking_uri)

client = MlflowClient()
experiment_name = "student_dropout_risk"

# Get or create experiment
experiment = client.get_experiment_by_name(experiment_name)
if not experiment:
    print(f"Experiment '{experiment_name}' not found. Creating it...")
    experiment_id = client.create_experiment(experiment_name)
else:
    experiment_id = experiment.experiment_id

# Search all runs in the experiment, order by f1_at_risk descending
runs = client.search_runs(
    experiment_ids=[experiment_id],
    order_by=["metrics.f1_at_risk DESC"]
)

if not runs:
    print("No runs found.")
    exit(1)

best_run = runs[0]
run_id = best_run.info.run_id
f1 = best_run.data.metrics.get('f1_at_risk', 0.0)
print(f"Best run: {run_id} with F1@risk={f1:.4f}")

# Register or update the 'student_dropout_best' model
model_name = "student_dropout_best"
try:
    client.create_registered_model(model_name)
    print(f"Created registered model '{model_name}'")
except Exception:
    print(f"Registered model '{model_name}' already exists, creating new version.")

# Create a new version from the best run
result = client.create_model_version(
    name=model_name,
    source=f"runs:/{run_id}/model",
    run_id=run_id
)
version = result.version
print(f"Created model version {version}")

# Transition to Production
client.transition_model_version_stage(
    name=model_name,
    version=version,
    stage="Production"
)
print(f"Model {model_name} version {version} promoted to Production")
