import os

files = [
    "data/features.py",
    "models/trainer.py",
    "models/retrain.py",
    "scripts/run_training.py",
    "scripts/register_best_model.py",
    "api/__init__.py",
    "api/main.py",
    "api/schemas.py",
    "api/predictor.py",
    "monitoring/exporter.py",
    "monitoring/prometheus.yml",
    "dags/student_pipeline.py",
    "docker-compose.yml",
    "Dockerfile",
    "frontend/Dockerfile",
    "frontend/app.py",
    "dvc.yaml",
    "params.yaml",
    "requirements.txt",
    "docs/HLD.md",
    "docs/LLD.md",
    "docs/user_manual.md",
    "docs/test_plan.md",
    "docs/test_report.md",
    "tests/test_features.py",
    "tests/test_api.py",
    "data/processed/baseline_stats.json",
    "data/processed/student_labels.csv",
    "monitoring/grafana_dashboard.json",  # optional but good to have
]

missing = [f for f in files if not os.path.exists(f)]
print(f"Checked {len(files)} files.")
if missing:
    print("Missing files:")
    for m in missing:
        print(f"  - {m}")
else:
    print("All required files present.")
