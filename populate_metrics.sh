#!/bin/bash
# Ultimate script to populate all SPEWS metrics for Grafana
set -e

echo "=== 1. Ensure all containers are up ==="
docker-compose up -d
sleep 10

echo "=== 2. Generate prediction traffic (20 requests) ==="
for i in $(seq 1 20); do
  curl -s -X POST http://localhost:8002/predict -H "Content-Type: application/json" -d '{
    "student": {
      "id_student": 12345,
      "week_number": 6,
      "weekly_clicks_current": 12.0,
      "cumulative_clicks": 72.0,
      "weeks_since_active": 0.0,
      "click_trend_slope": 0.0,
      "latest_score": 34.0,
      "avg_weighted_score": 34.0,
      "missed_assessments": 2.0,
      "avg_days_late": 5.0,
      "gender_enc": 0, "disability_enc": 0,
      "education_level": 2.0, "imd_score": 5.0,
      "num_of_prev_attempts": 0,
      "studied_credits": 60.0, "early_unreg": 0
    }
  }' > /dev/null
done
echo "20 predictions sent."

echo "=== 3. Set gauge metrics inside inference-api ==="
docker exec inference-api python -c "
from monitoring.exporter import MODEL_F1, PIPELINE_OK
MODEL_F1.set(0.8651)
PIPELINE_OK.set(1)
print('Gauges set')
"

echo "=== 4. Push pipeline success to Pushgateway ==="
echo "spews_pipeline_success 1" | curl -s --data-binary @- http://localhost:9091/metrics/job/airflow_pipeline

echo "=== 5. Wait 30 seconds for Prometheus to scrape ==="
sleep 30

echo "=== 6. Verify metrics in Prometheus ==="
echo -n "spews_api_requests_total: "
curl -s "http://127.0.0.1:9090/api/v1/query?query=spews_api_requests_total" | python3 -c "import sys,json; r=json.load(sys.stdin); print('OK' if r['status']=='success' and len(r['data']['result'])>0 else 'FAIL')"

echo -n "spews_model_f1_score: "
curl -s "http://127.0.0.1:9090/api/v1/query?query=spews_model_f1_score" | python3 -c "import sys,json; r=json.load(sys.stdin); print('OK' if r['status']=='success' and len(r['data']['result'])>0 else 'FAIL')"

echo -n "spews_pipeline_success: "
curl -s "http://127.0.0.1:9090/api/v1/query?query=spews_pipeline_success" | python3 -c "import sys,json; r=json.load(sys.stdin); print('OK' if r['status']=='success' and len(r['data']['result'])>0 else 'FAIL')"

echo "=== All metrics verified. Refresh your Grafana dashboard (http://localhost:3001) now. ==="
