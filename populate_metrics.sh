#!/bin/bash
# Populate metrics with many spikes/dips, shifting risk profiles
# Usage: ./populate_metrics.sh [REQUESTS] [DURATION_SECONDS] [MIN_DIP_SECS] [MAX_DIP_SECS] [BURSTS]
#   BURSTS : number of spikes (default: 20)

set -e

REQUESTS=${1:-400}
DURATION=${2:-180}
MIN_DIP=${3:-2}
MAX_DIP=${4:-8}
BURSTS=${5:-20}                # ← now configurable

echo "=== 1. Ensure all containers are up ==="
docker-compose up -d
sleep 8

echo "=== 2. Creating $BURSTS spikes + dips (${MIN_DIP}-${MAX_DIP}s) over ${DURATION}s, $REQUESTS total requests ==="

jitter() { awk -v min="$1" -v max="$2" 'BEGIN{srand(); print min+rand()*(max-min)}'; }

sent=0
cycle=0

while [ $sent -lt $REQUESTS ] && [ $cycle -lt $BURSTS ]; do
    # ---- Phase shift (High → Medium → Low) repeats every 4 bursts ----
    phase=$(( (cycle / 4) % 3 ))

    # ---- Burst size: random up to 3× the average burst size ----
    remain=$((REQUESTS - sent))
    max_bs=$(( remain / (BURSTS - cycle) * 3 ))   # scale for remaining bursts
    burst_size=$(( (RANDOM % (max_bs + 1) + 1) ))
    if [ $burst_size -lt 2 ]; then burst_size=2; fi
    if [ $((sent + burst_size)) -gt $REQUESTS ]; then
        burst_size=$((REQUESTS - sent))
    fi

    # ---- Send burst (spike) ----
    for ((b=1; b<=burst_size; b++)); do
        rand=$((RANDOM % 100))
        case $phase in
            0)   # High‑dominant
                if   [ $rand -lt 65 ]; then type="High"
                elif [ $rand -lt 90 ]; then type="Medium"
                else type="Low"; fi
                ;;
            1)   # Medium‑dominant
                if   [ $rand -lt 30 ]; then type="High"
                elif [ $rand -lt 80 ]; then type="Medium"
                else type="Low"; fi
                ;;
            2)   # Low‑dominant
                if   [ $rand -lt 15 ]; then type="High"
                elif [ $rand -lt 50 ]; then type="Medium"
                else type="Low"; fi
                ;;
        esac

        case "$type" in
            High)
                BASE_CLICKS=5;   BASE_CUM=70;   BASE_WEEKS=3; TREND=-3.0
                SCORE=25;        MISSED=3;      DAYS=12.0;  EARLY=1
                ;;
            Medium)
                BASE_CLICKS=30;  BASE_CUM=500;  BASE_WEEKS=1; TREND=-0.5
                SCORE=55;        MISSED=1;      DAYS=4.0;   EARLY=0
                ;;
            Low)
                BASE_CLICKS=120; BASE_CUM=1800; BASE_WEEKS=0; TREND=3.0
                SCORE=85;        MISSED=0;      DAYS=0.5;   EARLY=0
                ;;
        esac

        SID=$((RANDOM % 100000 + 10000))
        WEEK=$((RANDOM % 30 + 1))
        CLICKS_CURRENT=$(jitter $(echo "$BASE_CLICKS*0.7"|bc -l) $(echo "$BASE_CLICKS*1.3"|bc -l))
        CUM_CLICKS=$(jitter $(echo "$BASE_CUM*0.8"|bc -l) $(echo "$BASE_CUM*1.2"|bc -l))
        WEEKS_INACTIVE=$((BASE_WEEKS + RANDOM % 2))
        TREND_VALUE=$(jitter $(echo "$TREND-0.5"|bc -l) $(echo "$TREND+0.5"|bc -l))
        SCORE_LATEST=$(jitter $(echo "$SCORE-10"|bc -l) $(echo "$SCORE+10"|bc -l))
        SCORE_AVG=$(jitter $(echo "$SCORE-8"|bc -l) $(echo "$SCORE+8"|bc -l))
        MISSED_COUNT=$((MISSED + RANDOM % 2))
        DAYS_LATE=$(jitter $(echo "$DAYS*0.8"|bc -l) $(echo "$DAYS*1.2"|bc -l))
        GENDER_ENC=$((RANDOM % 2))
        DISABILITY_ENC=$((RANDOM % 2))
        EDUCATION_LEVEL=$((RANDOM % 5))
        IMD_SCORE=$((RANDOM % 10 + 1))
        PREV_ATTEMPTS=$((RANDOM % 3))
        CREDITS=$((RANDOM % 2 * 60 + 60))
        EARLY_UNREG=$EARLY

        curl -s -X POST http://localhost:8002/predict -H "Content-Type: application/json" -d "{
            \"student\": {
                \"id_student\": $SID,
                \"week_number\": $WEEK,
                \"weekly_clicks_current\": $CLICKS_CURRENT,
                \"cumulative_clicks\": $CUM_CLICKS,
                \"weeks_since_active\": $WEEKS_INACTIVE,
                \"click_trend_slope\": $TREND_VALUE,
                \"latest_score\": $SCORE_LATEST,
                \"avg_weighted_score\": $SCORE_AVG,
                \"missed_assessments\": $MISSED_COUNT,
                \"avg_days_late\": $DAYS_LATE,
                \"gender_enc\": $GENDER_ENC,
                \"disability_enc\": $DISABILITY_ENC,
                \"education_level\": $EDUCATION_LEVEL,
                \"imd_score\": $IMD_SCORE,
                \"num_of_prev_attempts\": $PREV_ATTEMPTS,
                \"studied_credits\": $CREDITS,
                \"early_unreg\": $EARLY_UNREG
            }
        }" > /dev/null
    done
    sent=$((sent + burst_size))
    cycle=$((cycle + 1))

    # ---- Dip (quiet period) ----
    if [ $sent -lt $REQUESTS ] && [ $cycle -lt $BURSTS ]; then
        dip=$(( (RANDOM % (MAX_DIP - MIN_DIP + 1)) + MIN_DIP ))
        sleep $dip
    fi
done

echo "$sent predictions sent in $BURSTS spikes."

# ---- Gauges and Pushgateway (unchanged) ----
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