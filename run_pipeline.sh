#!/bin/bash
# Full data + training pipeline — safe to run in background via nohup.
# Each step logs separately. The script exits on any failure.
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
LOG="$DIR/logs"
mkdir -p "$LOG"

cd "$DIR"

echo "=== [1/5] Data ingestion (game logs + pitcher stats) ===" | tee -a "$LOG/pipeline.log"
python data_ingestion.py 2>&1 | tee -a "$LOG/pipeline.log"

echo "" | tee -a "$LOG/pipeline.log"
echo "=== [2/5] Retrosheet SP assignments ===" | tee -a "$LOG/pipeline.log"
python fetch_sp_data.py 2>&1 | tee -a "$LOG/pipeline.log"

echo "" | tee -a "$LOG/pipeline.log"
echo "=== [3/5] Retrosheet event file parsing (in-season pitcher logs) ===" | tee -a "$LOG/pipeline.log"
python parse_retrosheet_events.py 2>&1 | tee -a "$LOG/pipeline.log"

echo "" | tee -a "$LOG/pipeline.log"
echo "=== [4/5] Feature engineering ===" | tee -a "$LOG/pipeline.log"
python feature_engineering.py 2>&1 | tee -a "$LOG/pipeline.log"

echo "" | tee -a "$LOG/pipeline.log"
echo "=== [5/5] Model training ===" | tee -a "$LOG/pipeline.log"
python train_model.py 2>&1 | tee -a "$LOG/pipeline.log"

echo "" | tee -a "$LOG/pipeline.log"
echo "=== Pipeline complete! ===" | tee -a "$LOG/pipeline.log"
date | tee -a "$LOG/pipeline.log"
