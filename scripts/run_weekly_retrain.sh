#!/bin/zsh
set -euo pipefail

PROJECT_DIR="/Users/dougboehm/Baseball Code/win_probability"
PYTHON="/Users/dougboehm/opt/anaconda3/bin/python"
LOG_DIR="$PROJECT_DIR/logs"
LOCK_DIR="$PROJECT_DIR/logs/weekly_retrain.lock"

mkdir -p "$LOG_DIR"

if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "[$(date)] weekly retrain already running; exiting" >> "$LOG_DIR/weekly_retrain.log"
  exit 0
fi
trap 'rmdir "$LOCK_DIR"' EXIT

{
  echo "============================================================"
  echo "[$(date)] starting weekly retrain"
  cd "$PROJECT_DIR"
  "$PYTHON" train_model.py
  echo "[$(date)] weekly retrain complete"
} >> "$LOG_DIR/weekly_retrain.log" 2>&1
