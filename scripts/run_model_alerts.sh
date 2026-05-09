#!/bin/zsh
set -euo pipefail

PROJECT_DIR="/Users/dougboehm/Baseball Code/win_probability"
PYTHON="/Users/dougboehm/opt/anaconda3/bin/python"
LOG_DIR="$PROJECT_DIR/logs"
LOCK_DIR="$PROJECT_DIR/logs/model_alerts.lock"

mkdir -p "$LOG_DIR"

if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "[$(date)] model alerts already running; exiting" >> "$LOG_DIR/model_alerts.log"
  exit 0
fi
trap 'rmdir "$LOCK_DIR"' EXIT

{
  echo "============================================================"
  echo "[$(date)] checking model alerts"
  cd "$PROJECT_DIR"
  "$PYTHON" model_alerts.py
  echo "[$(date)] model alerts complete"
} >> "$LOG_DIR/model_alerts.log" 2>&1
