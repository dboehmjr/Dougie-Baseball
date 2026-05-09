#!/bin/zsh
set -euo pipefail

PROJECT_DIR="/Users/dougboehm/Baseball Code/win_probability"
PYTHON="/Users/dougboehm/opt/anaconda3/bin/python"
LOG_DIR="$PROJECT_DIR/logs"
LOCK_DIR="$PROJECT_DIR/logs/daily_refresh.lock"

mkdir -p "$LOG_DIR"

if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "[$(date)] daily refresh already running; exiting" >> "$LOG_DIR/daily_refresh.log"
  exit 0
fi
trap 'rmdir "$LOCK_DIR"' EXIT

{
  echo "============================================================"
  echo "[$(date)] starting daily refresh"
  cd "$PROJECT_DIR"
  "$PYTHON" daily_update.py
  echo "[$(date)] daily refresh complete"
} >> "$LOG_DIR/daily_refresh.log" 2>&1
