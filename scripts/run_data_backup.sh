#!/bin/zsh
set -euo pipefail

PROJECT_DIR="/Users/dougboehm/Baseball Code/win_probability"
BACKUP_DIR="/Users/dougboehm/Library/CloudStorage/GoogleDrive-dboehmjr@gmail.com/My Drive/Baseball Code Backups"
LOG_DIR="$PROJECT_DIR/logs"
LOCK_DIR="$PROJECT_DIR/logs/data_backup.lock"
DAILY_LOCK="$PROJECT_DIR/logs/daily_refresh.lock"
WEEKLY_LOCK="$PROJECT_DIR/logs/weekly_retrain.lock"
STAMP="$(date +%F)"
TMP_ZIP="/tmp/win_probability_data_backup_${STAMP}_$$.zip"

mkdir -p "$LOG_DIR"

if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "[$(date)] data backup already running; exiting" >> "$LOG_DIR/data_backup.log"
  exit 0
fi
trap 'rm -f "$TMP_ZIP"; rmdir "$LOCK_DIR"' EXIT

{
  echo "============================================================"
  echo "[$(date)] starting data backup"

  if [[ ! -d "$BACKUP_DIR" ]]; then
    echo "[$(date)] creating backup directory: $BACKUP_DIR"
    mkdir -p "$BACKUP_DIR"
  fi

  waited=0
  while [[ -d "$DAILY_LOCK" || -d "$WEEKLY_LOCK" ]]; do
    if (( waited >= 120 )); then
      echo "[$(date)] refresh/retrain still running after 120 minutes; backing up anyway"
      break
    fi
    echo "[$(date)] refresh/retrain lock present; waiting 60 seconds"
    sleep 60
    waited=$(( waited + 1 ))
  done

  cd "$PROJECT_DIR"
  /usr/bin/zip -rq "$TMP_ZIP" data
  /usr/bin/unzip -tq "$TMP_ZIP"

  cp "$TMP_ZIP" "$BACKUP_DIR/win_probability_data_backup_latest.zip"
  cp "$TMP_ZIP" "$BACKUP_DIR/win_probability_data_backup_${STAMP}.zip"

  echo "[$(date)] backup complete:"
  echo "  $BACKUP_DIR/win_probability_data_backup_latest.zip"
  echo "  $BACKUP_DIR/win_probability_data_backup_${STAMP}.zip"
} >> "$LOG_DIR/data_backup.log" 2>&1
