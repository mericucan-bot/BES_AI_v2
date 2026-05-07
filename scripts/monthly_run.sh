#!/bin/bash
# BES AI Aylık Otomatik Pipeline
# launchd tarafından her ayın 1'inde çalıştırılır

PROJECT_DIR="$HOME/Desktop/BES_AI_v2"
VENV="$PROJECT_DIR/.venv/bin/python"
LOG="$PROJECT_DIR/logs/monthly_cron.log"

cd "$PROJECT_DIR" || exit 1
source "$PROJECT_DIR/.venv/bin/activate"

echo "========================================" >> "$LOG"
echo "$(date): Pipeline başlıyor" >> "$LOG"
echo "========================================" >> "$LOG"

# 1. Aylık portföy analizi
$VENV main.py --quiet 2>> "$LOG"
PIPELINE_EXIT=$?

# 2. ML model yeniden eğitimi (TEFAS cache güncelle + eğit)
$VENV main.py --ml-train --verbose >> "$LOG" 2>&1
ML_EXIT=$?

# 3. Backtest güncelle ve öğren
$VENV main.py --backtest --learn-from-backtest --bt-start 2024-06-01 --bt-end "$(date +%Y-%m-01)" --quiet 2>> "$LOG"

echo "$(date): Pipeline=$PIPELINE_EXIT, ML=$ML_EXIT" >> "$LOG"
echo "========================================" >> "$LOG"
