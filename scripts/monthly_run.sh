#!/bin/bash
# BES AI Aylık Otomatik Pipeline
# launchd tarafından her ayın 1'inde çalıştırılır

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV="$PROJECT_DIR/.venv/bin/python"
LOG="$PROJECT_DIR/logs/monthly_cron.log"

cd "$PROJECT_DIR" || exit 1
mkdir -p "$PROJECT_DIR/logs"
source "$PROJECT_DIR/.venv/bin/activate"

echo "========================================" >> "$LOG"
echo "$(date): Pipeline başlıyor" >> "$LOG"
echo "========================================" >> "$LOG"

# 0. TEFAS cache güncelle
echo "$(date): TEFAS cache güncelleniyor..." >> "$LOG"
$VENV -c "from src.data_collector import TEFASCollector; r=TEFASCollector().auto_refresh_cache(max_age_days=0); print('Cache güncellendi' if r else 'Cache güncel')" >> "$LOG" 2>&1

# 1. Tum portfoyleri sirayla kos + birlesik e-posta
# Not: birlesik PDF kapsam disi (PLAN-18); tek-portfoy PDF icin --report ayri kullan.
$VENV main.py --all-portfolios --quiet --email 2>> "$LOG"
PIPELINE_EXIT=$?

# 2. ML model yeniden eğitimi (TEFAS cache güncelle + eğit)
$VENV main.py --ml-train --verbose >> "$LOG" 2>&1
ML_EXIT=$?

# 3. Backtest güncelle ve öğren
$VENV main.py --backtest --learn-from-backtest --bt-start 2024-06-01 --bt-end "$(date +%Y-%m-01)" --quiet 2>> "$LOG"

echo "$(date): Pipeline=$PIPELINE_EXIT, ML=$ML_EXIT" >> "$LOG"
echo "========================================" >> "$LOG"
