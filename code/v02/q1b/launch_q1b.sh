#!/bin/bash
# Q1b starten: einfrieren, Selbsttest, Rauchtest auf echten Daten (muss 4/4 ok sein), dann 2 Worker in screen
cd "$(dirname "$0")"
PY=${PY:-$HOME/precap_mlx_venv/bin/python}
mkdir -p meta logs
if [ -d workers ]; then echo "workers/ existiert schon - Abbruch" | tee logs/launch.txt; exit 1; fi
shasum -a 256 q1b_run.py q1b_analyze.py Q1b_Praeregistrierung.md lr_from_p3a3.csv src/precap/*.py > meta/frozen_sha256.txt
date '+%Y-%m-%dT%H:%M:%S%z' > meta/frozen_at.txt
$PY q1b_run.py --selfcheck > meta/selfcheck.json 2>/dev/null || { echo "Selbsttest fehlgeschlagen" > logs/launch.txt; exit 1; }
grep -q '"q_fp8_grid_ok": true' meta/selfcheck.json || { echo "fp8-Gitter falsch" > logs/launch.txt; exit 1; }
$PY q1b_run.py --smoke --limit 4 --worker 0 > logs/smoke.log 2>&1
n=$(grep -c ',ok,' smoke/workers/w0.csv 2>/dev/null)
if [ "$n" != "4" ]; then echo "Rauchtest nicht 4/4 ok ($n) - kein Start" > logs/launch.txt; exit 1; fi
for w in 0 1; do screen -dmS q1bw$w caffeinate -i bash -c "$PY q1b_run.py --worker $w --workers 2 >> logs/w$w.log 2>&1"; done
sleep 2; { echo "gestartet $(date)"; screen -ls; } > logs/launch.txt
