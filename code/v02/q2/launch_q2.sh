#!/bin/bash
# Studie Q2 starten: einfrieren, Selbsttest, Rauchtest MNIST d100 (2 Tuning + 2 Bestaetigung, alle ok), dann 2 Worker in screen
cd "$(dirname "$0")"
PY=${PY:-$HOME/precap_mlx_venv/bin/python}
mkdir -p meta logs
if [ -d workers ] || [ -d tune ]; then echo "Ergebnisordner existiert schon - Abbruch" > logs/launch.txt; exit 1; fi
shasum -a 256 q2_run.py q2_analyze.py lr_from_p3a3.csv Q2_Praeregistrierung.md src/precap/*.py > meta/frozen_sha256.txt
date '+%Y-%m-%dT%H:%M:%S%z' > meta/frozen_at.txt
$PY q2_run.py --selfcheck > meta/selfcheck.json 2>/dev/null || { echo "Selbsttest fehlgeschlagen" > logs/launch.txt; exit 1; }
rm -rf smoke; $PY q2_run.py --smoke --worker 0 --workers 1 > logs/smoke.log 2>&1
a=$(grep -c ',ok,' smoke/tune/workers/w0.csv 2>/dev/null); b=$(grep -c ',ok,' smoke/workers/w0.csv 2>/dev/null)
if [ "$a" != "2" ] || [ "$b" != "2" ]; then echo "Rauchtest nicht ok (tune $a, confirm $b) - kein Start" > logs/launch.txt; exit 1; fi
for w in 0 1; do screen -dmS q2w$w caffeinate -i bash -c "$PY q2_run.py --worker $w --workers 2 >> logs/w$w.log 2>&1"; done
sleep 2; { echo "gestartet $(date)"; screen -ls; } > logs/launch.txt
