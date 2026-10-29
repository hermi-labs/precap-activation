#!/bin/bash
# Idee A starten: einfrieren, Selbsttest, Rauchtest Tiefe 200 (10 Tuning + 5 Bestaetigung, alle ok), dann 2 Worker in screen
cd "$(dirname "$0")"
PY=${PY:-$HOME/precap_mlx_venv/bin/python}
mkdir -p meta logs
if [ -d workers ] || [ -d tune ]; then echo "Ergebnisordner existiert schon - Abbruch" > logs/launch.txt; exit 1; fi
shasum -a 256 a_run.py a_analyze.py a_constants.py a_constants.json A_Praeregistrierung.md lr_from_p3a3.csv src/precap/*.py > meta/frozen_sha256.txt
date '+%Y-%m-%dT%H:%M:%S%z' > meta/frozen_at.txt
$PY a_run.py --selfcheck > meta/selfcheck.json 2>/dev/null || { echo "Selbsttest fehlgeschlagen" > logs/launch.txt; exit 1; }
$PY a_run.py --smoke --worker 0 --workers 1 > logs/smoke.log 2>&1
a=$(grep -c ',ok,' smoke/tune/workers/w0.csv 2>/dev/null); b=$(grep -c ',ok,' smoke/workers/w0.csv 2>/dev/null)
if [ "$a" != "10" ] || [ "$b" != "5" ]; then echo "Rauchtest nicht ok (tune $a, confirm $b) - kein Start" > logs/launch.txt; exit 1; fi
for w in 0 1; do screen -dmS aw$w caffeinate -i bash -c "$PY a_run.py --worker $w --workers 2 >> logs/w$w.log 2>&1"; done
sleep 2; { echo "gestartet $(date)"; screen -ls; } > logs/launch.txt
