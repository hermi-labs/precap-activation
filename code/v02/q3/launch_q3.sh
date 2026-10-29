#!/bin/bash
# Studie Q3 starten: einfrieren, Selbsttest, Rauchtest MNIST d100 (3 Bestaetigungslaeufe, alle ok), dann 2 Worker in screen
cd "$(dirname "$0")"
PY=${PY:-$HOME/precap_mlx_venv/bin/python}
mkdir -p meta logs
if [ -d workers ]; then echo "Ergebnisordner existiert schon - Abbruch" > logs/launch.txt; exit 1; fi
shasum -a 256 q3_run.py q3_analyze.py lr_from_p3a3.csv lr_tat_from_q2.csv Q3_Praeregistrierung.md src/precap/*.py > meta/frozen_sha256.txt
date '+%Y-%m-%dT%H:%M:%S%z' > meta/frozen_at.txt
$PY q3_run.py --selfcheck > meta/selfcheck.json 2>/dev/null || { echo "Selbsttest fehlgeschlagen" > logs/launch.txt; exit 1; }
grep -q '"qa_asym_grid_ok": true' meta/selfcheck.json && grep -q '"n_confirm": 180' meta/selfcheck.json || { echo "Selbsttest-Werte nicht ok" > logs/launch.txt; exit 1; }
rm -rf smoke; $PY q3_run.py --smoke --worker 0 --workers 1 > logs/smoke.log 2>&1
b=$(grep -c ',ok,' smoke/workers/w0.csv 2>/dev/null)
if [ "$b" != "3" ]; then echo "Rauchtest nicht ok (confirm $b) - kein Start" > logs/launch.txt; exit 1; fi
for w in 0 1; do screen -dmS q3w$w caffeinate -i bash -c "$PY q3_run.py --worker $w --workers 2 >> logs/w$w.log 2>&1"; done
sleep 2; { echo "gestartet $(date)"; screen -ls; } > logs/launch.txt
