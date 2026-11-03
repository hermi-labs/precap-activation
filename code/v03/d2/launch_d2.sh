#!/bin/bash
# D2 starten (mit Claude): einfrieren, Selbsttest, Rauchtest (2 kurze Laeufe), dann 2 Worker in screen
cd "$(dirname "$0")"
PY=${PY:-$HOME/precap_mlx_venv/bin/python}
mkdir -p meta logs
if [ -d results ]; then echo "Ergebnisordner existiert schon - Abbruch" > logs/launch.txt; exit 1; fi
shasum -a 256 d2_diag.py D2_Praeregistrierung.md ../b10/b10_run.py ../b13/b13_run.py > meta/frozen_sha256.txt
date '+%Y-%m-%dT%H:%M:%S%z' > meta/frozen_at.txt
$PY d2_diag.py --selfcheck > meta/selfcheck.json 2>/dev/null || { echo "Selbsttest fehlgeschlagen" > logs/launch.txt; exit 1; }
grep -q '"ok": true' meta/selfcheck.json || { echo "Selbsttest nicht ok" > logs/launch.txt; exit 1; }
[ -d smoke ] && mv smoke "smoke_alt_$(date +%H%M%S)"
$PY d2_diag.py --smoke > logs/smoke.log 2>&1
b=$(ls smoke/*.json 2>/dev/null | wc -l | tr -d ' ')
if [ "$b" != "2" ]; then echo "Rauchtest nicht ok ($b) - kein Start" > logs/launch.txt; exit 1; fi
for w in 0 1; do screen -dmS d2w$w caffeinate -i bash -c "$PY d2_diag.py --worker $w --workers 2 >> logs/w$w.log 2>&1"; done
sleep 2; { echo "gestartet $(date)"; screen -ls; } > logs/launch.txt
