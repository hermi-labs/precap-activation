#!/bin/bash
# Rest von launch_q1.sh: Tempo-Probe, dann 2 Worker in screen (caffeinate haelt den Mac wach)
cd "$(dirname "$0")"
PY=$HOME/precap_mlx_venv/bin/python
[ -d workers ] && { echo "workers/ existiert schon - Abbruch"; exit 1; }
$PY q1_run.py --probe > meta/probe.json 2> logs/probe.err
for w in 0 1; do screen -dmS q1w$w caffeinate -i bash -c "$PY q1_run.py --worker $w --workers 2 >> logs/w$w.log 2>&1"; done
echo "gestartet $(date)" > logs/started.txt; screen -ls >> logs/started.txt
