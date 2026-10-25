#!/bin/bash
# Korrigierte Vorlage (22./23.10.2026): caffeinate wird per -w an die Launcher-PID gebunden, aber nicht als Job vor 'wait' gestartet.
set -euo pipefail
cd "$(dirname "$0")"
PY=/Users/aaronschmitt/PyCharmMiscProject/.venv/bin/python
if [[ -z "${CAFFEINATED:-}" ]]; then export CAFFEINATED=1; exec caffeinate -i "$0" "$@"; fi
workers=${1:-6}; pids=()
for ((i=0;i<workers;i++)); do nohup "$PY" run_v14_transfer.py --worker "$i" --workers "$workers" > "logs/worker_${i}.stdout.log" 2>&1 & pids+=($!); done
for p in "${pids[@]}"; do wait "$p" || true; done
$PY analyze_v14.py > logs/analysis.log 2>&1 || echo 'analysis failed' >> logs/analysis.log
echo 'complete' > done.txt
