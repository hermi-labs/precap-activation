#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"; PY=/Users/aaronschmitt/PyCharmMiscProject/.venv/bin/python
if [[ -z "${CAFFEINATED:-}" ]]; then export CAFFEINATED=1; exec caffeinate -i "$0" "$@"; fi
W=${1:-6}; mkdir -p logs workers meta analysis
$PY run_v17_scale.py --selfcheck > logs/prelaunch_selfcheck.log
pids=(); for ((i=0;i<W;i++)); do nohup "$PY" run_v17_scale.py --worker "$i" --workers "$W" > "logs/worker_$i.stdout.log" 2>&1 & pids+=($!); done
for p in "${pids[@]}"; do wait "$p" || true; done
$PY run_v17_scale.py --integrity > logs/integrity.json
$PY analyze_v17.py > logs/analysis.log 2>&1; echo 'V17 complete' > done.txt
