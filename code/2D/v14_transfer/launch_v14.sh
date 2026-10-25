#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
PY=/Users/aaronschmitt/PyCharmMiscProject/.venv/bin/python
mkdir -p logs workers meta analysis
$PY run_v14_transfer.py --selfcheck > logs/prelaunch_selfcheck.log
workers=${1:-6}
echo "starting V14 with $workers workers"
echo $$ > meta/launcher.pid
caffeinate -i -w $$ &
echo $! > meta/caffeinate.pid
for ((i=0;i<workers;i++)); do
  echo "launch worker $i/$workers"
  nohup "$PY" run_v14_transfer.py --worker "$i" --workers "$workers" > "logs/worker_${i}.stdout.log" 2>&1 &
  echo $! > "meta/worker_${i}.pid"
done
echo "waiting for workers..."
wait
echo "workers finished; starting analysis"
$PY analyze_v14.py > logs/analysis.log 2>&1 || echo 'analysis failed' >> logs/analysis.log
echo 'V14 complete' > done.txt
