#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
PY=/Users/aaronschmitt/PyCharmMiscProject/.venv/bin/python
mkdir -p logs workers meta analysis
$PY run_v11_confirmation.py --selfcheck > logs/prelaunch_selfcheck.log
workers=${1:-6}
echo $$ > meta/launcher.pid
caffeinate -i -w $$ &
echo $! > meta/caffeinate.pid
for ((i=0;i<workers;i++)); do
  nohup "$PY" run_v11_confirmation.py --worker "$i" --workers "$workers" > "logs/worker_${i}.stdout.log" 2>&1 &
  echo $! > "meta/worker_${i}.pid"
done
wait
$PY analyze_v11.py > logs/analysis.log 2>&1 || echo 'analysis failed' >> logs/analysis.log
echo 'V11 complete' > done.txt
