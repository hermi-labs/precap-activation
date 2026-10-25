#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
PY=/Users/aaronschmitt/PyCharmMiscProject/.venv/bin/python
mkdir -p logs workers meta results analysis
$PY run_v10_mechanism.py --selfcheck > logs/prelaunch_selfcheck.log
workers=${1:-4}
echo $$ > meta/launcher.pid
for ((i=0;i<workers;i++)); do
  nohup "$PY" run_v10_mechanism.py --worker "$i" --workers "$workers" > "logs/worker_${i}.stdout.log" 2>&1 &
  echo $! > "meta/worker_${i}.pid"
done
# Keep the Mac awake while this launcher waits for all workers.
caffeinate -i -w $$ &
echo $! > meta/caffeinate.pid
wait
