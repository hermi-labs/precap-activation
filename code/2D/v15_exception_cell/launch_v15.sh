#!/bin/zsh
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-/Users/aaronschmitt/PyCharmMiscProject/.venv/bin/python}
workers=${1:-6}
echo "starting V15 with $workers workers"
$PY run_v15_exception.py --selfcheck > logs/prelaunch_selfcheck.log
for ((i=0;i<workers;i++)); do
  echo "launch worker $i/$workers"
  nohup "$PY" run_v15_exception.py --worker "$i" --workers "$workers" > "logs/worker_${i}.stdout.log" 2>&1 &
  echo $! > "logs/worker_${i}.pid"
done
echo "waiting for workers..."
wait || true
echo "workers finished; starting analysis"
$PY analyze_v15.py > logs/analysis.log 2>&1 || echo 'analysis failed' >> logs/analysis.log
echo 'V15 complete' > done.txt
