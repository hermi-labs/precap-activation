#!/bin/bash
# V16 Tiefenstudie. Schlafschutz: der gesamte Launcher laeuft unter caffeinate (kein Hintergrundjob vor wait).
set -euo pipefail
cd "$(dirname "$0")"
PY=/Users/aaronschmitt/PyCharmMiscProject/.venv/bin/python
if [[ -z "${CAFFEINATED:-}" ]]; then export CAFFEINATED=1; exec caffeinate -i "$0" "$@"; fi
mkdir -p logs workers meta analysis
workers=${1:-6}
echo "starting V16 with $workers workers"
$PY run_v16_depth.py --selfcheck > logs/prelaunch_selfcheck.log
echo $$ > meta/launcher.pid; pids=()
for ((i=0;i<workers;i++)); do
  echo "launch worker $i/$workers"
  nohup "$PY" run_v16_depth.py --worker "$i" --workers "$workers" > "logs/worker_${i}.stdout.log" 2>&1 &
  pids+=($!); echo $! > "meta/worker_${i}.pid"
done
echo "waiting for workers..."
for p in "${pids[@]}"; do wait "$p" || true; done
echo "workers finished; integrity + analysis"
$PY run_v16_depth.py --integrity > logs/integrity.json 2>&1 || true
$PY analyze_v16.py > logs/analysis.log 2>&1 || echo 'analysis failed' >> logs/analysis.log
echo 'V16 complete' > done.txt
