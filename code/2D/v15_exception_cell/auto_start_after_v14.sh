#!/bin/zsh
set -euo pipefail
cd "$(dirname "$0")"
V14="/Users/aaronschmitt/PyCharmMiscProject/2D/v14_transfer"
V15="/Users/aaronschmitt/PyCharmMiscProject/2D/v15_exception_cell"
LOG="$V15/logs/auto_start_after_v14.log"
PIDFILE="$V14/logs/worker_0.pid"
echo "[$(date)] watcher started" >> "$LOG"
if [[ -f "$V14/done.txt" ]]; then
  echo "[$(date)] v14 already done -> start v15" >> "$LOG"
  exec "$V15/launch_v15.sh"
fi
if [[ -f "$PIDFILE" ]]; then
  pid=$(cat "$PIDFILE" 2>/dev/null || true)
  if [[ -n "${pid:-}" ]]; then
    echo "[$(date)] watching pid $pid" >> "$LOG"
    while kill -0 "$pid" 2>/dev/null; do sleep 60; done
    echo "[$(date)] pid ended, checking completion" >> "$LOG"
  fi
fi
while true; do
  if [[ -f "$V14/done.txt" ]]; then
    echo "[$(date)] detected v14 done.txt -> start v15" >> "$LOG"
    exec "$V15/launch_v15.sh"
  fi
  if [[ -f "$V14/logs/analysis.log" ]] && grep -q 'complete\|analysis' "$V14/logs/analysis.log" 2>/dev/null; then
    echo "[$(date)] detected v14 analysis activity -> start v15" >> "$LOG"
    exec "$V15/launch_v15.sh"
  fi
  sleep 60
done
