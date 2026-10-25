#!/bin/zsh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
PY="/Users/aaronschmitt/PyCharmMiscProject/.venv/bin/python"
cd "$ROOT"
"$PY" run_v5_depth64.py selfcheck > logs/prelaunch_selfcheck.log
"$PY" run_v5_depth64.py freeze > logs/freeze.log
for shard in 0 1 2 3; do
  caffeinate -dimsu "$PY" run_v5_depth64.py worker --shard "$shard" --n 4 > "logs/worker_${shard}.stdout.log" 2>&1 &
  echo $! > "meta/worker_${shard}.pid"
done
wait
"$PY" run_v5_depth64.py merge > logs/merge.log
printf 'DONE\n' > done.txt
