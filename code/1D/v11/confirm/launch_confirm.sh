#!/bin/zsh
cd /Users/aaronschmitt/PyCharmMiscProject/v11/confirm || exit 1
mkdir -p results logs meta
date +%s > meta/confirm_start.txt
: > meta/confirm_worker_pids.txt
nohup caffeinate -dimsu bash -lc '
  cd /Users/aaronschmitt/PyCharmMiscProject/v11/confirm || exit 1
  for i in 0 1 2 3 4 5 6; do
    /Users/aaronschmitt/PyCharmMiscProject/.venv/bin/python capexp_v11_confirm.py worker --shard $i --n 7 > logs/confirm_w$i.log 2>&1 &
    echo $! >> meta/confirm_worker_pids.txt
  done
  wait
  /Users/aaronschmitt/PyCharmMiscProject/.venv/bin/python capexp_v11_confirm.py merge > logs/confirm_merge.log 2>&1 && touch confirm_done.txt
  echo "$(date "+%F %T") confirm finished" >> logs/confirm_progress.log
' > logs/confirm_launcher.log 2>&1 &
echo $! > meta/confirm_launcher.pid
disown 2>/dev/null || true
