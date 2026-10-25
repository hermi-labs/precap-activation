#!/bin/zsh
cd /Users/aaronschmitt/PyCharmMiscProject/2D/v1_pilot || exit 1
mkdir -p results logs meta
date +%s > meta/start.txt
: > meta/worker_pids.txt
nohup caffeinate -dimsu bash -lc '
 cd /Users/aaronschmitt/PyCharmMiscProject/2D/v1_pilot || exit 1
 for i in 0 1 2 3 4 5 6; do
  /Users/aaronschmitt/PyCharmMiscProject/.venv/bin/python capexp_2d_v1.py worker --shard $i --n 7 > logs/w$i.log 2>&1 &
  echo $! >> meta/worker_pids.txt
 done
 wait
 /Users/aaronschmitt/PyCharmMiscProject/.venv/bin/python capexp_2d_v1.py merge > logs/merge.log 2>&1 && touch done.txt
 echo "$(date '+%F %T') 2D v1 finished" >> logs/progress.log
' > logs/launcher.log 2>&1 &
echo $! > meta/launcher.pid
