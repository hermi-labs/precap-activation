#!/bin/zsh
cd /Users/aaronschmitt/PyCharmMiscProject/2D/v2_holdout || exit 1
mkdir -p results logs meta
date +%s > meta/start.txt
caffeinate -dimsu bash -lc '
cd /Users/aaronschmitt/PyCharmMiscProject/2D/v2_holdout || exit 1
pids=""
for i in 0 1 2 3 4 5 6; do
  /Users/aaronschmitt/PyCharmMiscProject/.venv/bin/python run_holdout_2d_v2.py worker --shard "$i" --n 7 > "logs/w${i}.log" 2>&1 &
  pids="$pids $!"
done
wait $pids
/Users/aaronschmitt/PyCharmMiscProject/.venv/bin/python run_holdout_2d_v2.py merge > logs/merge.log 2>&1 && touch done.txt
' > logs/launcher.log 2>&1 &
echo $! > meta/launcher.pid
