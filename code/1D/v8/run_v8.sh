#!/bin/bash
cd "$(dirname "$0")"
PY=/Users/aaronschmitt/PyCharmMiscProject/.venv/bin/python
NS=7
mkdir -p logs
run() {
  for ((i=0;i<NS;i++)); do
    $PY capexp_v8.py --stage $1 --seeds $2 --shard $i --nshards $NS >> logs/v8_$1_$i.txt 2>&1 &
  done
  wait
  echo "$(date '+%F %T') finished stage $1 seeds=$2" >> progress.log
}
run E 6
run F 4
run G 5
echo finished > v8_done.txt
