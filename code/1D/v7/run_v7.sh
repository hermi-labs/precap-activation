#!/bin/bash
cd "$(dirname "$0")"
PY=/Users/aaronschmitt/PyCharmMiscProject/.venv/bin/python
NC=$(sysctl -n hw.ncpu); NS=$((NC-1)); [ $NS -gt 10 ] && NS=10; [ $NS -lt 1 ] && NS=1
mkdir -p logs
run() {
  for ((i=0;i<NS;i++)); do
    $PY capexp_v7_final_tuning.py --stage $1 --seeds $2 --steps $3 --out-prefix $4 --shard $i --nshards $NS >> logs/$4_$1_$i.txt 2>&1 &
  done
  wait
  echo "$(date '+%F %T') finished stage $1 seeds=$2 steps=$3" >> progress.log
}
run A 3 800 experiments_v7
run B 3 800 experiments_v7
run C 3 800 experiments_v7
run A 5 800 experiments_v7
run D 3 3000 experiments_v7_long
echo finished > v7_done.txt
