#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs
N=4
for i in $(seq 0 $((N-1))); do
  /Users/aaronschmitt/PyCharmMiscProject/.venv/bin/python capexp_v11_proto.py --mode mini --shard "$i" --nshards "$N" --out v11_mini_results.csv > "logs/v11_$i.txt" 2>&1 &
done
status=0
for pid in $(jobs -p); do wait "$pid" || status=1; done
if [ "$status" -ne 0 ]; then echo "v11 mini worker failed" >&2; exit 1; fi
touch v11_mini_done.txt
echo "$(date '+%F %T') finished v11 mini" >> progress.log
