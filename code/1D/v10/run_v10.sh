#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs
N=7
for i in $(seq 0 $((N-1))); do
  /Users/aaronschmitt/PyCharmMiscProject/.venv/bin/python capexp_v10.py --shard "$i" --nshards "$N" --out v10_results.csv > "logs/v10_$i.txt" 2>&1 &
done
status=0
for pid in $(jobs -p); do wait "$pid" || status=1; done
if [ "$status" -ne 0 ]; then echo "v10 worker failed" >&2; exit 1; fi
touch v10_done.txt
echo "$(date '+%F %T') finished v10" >> progress.log
