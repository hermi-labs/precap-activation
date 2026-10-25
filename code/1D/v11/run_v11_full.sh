#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs
N=7
for i in $(seq 0 $((N-1))); do
  /Users/aaronschmitt/PyCharmMiscProject/.venv/bin/python capexp_v11_proto.py --mode full --shard "$i" --nshards "$N" --out v11_full_results.csv > "logs/v11_full_$i.txt" 2>&1 &
done
status=0
for pid in $(jobs -p); do wait "$pid" || status=1; done
if [ "$status" -ne 0 ]; then echo "v11 full worker failed" >&2; exit 1; fi
touch v11_full_done.txt
echo "$(date '+%F %T') finished v11 full" >> progress.log
