#!/bin/zsh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
PY="/Users/aaronschmitt/PyCharmMiscProject/.venv/bin/python"
cd "$ROOT"
caffeinate -dimsu -w $$ &
NCPU=$(sysctl -n hw.ncpu); NW=4; if [ "$NCPU" -ge 10 ]; then NW=6; fi
rm -f workers/*.csv(N) results/*.csv(N) done.txt
"$PY" run_v7a_seed_extension.py --selfcheck | tee logs/prelaunch_selfcheck.log
for w in $(seq 0 $((NW-1))); do
  nohup "$PY" run_v7a_seed_extension.py --worker "$w" --workers "$NW" > "logs/worker_${w}.stdout.log" 2>&1 &
  echo $! > "meta/worker_${w}.pid"
done
wait
"$PY" run_v7a_seed_extension.py --integrity | tee logs/integrity.txt
"$PY" - <<'PY'
import csv, glob
rows=[]
for p in glob.glob('workers/w*_checkpoints.csv'): rows+=list(csv.DictReader(open(p)))
rows.sort(key=lambda x:(x['class'],x['task'],int(x['depth']),x['act'],int(x['seed']),int(x['step'])))
w=csv.DictWriter(open('results/checkpoint_metrics.csv','w',newline=''),fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
print('merged_rows',len(rows))
PY
printf 'DONE\n' > done.txt
