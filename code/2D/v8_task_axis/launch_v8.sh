#!/bin/zsh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)";PY="/Users/aaronschmitt/PyCharmMiscProject/.venv/bin/python";cd "$ROOT";caffeinate -dimsu -w $$ &;NW=4
rm -f workers/*.csv(N) results/*.csv(N) done.txt
"$PY" run_v8_task_axis.py --selfcheck | tee logs/prelaunch_selfcheck.log
for w in $(seq 0 $((NW-1)));do nohup "$PY" run_v8_task_axis.py --worker "$w" --workers "$NW" > "logs/worker_${w}.stdout.log" 2>&1 &;echo $! > "meta/worker_${w}.pid";done
wait
"$PY" run_v8_task_axis.py --integrity | tee logs/integrity.txt
"$PY" - <<'PY'
import csv,glob
r=[]
for p in glob.glob('workers/w*_checkpoints.csv'):r+=list(csv.DictReader(open(p)))
assert len(r)==5760 and len({(x['run_id'],x['step']) for x in r})==5760
r.sort(key=lambda x:(x['group'],x['task'],int(x['depth']),x['act'],int(x['seed']),int(x['step'])))
with open('results/checkpoint_metrics.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=r[0].keys());w.writeheader();w.writerows(r)
PY
printf 'DONE\n' > done.txt
