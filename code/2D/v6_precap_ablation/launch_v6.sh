#!/bin/zsh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
PY="/Users/aaronschmitt/PyCharmMiscProject/.venv/bin/python"
cd "$ROOT"
rm -f workers/*.csv(N) results/*.csv(N) done.txt logs/*(N) meta/worker_*.pid(N)
"$PY" run_v6_precap_ablation.py --selfcheck | tee logs/prelaunch_selfcheck.log
for w in 0 1 2 3; do
  nohup "$PY" run_v6_precap_ablation.py --worker "$w" --workers 4 > "logs/worker_${w}.stdout.log" 2>&1 &
  echo $! > "meta/worker_${w}.pid"
done
wait
"$PY" run_v6_precap_ablation.py --integrity | tee logs/integrity.txt
"$PY" - <<'PY'
import csv, glob
rows = []
for p in glob.glob('workers/w*_checkpoints.csv'):
    with open(p) as f:
        rows += list(csv.DictReader(f))
rows.sort(key=lambda x: (x['class'], x['task'], int(x['depth']), x['act'], int(x['seed']), int(x['step'])))
if not rows:
    raise SystemExit('no checkpoint rows found to merge')
with open('results/checkpoint_metrics.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)
print(f'merged_rows={len(rows)}')
PY
printf 'DONE\n' > done.txt
