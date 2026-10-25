import csv, statistics
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parent
p=ROOT/'results'/'checkpoint_metrics.csv'
if not p.exists(): raise SystemExit('No snapshot metrics yet. Run diagnostic snapshots first.')
rows=list(csv.DictReader(p.open()))
groups=defaultdict(list)
for r in rows: groups[(r['class'],r['task'],r['depth'],r['act'],r['step'])].append(r)
out=[]
for k,rs in sorted(groups.items()):
 out.append(dict(klass=k[0],task=k[1],depth=k[2],act=k[3],step=k[4],n=len(rs),median_test_mse=statistics.median(float(r['test_mse']) for r in rs),median_grad=statistics.median(float(r['grad_norm']) for r in rs),median_peak=statistics.median(float(r['hmax_peak']) for r in rs),median_cap=statistics.median(float(r['cap_fraction']) for r in rs)))
with (ROOT/'results'/'v4_snapshot_summary.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=out[0].keys()); w.writeheader(); w.writerows(out)
print(f'wrote {len(out)} grouped rows to results/v4_snapshot_summary.csv')
