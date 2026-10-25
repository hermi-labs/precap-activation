from pathlib import Path
import csv
R=Path(__file__).resolve().parent;p=R/'results'/'checkpoint_metrics.csv';rows=[]
if p.exists():rows=list(csv.DictReader(open(p)))
else:
 for f in (R/'workers').glob('w*_checkpoints.csv'):rows+=list(csv.DictReader(open(f)))
print({'full_runs':f"{len({r['run_id'] for r in rows if r.get('step')=='1200'})}/960",'checkpoint_rows':f'{len(rows)}/5760','done':(R/'done.txt').exists()})
