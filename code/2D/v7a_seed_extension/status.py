from pathlib import Path
import csv
R=Path(__file__).resolve().parent
p=R/'results'/'checkpoint_metrics.csv'
rows=[]
if p.exists(): rows=list(csv.DictReader(open(p)))
else:
 for f in sorted((R/'workers').glob('w*_checkpoints.csv')): rows+=list(csv.DictReader(open(f)))
full={r['run_id'] for r in rows if r.get('step')=='1200'}
print({'full_runs':f'{len(full)}/300','checkpoint_rows':f'{len(rows)}/1800','done':(R/'done.txt').exists()})
