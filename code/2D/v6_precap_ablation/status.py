from pathlib import Path
import csv
R=Path(__file__).resolve().parent
p=R/'results'/'checkpoint_metrics.csv'
if not p.exists():
 files=sorted((R/'workers').glob('w*_checkpoints.csv'))
 p=files[0] if files else None
rows=[] if p is None else list(csv.DictReader(open(p)))
full={r['run_id'] for r in rows if r.get('step')=='1200'}
print({'full_runs':f'{len(full)}/105','checkpoint_rows':f'{len(rows)}/630','done':(R/'done.txt').exists(),'source':str(p) if p else 'none'})
