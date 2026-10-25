import csv
from pathlib import Path
ROOT=Path(__file__).resolve().parent
rows=[]
for f in sorted((ROOT/'workers').glob('w*_checkpoints.csv')): rows+=list(csv.DictReader(open(f)))
full={r['run_id'] for r in rows if r['step']=='1200'}
print({'full_runs':f'{len(full)}/160','checkpoint_rows':f'{len(rows)}/1120','done':(ROOT/'done.txt').exists()})
