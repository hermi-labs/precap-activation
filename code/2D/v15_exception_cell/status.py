from pathlib import Path
import csv,glob,json
ROOT=Path(__file__).resolve().parent
rows=[]
for f in sorted(glob.glob(str(ROOT/'workers'/'w*_checkpoints.csv'))): rows+=list(csv.DictReader(open(f)))
full={r['run_id'] for r in rows if r.get('step')=='1200'}
print(json.dumps({'full_runs':f'{len(full)}/200','checkpoint_rows':f'{len(rows)}/1400','done':(ROOT/'done.txt').exists()},indent=2))
