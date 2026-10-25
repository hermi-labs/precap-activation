import csv,glob,collections,datetime
from pathlib import Path
R=Path(__file__).resolve().parent;EXP=4050;EXPROWS=EXP*8
rows=[];[rows.extend(csv.DictReader(open(f))) for f in sorted(glob.glob(str(R/'workers'/'w*_checkpoints.csv')))]
full={r['run_id'] for r in rows if r['step']=='3000'};bad={r['run_id'] for r in rows if r['run_status']!='ok'}
keys={(r['run_id'],r['step']) for r in rows}
print(f"Runs fertig: {len(full)+len(bad)}/{EXP} ({(len(full)+len(bad))/EXP:.1%})  davon nicht-endlich: {len(bad)}")
print(f"Zeilen: {len(rows)}/{EXPROWS}, eindeutig: {len(keys)}, Duplikate: {len(rows)-len(keys)}")
c=collections.Counter((r['task'],r['depth']) for r in rows if r['step']=='3000' or r['run_status']!='ok')
for k in sorted(c):print(f"  {k[0]:14s} Tiefe {k[1]:>3s}: {c[k]}/270")
print('done.txt:',(R/'done.txt').exists(),' Stand',datetime.datetime.now().strftime('%H:%M:%S'))
