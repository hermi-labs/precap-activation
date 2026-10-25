import csv, glob, os, time
from collections import Counter
H=os.path.dirname(os.path.abspath(__file__)); TOTAL=2400
rows=[]
for p in glob.glob(H+'/results/confirm_w*.csv'):
    with open(p,newline='') as f: rows+=list(csv.DictReader(f))
n=len(rows)
sp=H+'/meta/confirm_start.txt'
start=float(open(sp).read()) if os.path.exists(sp) else time.time()
el=time.time()-start; rate=n/el*3600 if el>0 and n else 0
bad=sum(r['bad']=='True' for r in rows)
print(time.strftime('%F %T'),f'| {n}/{TOTAL} Runs = {n/TOTAL:.1%} | bad={bad} | Laufzeit {el/60:.0f} min | {rate:.0f} Runs/h')
if rate: print('Rest ca.',f'{(TOTAL-n)/rate:.1f} h (grob)')
print('Tiefen:',dict(sorted(Counter(r['depth'] for r in rows).items(),key=lambda x:int(x[0]))))
print('Seeds (Runs):',dict(sorted(Counter(r['seed'] for r in rows).items())))
print('DONE-Marker:', 'ja' if os.path.exists(H+'/confirm_done.txt') else 'nein')
