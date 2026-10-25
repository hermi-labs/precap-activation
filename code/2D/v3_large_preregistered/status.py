import csv,glob,os,time
H=os.path.dirname(os.path.abspath(__file__)); rows=[]
for p in glob.glob(H+'/results/w*.csv'): rows += list(csv.DictReader(open(p)))
n=len(rows); total=900
start=float(open(H+'/meta/start.txt').read()) if os.path.exists(H+'/meta/start.txt') else None
rate=0.0; remain=None
if start:
    elapsed=max((time.time()-start)/3600,1e-9); rate=n/elapsed; remain=(total-n)/rate if rate>0 else None
print(time.strftime('%F %T'),f'| {n}/{total} = {n/total:.1%} | bad={sum(r["bad"]=="True" for r in rows)} | {rate:.0f} Runs/h')
if remain is not None: print(f'Rest ca. {remain:.2f} h')
print('Done',os.path.exists(H+'/done.txt'))
