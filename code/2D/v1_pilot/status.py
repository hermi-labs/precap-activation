import csv,glob,os,time
H=os.path.dirname(os.path.abspath(__file__)); T=135; r=[]
for p in glob.glob(H+'/results/w*.csv'):
    with open(p,newline='') as f: r+=list(csv.DictReader(f))
n=len(r); sp=H+'/meta/start.txt'; start=float(open(sp).read()) if os.path.exists(sp) else time.time(); e=time.time()-start; rate=n/e*3600 if n else 0
print(time.strftime('%F %T'), f'| {n}/{T} = {n/T:.1%} | bad={sum(x["bad"]=="True" for x in r)} | {rate:.0f} Runs/h')
if rate: print('Rest ca.', f'{(T-n)/rate:.1f} h')
print('Done', os.path.exists(H+'/done.txt'))
