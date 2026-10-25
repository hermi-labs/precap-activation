import csv,glob,os,time
from collections import Counter
H=os.path.dirname(os.path.abspath(__file__));T=1620;r=[]
for p in glob.glob(H+'/results/v12_w*.csv'):
 with open(p,newline='') as f:r+=list(csv.DictReader(f))
n=len(r);sp=H+'/meta/start.txt';start=float(open(sp).read()) if os.path.exists(sp) else time.time();e=time.time()-start;rate=n/e*3600 if n else 0
print(time.strftime('%F %T'),f'| {n}/{T} = {n/T:.1%} | bad={sum(x["bad"]=="True" for x in r)} | {rate:.0f} Runs/h')
if rate:print('Rest ca.',f'{(T-n)/rate:.1f} h')
print('Tiefen',dict(sorted(Counter(x['depth'] for x in r).items(),key=lambda z:int(z[0]))));print('Done',os.path.exists(H+'/done.txt'))
