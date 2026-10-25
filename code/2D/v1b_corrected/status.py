import csv,glob,os,time
H=os.path.dirname(os.path.abspath(__file__));R=[]
for p in glob.glob(H+'/results/w*.csv'):R+=list(csv.DictReader(open(p)))
n=len(R);T=135;st=float(open(H+'/meta/start.txt').read()) if os.path.exists(H+'/meta/start.txt') else time.time();rate=n/max(time.time()-st,1)*3600
print(time.strftime('%F %T'),f'| {n}/{T} = {n/T:.1%} | bad={sum(r["bad"]=="True" for r in R)} | {rate:.0f} Runs/h');print('Done',os.path.exists(H+'/done.txt'))
