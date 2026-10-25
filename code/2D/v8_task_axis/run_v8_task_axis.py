import argparse,csv,json,math,random
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset
torch.set_num_threads(1)
ROOT=Path(__file__).resolve().parent
WIDTH=64; STEPS=1200; BATCH=128; INIT_K=.5; LR=.001; A=1.; GAMMA=.01
ACTS=['precap_base','precap_s225','silu','gelu']; SEEDS=list(range(3000,3020)); CHECKPOINTS=[0,100,300,500,800,1200]
TASKS=[('radial','radial_ripple'),('radial','radial_highfreq'),('radial','radial_bessel'),('ridge','ridge_mix'),('ridge','ridge_highfreq'),('ridge','ridge_curved')]
DEPTHS=[16,32]; PARAMS={'precap_base':(1.,1.5),'precap_s225':(1.,2.25)}
MAIN=['run_id','group','task','depth','act','seed','step','train_mse','test_mse','global_grad_norm','hmax_peak','cap_fraction','cap_c','soft_s','timestamp']
class CapExp(nn.Module):
 def __init__(self,c,a): super().__init__();self.c=c;self.a=a
 def forward(self,x):
  z=self.a*x;ec=math.exp(self.c);return torch.where(z<=self.c,torch.exp(z)-1,ec-1+ec*(z-self.c))
class PreCap(nn.Module):
 def __init__(self,c,a,g,s): super().__init__();self.c=c;self.a=a;self.g=g;self.s=s;self.cap=CapExp(c,1.)
 def forward(self,x):
  z=self.a*x;u=z+self.g*z*z;return self.cap(self.s*torch.tanh(u/self.s))
def act(n):
 if n in PARAMS: c,s=PARAMS[n];return PreCap(c,A,GAMMA,s)
 return nn.SiLU() if n=='silu' else nn.GELU()
def fn(n,x,y):
 r2=x*x+y*y
 if n=='radial_ripple': return torch.sin(3*r2)+.3*x*y
 if n=='radial_highfreq': return torch.sin(7*r2)+.15*torch.cos(4*r2)
 if n=='radial_bessel': return torch.cos(5*torch.sqrt(r2+.05))
 if n=='ridge_mix': return torch.sin(4*x)+.5*torch.cos(3*y)+.2*(x+y)**2
 if n=='ridge_highfreq': return torch.sin(9*x)+.4*torch.cos(7*y)
 if n=='ridge_curved': return torch.sin(5*(x+.35*y*y))+.35*torch.cos(4*y)
 raise ValueError(n)
def data(task,n,seed):
 g=torch.Generator().manual_seed(seed);x=torch.empty(n,2).uniform_(-1,1,generator=g);return x,fn(task,x[:,0],x[:,1]).unsqueeze(1)
class Net(nn.Module):
 def __init__(self,depth,name):
  super().__init__();self.depth=depth;self.layers=nn.ModuleList();d=2
  for _ in range(depth):self.layers.append(nn.Linear(d,WIDTH));d=WIDTH
  self.out=nn.Linear(WIDTH,1);self.af=act(name)
  for m in self.modules():
   if isinstance(m,nn.Linear):nn.init.normal_(m.weight,0,INIT_K/math.sqrt(m.in_features));nn.init.zeros_(m.bias)
 def forward(self,x,collect=False):
  h=x;ss=[]
  for layer in self.layers:
   pre=layer(h);post=self.af(pre);skip=h.mean(1,keepdim=True).expand_as(post) if h.shape[1]!=post.shape[1] else h;h=skip+post/math.sqrt(self.depth)
   if collect:ss.append(pre.detach())
  y=self.out(h);return (y,ss) if collect else y
def configs():
 for group,task in TASKS:
  for depth in DEPTHS:
   for seed in SEEDS:
    for name in ACTS:yield {'group':group,'task':task,'depth':depth,'seed':seed,'act':name}
def rid(c):return f"{c['task']}_d{c['depth']}_{c['act']}_s{c['seed']}"
def append(path,row):
 new=not path.exists()
 with path.open('a',newline='') as f:
  w=csv.DictWriter(f,fieldnames=MAIN)
  if new:w.writeheader()
  w.writerow(row)
def save(c,m,x,y,xt,yt,step,grad,peak,wid):
 m.eval()
 with torch.no_grad():
  pred,ss=m(x,collect=True);tr=float(F.mse_loss(pred,y));te=float(F.mse_loss(m(xt),yt));cv,sv=PARAMS.get(c['act'],(float('nan'),float('nan')));cap=0.
  if c['act'] in PARAMS:
   hit=tot=0
   for pre in ss:
    v=sv*torch.tanh((A*pre+GAMMA*(A*pre)**2)/sv);hit+=int((v>cv).sum());tot+=v.numel()
   cap=hit/tot
  append(ROOT/'workers'/f'w{wid}_checkpoints.csv',dict(run_id=rid(c),**c,step=step,train_mse=tr,test_mse=te,global_grad_norm=grad,hmax_peak=peak,cap_fraction=cap,cap_c=cv,soft_s=sv,timestamp=datetime.now().isoformat(timespec='seconds')))
def execute(c,wid):
 random.seed(c['seed']);np.random.seed(c['seed']);torch.manual_seed(c['seed']);x,y=data(c['task'],2048,c['seed']);xt,yt=data(c['task'],512,c['seed']+10000);dl=DataLoader(TensorDataset(x,y),batch_size=BATCH,shuffle=True);it=iter(dl);m=Net(c['depth'],c['act']);o=torch.optim.Adam(m.parameters(),lr=LR);peak=grad=0.;save(c,m,x,y,xt,yt,0,grad,peak,wid)
 for step in range(1,STEPS+1):
  try:xb,yb=next(it)
  except StopIteration:it=iter(dl);xb,yb=next(it)
  o.zero_grad();loss=F.mse_loss(m(xb),yb);loss.backward();grad=math.sqrt(sum(float(p.grad.detach().square().sum()) for p in m.parameters() if p.grad is not None));o.step()
  with torch.no_grad():peak=max(peak,float(m(x).abs().max()))
  if step in CHECKPOINTS:save(c,m,x,y,xt,yt,step,grad,peak,wid)
def selfcheck():
 x=torch.tensor([-3.,-1.,0.,.5,1.,2.,3.],dtype=torch.float64);m=PreCap(1.,1.,.01,1.5).double();u=x+.01*x*x;v=1.5*torch.tanh(u/1.5);ec=math.exp(1.);e=torch.where(v<=1.,torch.exp(v)-1,torch.tensor(ec-1,dtype=torch.float64)+ec*(v-1));return {'formula_max_abs_err':float((m(x)-e).abs().max()),'tasks':{n:[float(z) for z in fn(n,torch.tensor([-.7,0.,.8]),torch.tensor([.3,-.2,.5]))] for _,n in TASKS}}
def integrity():
 rows=[]
 for f in sorted((ROOT/'workers').glob('w*_checkpoints.csv')):rows+=list(csv.DictReader(open(f)))
 exp=len(list(configs()))*len(CHECKPOINTS);keys={(r['run_id'],r['step']) for r in rows};full={r['run_id'] for r in rows if r['step']==str(STEPS)};return {'checkpoint_rows':len(rows),'expected_checkpoint_rows':exp,'unique_checkpoint_keys':len(keys),'full_runs':len(full),'expected_full_runs':len(list(configs()))}
def main():
 p=argparse.ArgumentParser();p.add_argument('--worker',type=int);p.add_argument('--workers',type=int,default=1);p.add_argument('--selfcheck',action='store_true');p.add_argument('--integrity',action='store_true');a=p.parse_args()
 if a.selfcheck:print(json.dumps({'configs':len(list(configs())),'formula':selfcheck(),'tasks':TASKS,'depths':DEPTHS,'acts':ACTS,'seeds':SEEDS,'checkpoints':CHECKPOINTS},indent=2));return
 if a.integrity:print(json.dumps(integrity(),indent=2));return
 for i,c in enumerate(configs()):
  if i%a.workers==a.worker:execute(c,a.worker);print('done',rid(c),flush=True)
if __name__=='__main__':main()
