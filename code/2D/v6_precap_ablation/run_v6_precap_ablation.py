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
WIDTH=64; STEPS=1200; BATCH=128; INIT_K=.5; LR=.001
A=1.; GAMMA=.01
ACTS=['precap_c075','precap_base','precap_c125','precap_s075','precap_s225','silu','gelu']
SEEDS=[900,901,902,903,904]
CHECKPOINTS=[0,100,300,500,800,1200]
CELLS=[('mechanism','radial_ripple',64),('mechanism','ridge_mix',64),('anchor','radial_ripple',16)]
PARAMS={'precap_c075':(0.75,1.50),'precap_base':(1.00,1.50),'precap_c125':(1.25,1.50),'precap_s075':(1.00,0.75),'precap_s225':(1.00,2.25)}
MAIN=['run_id','class','task','depth','act','seed','step','train_mse','test_mse','global_grad_norm','hmax_peak','cap_fraction','cap_c','soft_s','timestamp']
class CapExp(nn.Module):
 def __init__(self,c,a): super().__init__(); self.c=c; self.a=a
 def forward(self,x):
  z=self.a*x; ec=math.exp(self.c)
  return torch.where(z<=self.c, torch.exp(z)-1, ec-1+ec*(z-self.c))
class PreCap(nn.Module):
 def __init__(self,c,a,g,s): super().__init__(); self.c=c; self.a=a; self.g=g; self.s=s; self.cap=CapExp(c,1.)
 def forward(self,x):
  z=self.a*x; u=z+self.g*z*z; return self.cap(self.s*torch.tanh(u/self.s))
def act(name):
 if name in PARAMS:
  c,s=PARAMS[name]; return PreCap(c,A,GAMMA,s)
 return nn.SiLU() if name=='silu' else nn.GELU()
def fn(n,x,y):
 if n=='radial_ripple': return torch.sin(3*(x*x+y*y))+.3*x*y
 if n=='ridge_mix': return torch.sin(4*x)+.5*torch.cos(3*y)+.2*(x+y)**2
 raise ValueError(n)
def data(task,n,seed):
 g=torch.Generator().manual_seed(seed); x=torch.empty(n,2).uniform_(-1,1,generator=g)
 return x, fn(task,x[:,0],x[:,1]).unsqueeze(1)
class Net(nn.Module):
 def __init__(self,depth,name):
  super().__init__(); self.depth=depth; self.layers=nn.ModuleList(); d=2
  for _ in range(depth): self.layers.append(nn.Linear(d,WIDTH)); d=WIDTH
  self.out=nn.Linear(WIDTH,1); self.af=act(name)
  for m in self.modules():
   if isinstance(m,nn.Linear): nn.init.normal_(m.weight,0,INIT_K/math.sqrt(m.in_features)); nn.init.zeros_(m.bias)
 def forward(self,x,collect=False):
  h=x; stats=[]
  for layer in self.layers:
   pre=layer(h); post=self.af(pre)
   skip=h.mean(1,keepdim=True).expand_as(post) if h.shape[1]!=post.shape[1] else h
   res=post/math.sqrt(self.depth); h=skip+res
   if collect: stats.append(pre.detach())
  y=self.out(h); return (y,stats) if collect else y
def configs():
 for klass,task,depth in CELLS:
  for seed in SEEDS:
   for name in ACTS:
    yield {'class':klass,'task':task,'depth':depth,'seed':seed,'act':name}
def rid(c): return f"{c['task']}_d{c['depth']}_{c['act']}_s{c['seed']}"
def append(path,row):
 new=not path.exists()
 with path.open('a',newline='') as f:
  w=csv.DictWriter(f,fieldnames=MAIN)
  if new: w.writeheader()
  w.writerow(row)
def save(c,m,x,y,xt,yt,step,global_grad,peak,worker):
 m.eval()
 with torch.no_grad():
  pred,ss=m(x,collect=True); tr=float(F.mse_loss(pred,y)); te=float(F.mse_loss(m(xt),yt))
  cval,sval=PARAMS.get(c['act'],(float('nan'),float('nan'))); cap=0.
  if c['act'] in PARAMS:
   hit=total=0
   for pre in ss:
    z=A*pre; u=z+GAMMA*z*z; v=sval*torch.tanh(u/sval); hit += int((v>cval).sum()); total += v.numel()
   cap=hit/total
  append(ROOT/'workers'/f'w{worker}_checkpoints.csv', dict(run_id=rid(c),**c,step=step,train_mse=tr,test_mse=te,global_grad_norm=global_grad,hmax_peak=peak,cap_fraction=cap,cap_c=cval,soft_s=sval,timestamp=datetime.now().isoformat(timespec='seconds')))
def execute(c,worker):
 random.seed(c['seed']); np.random.seed(c['seed']); torch.manual_seed(c['seed'])
 x,y=data(c['task'],2048,c['seed']); xt,yt=data(c['task'],512,c['seed']+10000)
 dl=DataLoader(TensorDataset(x,y),batch_size=BATCH,shuffle=True); it=iter(dl)
 m=Net(c['depth'],c['act']); opt=torch.optim.Adam(m.parameters(),lr=LR); peak=grad=0.; save(c,m,x,y,xt,yt,0,grad,peak,worker)
 for step in range(1,STEPS+1):
  try: xb,yb=next(it)
  except StopIteration: it=iter(dl); xb,yb=next(it)
  opt.zero_grad(); pred=m(xb); loss=F.mse_loss(pred,yb); loss.backward()
  grad=math.sqrt(sum(float(p.grad.detach().square().sum()) for p in m.parameters() if p.grad is not None)); opt.step()
  with torch.no_grad(): peak=max(peak,float(m(x).abs().max()))
  if step in CHECKPOINTS: save(c,m,x,y,xt,yt,step,grad,peak,worker)
def formula_selfcheck():
 x=torch.tensor([-3.,-1.,0.,.5,1.,2.,3.],dtype=torch.float64)
 m=PreCap(1.,1.,.01,1.5).double(); z=x; u=z+.01*z*z; v=1.5*torch.tanh(u/1.5); ec=math.exp(1.)
 expected=torch.where(v<=1., torch.exp(v)-1, torch.tensor(ec-1,dtype=torch.float64)+ec*(v-1))
 err=float((m(x)-expected).abs().max())
 eps=1e-7; cap=CapExp(1.,1.).double(); left=cap(torch.tensor([1.-eps],dtype=torch.float64)); right=cap(torch.tensor([1.+eps],dtype=torch.float64)); jump=float((right-left).abs())
 return {'formula_max_abs_err':err,'cap_branch_jump':jump,'baseline_params':{'C':1.,'A':1.,'GAMMA':.01,'SOFT_S':1.5}}
def integrity():
 files=sorted((ROOT/'workers').glob('w*_checkpoints.csv')); rows=[]
 for f in files:
  with f.open() as h: rows.extend(list(csv.DictReader(h)))
 expected=len(list(configs()))*len(CHECKPOINTS); keys={(r['run_id'],r['step']) for r in rows}; full={r['run_id'] for r in rows if r['step']==str(STEPS)}
 return {'checkpoint_rows':len(rows),'expected_checkpoint_rows':expected,'unique_checkpoint_keys':len(keys),'expected_unique_checkpoint_keys':expected,'full_runs':len(full),'expected_full_runs':len(list(configs()))}
def main():
 p=argparse.ArgumentParser(); p.add_argument('--worker',type=int); p.add_argument('--workers',type=int,default=1); p.add_argument('--selfcheck',action='store_true'); p.add_argument('--integrity',action='store_true'); a=p.parse_args()
 if a.selfcheck:
  print(json.dumps({'configs':len(list(configs())),'formula':formula_selfcheck(),'cells':CELLS,'acts':ACTS,'seeds':SEEDS,'checkpoints':CHECKPOINTS},indent=2)); return
 if a.integrity:
  print(json.dumps(integrity(),indent=2)); return
 cs=list(configs())
 for i,c in enumerate(cs):
  if i % a.workers == a.worker: execute(c,a.worker); print(f"done {rid(c)}", flush=True)
if __name__=='__main__': main()
