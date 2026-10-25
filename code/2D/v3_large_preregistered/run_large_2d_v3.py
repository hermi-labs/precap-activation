import argparse,csv,math,os,random,time
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset

torch.set_num_threads(1)
WIDTH=64; STEPS=3000; BATCH=128; DEPTHS=[4,8,16,32,64]; SEEDS=list(range(500,510)); ACTS=['precap','silu','gelu']; TASKS=['radial_ripple','ridge_mix','corner_kink','smooth_bowl','highfreq_grid','angled_kink']
C=1.0; A=1.0; GAMMA=0.01; SOFT_S=1.5; INIT_K=0.5; LR=0.001
FIELDS=['task','act','c','a','gamma','soft_s','depth','arch','init_k','lr','width','seed','timestamp','steps','bad','test_loss','grad_final','hmax_final','hmax_peak','cap_final']
class CapExp(nn.Module):
 def __init__(self,c,a): super().__init__(); self.c=c; self.a=a
 def forward(self,x):
  z=self.a*x; ec=math.exp(self.c)
  return torch.where(z<=self.c,torch.exp(z)-1,ec-1+ec*(z-self.c))
class PreCap(nn.Module):
 def __init__(self,c,a,g,s): super().__init__(); self.c=c; self.a=a; self.g=g; self.s=s; self.cap=CapExp(c,1.0)
 def forward(self,x):
  z=self.a*x; u=z+self.g*z*z
  return self.cap(self.s*torch.tanh(u/self.s))
def make_act(n): return PreCap(C,A,GAMMA,SOFT_S) if n=='precap' else (nn.SiLU() if n=='silu' else nn.GELU())
def fn(n,x,y):
 if n=='radial_ripple': return torch.sin(3*(x*x+y*y))+0.3*x*y
 if n=='ridge_mix': return torch.sin(4*x)+0.5*torch.cos(3*y)+0.2*(x+y)**2
 if n=='corner_kink': return F.relu(x+y-0.2)+0.3*torch.abs(x-y)
 if n=='smooth_bowl': return 0.40*(x*x+y*y)+0.15*torch.sin(2*x-y)
 if n=='highfreq_grid': return 0.25*torch.sin(8*x)*torch.cos(8*y)+0.15*torch.sin(5*(x+y))
 if n=='angled_kink': return 0.55*torch.abs(0.8*x-0.6*y)+0.18*torch.sin(3*x+y)
 raise ValueError(n)
def data(t,n,seed):
 g=torch.Generator().manual_seed(seed)
 p=torch.empty(n,2).uniform_(-1,1,generator=g)
 return p,fn(t,p[:,0],p[:,1]).unsqueeze(1)
class Net(nn.Module):
 def __init__(self,depth,act,width=WIDTH):
  super().__init__(); self.depth=depth; self.layers=nn.ModuleList(); din=2
  for _ in range(depth): self.layers.append(nn.Linear(din,width)); din=width
  self.out=nn.Linear(width,1); self.af=make_act(act)
  for m in self.modules():
   if isinstance(m,nn.Linear): nn.init.normal_(m.weight,0,INIT_K/math.sqrt(m.in_features)); nn.init.zeros_(m.bias)
 def forward(self,x,collect=False):
  h=x; stats=[]
  for l in self.layers:
   pre=l(h); y=self.af(pre)
   hskip=h.mean(dim=1,keepdim=True).expand_as(y) if h.shape[1]!=y.shape[1] else h
   h=hskip+y/math.sqrt(self.depth)
   if collect: stats.append((pre,h))
  out=self.out(h)
  return (out,stats) if collect else out
def run(c):
 random.seed(c['seed']); np.random.seed(c['seed']); torch.manual_seed(c['seed'])
 x,y=data(c['task'],2048,c['seed']); xt,yt=data(c['task'],512,c['seed']+10000)
 dl=DataLoader(TensorDataset(x,y),batch_size=BATCH,shuffle=True); it=iter(dl)
 m=Net(c['depth'],c['act']); opt=torch.optim.Adam(m.parameters(),lr=LR)
 bad=False; peak=0.0; gf=float('nan'); hf=float('nan'); cf=0.0
 for step in range(1,STEPS+1):
  try: xb,yb=next(it)
  except StopIteration: it=iter(dl); xb,yb=next(it)
  pred=m(xb); loss=F.mse_loss(pred,yb)
  if not torch.isfinite(loss): bad=True; break
  opt.zero_grad(); loss.backward()
  gf=math.sqrt(sum(float(p.grad.detach().square().sum()) for p in m.parameters() if p.grad is not None))
  if not math.isfinite(gf): bad=True; break
  opt.step()
  if step in {100,500,1500,3000}:
   with torch.no_grad():
    _,ss=m(xt,collect=True)
    hf=max(float(h.abs().max()) for _,h in ss); peak=max(peak,hf)
    cf=sum(float((z>C).float().mean()) for z,_ in ss)/len(ss) if c['act']=='precap' else 0.0
   print(f'checkpoint {step} {c["task"]} {c["act"]} d={c["depth"]} loss={loss.item():.5g} hmax={hf:.5g} grad={gf:.5g} cap={cf:.3f}',flush=True)
 with torch.no_grad(): tl=float(F.mse_loss(m(xt),yt)) if not bad else float('inf')
 return dict(task=c['task'],act=c['act'],c=C,a=A,gamma=GAMMA,soft_s=SOFT_S,depth=c['depth'],arch='res_inv',init_k=INIT_K,lr=LR,width=WIDTH,seed=c['seed'],timestamp=datetime.now().isoformat(timespec='seconds'),steps=STEPS,bad=bad,test_loss=tl,grad_final=gf,hmax_final=hf,hmax_peak=peak,cap_final=cf)
def configs(): return [dict(task=t,act=a,depth=d,seed=s) for s in SEEDS for t in TASKS for d in DEPTHS for a in ACTS]
def keys(r): return (r['task'],r['act'],r['depth'],r['arch'],r['width'],r['seed'])
def worker(sh,n):
 H=os.path.dirname(os.path.abspath(__file__)); p=f'{H}/results/w{sh}.csv'; have=set()
 if os.path.exists(p): have={keys(r) for r in csv.DictReader(open(p))}
 cs=[c for i,c in enumerate(configs()) if i%n==sh]
 for j,c in enumerate(cs,1):
  k=(c['task'],c['act'],str(c['depth']),'res_inv',str(WIDTH),str(c['seed']))
  if k in have: continue
  t=time.time(); r=run(c); new=not os.path.exists(p)
  with open(p,'a',newline='') as f:
   w=csv.DictWriter(f,fieldnames=FIELDS)
   if new: w.writeheader()
   w.writerow(r)
  with open(f'{H}/logs/progress.log','a') as f:
   f.write(f'{datetime.now():%F %T} w{sh} {j}/{len(cs)} {c["task"]} {c["act"]} d={c["depth"]} seed={c["seed"]} test={r["test_loss"]:.4g} bad={r["bad"]} {time.time()-t:.0f}s\n')
def merge():
 H=os.path.dirname(os.path.abspath(__file__)); rows=[]
 for i in range(7):
  p=f'{H}/results/w{i}.csv'
  if os.path.exists(p): rows += list(csv.DictReader(open(p)))
 ks={keys(r) for r in rows}; msg=f'rows={len(rows)} expected={len(configs())} unique={len(ks)} bad={sum(r["bad"]=="True" for r in rows)}'
 with open(f'{H}/v3_large_results.csv','w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
 open(f'{H}/logs/integrity.txt','w').write(msg+'\n'); print(msg)
 return len(rows)==len(configs()) and len(ks)==len(configs())
def selfcheck():
 x=torch.linspace(-5,5,10001)
 z=A*x; u=z+GAMMA*z*z; v=SOFT_S*torch.tanh(u/SOFT_S); ec=math.exp(C)
 expected=torch.where(v<=C,torch.exp(v)-1,ec-1+ec*(v-C))
 got=PreCap(C,A,GAMMA,SOFT_S)(x); err=float((got-expected).abs().max())
 assert err<1e-6, err
 for d in DEPTHS:
  m=Net(d,'precap'); q=torch.randn(7,2); out,ss=m(q,collect=True); assert out.shape==(7,1) and len(ss)==d
 print(f'SELFCHECK_OK formula_max_abs_err={err:.3g} configs={len(configs())}')
if __name__=='__main__':
 ap=argparse.ArgumentParser(); ap.add_argument('cmd',choices=['selfcheck','worker','merge']); ap.add_argument('--shard',type=int,default=0); ap.add_argument('--n',type=int,default=7); a=ap.parse_args()
 if a.cmd=='selfcheck': selfcheck()
 elif a.cmd=='worker': worker(a.shard,a.n)
 else: raise SystemExit(0 if merge() else 1)
