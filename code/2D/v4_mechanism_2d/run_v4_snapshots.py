import argparse,csv,hashlib,json,math,os,random
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

torch.set_num_threads(1)
ROOT=Path(__file__).resolve().parent
WIDTH=64; STEPS=3000; BATCH=128; INIT_K=.5; LR=.001
C=1.; A=1.; GAMMA=.01; SOFT_S=1.5
ACTS=['precap','silu','gelu']; SEEDS=[700,701,702,703,704]; CHECKPOINTS=[0,100,500,1500,3000]
CELLS=[('winner','radial_ripple',16),('winner','angled_kink',32),('boundary','radial_ripple',64),('boundary','ridge_mix',64),('counterexample','highfreq_grid',32)]
STAT_FIELDS=['run_id','class','task','depth','act','seed','step','train_mse','test_mse','grad_norm','hmax_peak','cap_fraction','timestamp']
LAYER_FIELDS=['run_id','class','task','depth','act','seed','step','layer','tensor','mean','std','p01','p50','p99','max_abs']

class CapExp(nn.Module):
 def __init__(self,c,a): super().__init__(); self.c=c; self.a=a
 def forward(self,x):
  z=self.a*x; ec=math.exp(self.c)
  return torch.where(z<=self.c,torch.exp(z)-1,ec-1+ec*(z-self.c))
class PreCap(nn.Module):
 def __init__(self,c,a,g,s): super().__init__(); self.c=c; self.a=a; self.g=g; self.s=s; self.cap=CapExp(c,1.)
 def forward(self,x):
  z=self.a*x; u=z+self.g*z*z
  return self.cap(self.s*torch.tanh(u/self.s))
def make_act(name): return PreCap(C,A,GAMMA,SOFT_S) if name=='precap' else (nn.SiLU() if name=='silu' else nn.GELU())
def fn(name,x,y):
 if name=='radial_ripple': return torch.sin(3*(x*x+y*y))+.3*x*y
 if name=='ridge_mix': return torch.sin(4*x)+.5*torch.cos(3*y)+.2*(x+y)**2
 if name=='corner_kink': return F.relu(x+y-.2)+.3*torch.abs(x-y)
 if name=='smooth_bowl': return .40*(x*x+y*y)+.15*torch.sin(2*x-y)
 if name=='highfreq_grid': return .25*torch.sin(8*x)*torch.cos(8*y)+.15*torch.sin(5*(x+y))
 if name=='angled_kink': return .55*torch.abs(.8*x-.6*y)+.18*torch.sin(3*x+y)
 raise ValueError(name)
def data(task,n,seed):
 g=torch.Generator().manual_seed(seed); x=torch.empty(n,2).uniform_(-1,1,generator=g)
 return x,fn(task,x[:,0],x[:,1]).unsqueeze(1)
class Net(nn.Module):
 def __init__(self,depth,act):
  super().__init__(); self.depth=depth; self.layers=nn.ModuleList(); d=2
  for _ in range(depth): self.layers.append(nn.Linear(d,WIDTH)); d=WIDTH
  self.out=nn.Linear(WIDTH,1); self.af=make_act(act)
  for m in self.modules():
   if isinstance(m,nn.Linear): nn.init.normal_(m.weight,0,INIT_K/math.sqrt(m.in_features)); nn.init.zeros_(m.bias)
 def forward(self,x,collect=False):
  h=x; out=[]
  for layer in self.layers:
   pre=layer(h); post=self.af(pre); skip=h.mean(1,keepdim=True).expand_as(post) if h.shape[1]!=post.shape[1] else h
   h=skip+post/math.sqrt(self.depth)
   if collect: out.append((pre.detach(),post.detach(),(post/math.sqrt(self.depth)).detach()))
  y=self.out(h); return (y,out) if collect else y
def configs():
 for klass,task,depth in CELLS:
  for seed in SEEDS:
   for act in ACTS: yield dict(klass=klass,task=task,depth=depth,seed=seed,act=act)
def run_id(c): return f"{c['task']}_d{c['depth']}_{c['act']}_s{c['seed']}"
def qstats(x):
 q=torch.quantile(x.flatten(),torch.tensor([.01,.5,.99])); return dict(mean=float(x.mean()),std=float(x.std(unbiased=False)),p01=float(q[0]),p50=float(q[1]),p99=float(q[2]),max_abs=float(x.abs().max()))
def append(path,fields,row):
 new=not path.exists()
 with path.open('a',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields)
  if new:w.writeheader()
  w.writerow(row)
def snapshot(c,model,x,y,xt,yt,step,grad_norm,peak):
 model.eval()
 with torch.no_grad():
  pred,stats=model(x,collect=True); tr=float(F.mse_loss(pred,y)); te=float(F.mse_loss(model(xt),yt))
 cap=0.0
 if c['act']=='precap':
  total=0; capped=0
  for pre,_,_ in stats:
   z=A*pre; u=z+GAMMA*z*z; v=SOFT_S*torch.tanh(u/SOFT_S); capped+=int((v>C).sum()); total+=v.numel()
  cap=capped/total
 rid=run_id(c); stamp=datetime.now().isoformat(timespec='seconds')
 append(ROOT/'results'/'checkpoint_metrics.csv',STAT_FIELDS,dict(run_id=rid,**{'class':c['klass']},task=c['task'],depth=c['depth'],act=c['act'],seed=c['seed'],step=step,train_mse=tr,test_mse=te,grad_norm=grad_norm,hmax_peak=peak,cap_fraction=cap,timestamp=stamp))
 for i,(pre,post,res) in enumerate(stats):
  for tensor,name in [(pre,'pre'),(post,'post'),(res,'residual')]:
   append(ROOT/'results'/'layer_metrics.csv',LAYER_FIELDS,dict(run_id=rid,**{'class':c['klass']},task=c['task'],depth=c['depth'],act=c['act'],seed=c['seed'],step=step,layer=i,tensor=name,**qstats(tensor)))
 ck=ROOT/'snapshots'/f'{rid}_step{step}.pt'; torch.save({'state_dict':model.state_dict(),'config':c,'step':step},ck)
 return te
def execute(c):
 random.seed(c['seed']); np.random.seed(c['seed']); torch.manual_seed(c['seed'])
 x,y=data(c['task'],2048,c['seed']); xt,yt=data(c['task'],512,c['seed']+10000)
 dl=DataLoader(TensorDataset(x,y),batch_size=BATCH,shuffle=True); it=iter(dl); m=Net(c['depth'],c['act']); opt=torch.optim.Adam(m.parameters(),lr=LR)
 peak=0.; grad=0.; snapshot(c,m,x,y,xt,yt,0,grad,peak)
 for step in range(1,STEPS+1):
  try: xb,yb=next(it)
  except StopIteration: it=iter(dl); xb,yb=next(it)
  loss=F.mse_loss(m(xb),yb); opt.zero_grad(); loss.backward()
  grad=math.sqrt(sum(float(p.grad.detach().square().sum()) for p in m.parameters() if p.grad is not None))
  if not math.isfinite(grad) or not torch.isfinite(loss): raise RuntimeError(f'nonfinite {run_id(c)} at {step}')
  opt.step()
  with torch.no_grad(): _,ss=m(xb,collect=True); peak=max(peak,max(float(t.abs().max()) for z in ss for t in z[:2]))
  if step in CHECKPOINTS: snapshot(c,m,x,y,xt,yt,step,grad,peak)
def selfcheck():
 x=torch.linspace(-5,5,10001); z=A*x; u=z+GAMMA*z*z; v=SOFT_S*torch.tanh(u/SOFT_S); ec=math.exp(C)
 ref=torch.where(v<=C,torch.exp(v)-1,ec-1+ec*(v-C)); err=float((PreCap(C,A,GAMMA,SOFT_S)(x)-ref).abs().max()); assert err<1e-6
 m=Net(4,'precap'); out,ss=m(torch.randn(7,2),collect=True); assert out.shape==(7,1) and len(ss)==4
 assert len(list(configs()))==75
 print(f'SELFCHECK_OK formula_max_abs_err={err:.3g} configs=75 checkpoints={CHECKPOINTS}')
def freeze():
 hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'run_v4_snapshots.py',ROOT/'meta'/'manifest.json',ROOT/'docs'/'PreCap_Formel_Freeze_v4.md']}
 (ROOT/'meta'/'frozen_sha256.json').write_text(json.dumps(hashes,indent=2)+'\n'); print(json.dumps(hashes,indent=2))
if __name__=='__main__':
 ap=argparse.ArgumentParser(); ap.add_argument('cmd',choices=['selfcheck','freeze','run']); ap.add_argument('--limit',type=int,default=0); a=ap.parse_args()
 if a.cmd=='selfcheck': selfcheck()
 elif a.cmd=='freeze': freeze()
 else:
  cs=list(configs()); cs=cs[:a.limit] if a.limit else cs
  for i,c in enumerate(cs,1): print(f'[{i}/{len(cs)}] {run_id(c)}',flush=True); execute(c)
