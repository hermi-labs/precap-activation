import argparse,csv,hashlib,json,math,random
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset

torch.set_num_threads(1)
ROOT=Path(__file__).resolve().parent
WIDTH=64; STEPS=3000; BATCH=128; INIT_K=.5; LR=.001
C=1.; A=1.; GAMMA=.01; SOFT_S=1.5
ACTS=['precap','silu','gelu']; SEEDS=[800,801,802,803,804]; CHECKPOINTS=[0,100,300,500,800,1200,1500,2200,3000]
CELLS=[('boundary','radial_ripple',64),('boundary','ridge_mix',64),('anchor','radial_ripple',16),('counterexample','highfreq_grid',32)]
MAIN=['run_id','class','task','depth','act','seed','step','train_mse','test_mse','global_grad_norm','hmax_peak','cap_fraction','timestamp']
LAYER=['run_id','class','task','depth','act','seed','step','layer','tensor','mean','std','p01','p50','p99','max_abs','weight_grad_norm','residual_to_post_std']
class CapExp(nn.Module):
 def __init__(self,c,a): super().__init__();self.c=c;self.a=a
 def forward(self,x):
  z=self.a*x;ec=math.exp(self.c);return torch.where(z<=self.c,torch.exp(z)-1,ec-1+ec*(z-self.c))
class PreCap(nn.Module):
 def __init__(self,c,a,g,s): super().__init__();self.c=c;self.a=a;self.g=g;self.s=s;self.cap=CapExp(c,1.)
 def forward(self,x):
  z=self.a*x;u=z+self.g*z*z;return self.cap(self.s*torch.tanh(u/self.s))
def act(name): return PreCap(C,A,GAMMA,SOFT_S) if name=='precap' else (nn.SiLU() if name=='silu' else nn.GELU())
def fn(n,x,y):
 if n=='radial_ripple': return torch.sin(3*(x*x+y*y))+.3*x*y
 if n=='ridge_mix': return torch.sin(4*x)+.5*torch.cos(3*y)+.2*(x+y)**2
 if n=='highfreq_grid': return .25*torch.sin(8*x)*torch.cos(8*y)+.15*torch.sin(5*(x+y))
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
  h=x;stats=[]
  for layer in self.layers:
   pre=layer(h);post=self.af(pre);skip=h.mean(1,keepdim=True).expand_as(post) if h.shape[1]!=post.shape[1] else h;res=post/math.sqrt(self.depth);h=skip+res
   if collect:stats.append((pre.detach(),post.detach(),res.detach()))
  y=self.out(h);return (y,stats) if collect else y
def configs():
 for klass,task,depth in CELLS:
  for seed in SEEDS:
   for name in ACTS:yield {'class':klass,'task':task,'depth':depth,'seed':seed,'act':name}
def rid(c):return f"{c['task']}_d{c['depth']}_{c['act']}_s{c['seed']}"
def append(path,fields,row):
 new=not path.exists()
 with path.open('a',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields)
  if new:w.writeheader()
  w.writerow(row)
def qstats(x):
 q=torch.quantile(x.flatten(),torch.tensor([.01,.5,.99]));return [float(x.mean()),float(x.std(unbiased=False)),float(q[0]),float(q[1]),float(q[2]),float(x.abs().max())]
def layer_gnorm(layer):
 return math.sqrt(sum(float(p.grad.detach().square().sum()) for p in layer.parameters() if p.grad is not None))
def save(c,m,x,y,xt,yt,step,global_grad,peak,worker):
 m.eval()
 with torch.no_grad():
  pred,ss=m(x,collect=True);tr=float(F.mse_loss(pred,y));te=float(F.mse_loss(m(xt),yt))
 cap=0.
 if c['act']=='precap':
  total=hit=0
  for pre,_,_ in ss:
   z=A*pre;u=z+GAMMA*z*z;v=SOFT_S*torch.tanh(u/SOFT_S);hit+=int((v>C).sum());total+=v.numel()
  cap=hit/total
 stamp=datetime.now().isoformat(timespec='seconds');r=rid(c)
 append(ROOT/'workers'/f'w{worker}_checkpoints.csv',MAIN,dict(run_id=r,**c,step=step,train_mse=tr,test_mse=te,global_grad_norm=global_grad,hmax_peak=peak,cap_fraction=cap,timestamp=stamp))
 for i,(pre,post,res) in enumerate(ss):
  g=layer_gnorm(m.layers[i]) if step>0 else 0.
  ratio=float(res.std(unbiased=False)/(post.std(unbiased=False)+1e-12))
  for tensor,name in [(pre,'pre'),(post,'post'),(res,'residual')]:
   a,b,c1,d,e,f=qstats(tensor)
   append(ROOT/'workers'/f'w{worker}_layers.csv',LAYER,dict(run_id=r,**c,step=step,layer=i,tensor=name,mean=a,std=b,p01=c1,p50=d,p99=e,max_abs=f,weight_grad_norm=g,residual_to_post_std=ratio))
 torch.save({'state_dict':m.state_dict(),'config':c,'step':step},ROOT/'snapshots'/f'{r}_step{step}.pt')
def execute(c,worker):
 random.seed(c['seed']);np.random.seed(c['seed']);torch.manual_seed(c['seed']);x,y=data(c['task'],2048,c['seed']);xt,yt=data(c['task'],512,c['seed']+10000)
 dl=DataLoader(TensorDataset(x,y),batch_size=BATCH,shuffle=True);it=iter(dl);m=Net(c['depth'],c['act']);opt=torch.optim.Adam(m.parameters(),lr=LR);peak=grad=0.;save(c,m,x,y,xt,yt,0,grad,peak,worker)
 for step in range(1,STEPS+1):
  try:xb,yb=next(it)
  except StopIteration:it=iter(dl);xb,yb=next(it)
  loss=F.mse_loss(m(xb),yb);opt.zero_grad();loss.backward();grad=math.sqrt(sum(float(p.grad.detach().square().sum()) for p in m.parameters() if p.grad is not None))
  if not math.isfinite(grad) or not torch.isfinite(loss):raise RuntimeError(f'nonfinite {rid(c)} step={step}')
  opt.step()
  with torch.no_grad():_,ss=m(xb,collect=True);peak=max(peak,max(float(z.abs().max()) for t in ss for z in t[:2]))
  if step in CHECKPOINTS:save(c,m,x,y,xt,yt,step,grad,peak,worker)
def selfcheck():
 x=torch.linspace(-5,5,10001);z=A*x;u=z+GAMMA*z*z;v=SOFT_S*torch.tanh(u/SOFT_S);ec=math.exp(C);ref=torch.where(v<=C,torch.exp(v)-1,ec-1+ec*(v-C));err=float((PreCap(C,A,GAMMA,SOFT_S)(x)-ref).abs().max());assert err<1e-6
 m=Net(4,'precap');out,ss=m(torch.randn(5,2),collect=True);assert out.shape==(5,1) and len(ss)==4;assert len(list(configs()))==60
 print(f'SELFCHECK_OK formula_max_abs_err={err:.3g} configs=60 checkpoints={CHECKPOINTS}')
def worker(shard,n):
 cs=[c for i,c in enumerate(configs()) if i%n==shard]
 done=set();p=ROOT/'workers'/f'w{shard}_checkpoints.csv'
 if p.exists():
  done={r['run_id'] for r in csv.DictReader(p.open()) if r['step']=='3000'}
 for i,c in enumerate(cs,1):
  if rid(c) in done:continue
  with (ROOT/'logs'/f'w{shard}.log').open('a') as f:f.write(f'start {i}/{len(cs)} {rid(c)}\n')
  execute(c,shard)
  with (ROOT/'logs'/f'w{shard}.log').open('a') as f:f.write(f'done {i}/{len(cs)} {rid(c)}\n')
def merge():
 ck=[];la=[]
 for i in range(4):
  for name,target in [('checkpoints',ck),('layers',la)]:
   p=ROOT/'workers'/f'w{i}_{name}.csv'
   if p.exists():target+=list(csv.DictReader(p.open()))
 for name,rows,fields in [('checkpoint_metrics.csv',ck,MAIN),('layer_metrics.csv',la,LAYER)]:
  with (ROOT/'results'/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
 full={r['run_id'] for r in ck if r['step']=='3000'};ckeys={(r['run_id'],r['step']) for r in ck};lkeys={(r['run_id'],r['step'],r['layer'],r['tensor']) for r in la}
 manifest=json.loads((ROOT/'meta'/'manifest.json').read_text())
 expected_runs=manifest['planned_runs']
 expected_checkpoint_rows=expected_runs*len(manifest['checkpoints'])
 expected_layer=sum(c['depth'] for c in manifest['cells'])*len(manifest['activations'])*len(manifest['fresh_seeds'])*len(manifest['checkpoints'])*3
 msg=f'full_runs={len(full)}/{expected_runs} checkpoint_rows={len(ck)}/{expected_checkpoint_rows} unique_checkpoint_keys={len(ckeys)} layer_rows={len(la)}/{expected_layer} unique_layer_keys={len(lkeys)}'
 (ROOT/'logs'/'integrity.txt').write_text(msg+'\n');print(msg)
 return len(full)==expected_runs and len(ck)==expected_checkpoint_rows and len(ckeys)==len(ck) and len(la)==expected_layer and len(lkeys)==len(la)
def freeze():
 p=[ROOT/'run_v5_depth64.py',ROOT/'meta'/'manifest.json',ROOT/'docs'/'PreCap_Formel_Freeze_v5.md'];(ROOT/'meta'/'frozen_sha256.json').write_text(json.dumps({x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in p},indent=2)+'\n')
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('cmd',choices=['selfcheck','worker','merge','freeze']);ap.add_argument('--shard',type=int,default=0);ap.add_argument('--n',type=int,default=4);a=ap.parse_args()
 if a.cmd=='selfcheck':selfcheck()
 elif a.cmd=='worker':worker(a.shard,a.n)
 elif a.cmd=='merge':raise SystemExit(0 if merge() else 1)
 else:freeze();print('FREEZE_OK')
