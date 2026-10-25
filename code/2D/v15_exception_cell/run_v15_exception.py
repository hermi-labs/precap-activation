import argparse,csv,json,math,random,re
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset
torch.set_num_threads(1)
ROOT=Path(__file__).resolve().parent
OUTDIR=ROOT
WIDTH=64; STEPS=1200; BATCH=128; INIT_K=.5; LR=.001; A=1.; GAMMA=.01
ACTS=["precap_base","precap_s225"]; SEEDS=list(range(9500,9600)); CHECKPOINTS=[0,50,100,200,400,800,1200]
V10_SEEDS=list(range(400,410))+list(range(500,510))+list(range(900,905))+list(range(5000,5020))+list(range(6000,6020))+list(range(7000,7020))+list(range(8000,8030))+list(range(9000,9020))
TASKS=[("freq","rot_ang53_f150_a025")]
DEPTHS=[32]; PARAMS={"precap_base":(1.,1.5),"precap_s225":(1.,2.25)}
SMOKE=False
# NOTE: column hmax_peak is the running maximum over training steps of max |network output| on the training set (kept name for V10 compatibility).
MAIN=["run_id","group","task","depth","act","seed","step","train_mse","test_mse","global_grad_norm","hmax_peak","cap_fraction","cap_c","soft_s","layer_pre_p01","layer_pre_p50","layer_pre_p99","layer_post_p01","layer_post_p50","layer_post_p99","pre_p99_first","pre_p99_mid","pre_p99_last","timestamp"]
class CapExp(nn.Module):
 def __init__(self,c,a): super().__init__();self.c=c;self.a=a
 def forward(self,x):
  z=self.a*x;ec=math.exp(self.c);return torch.where(z<=self.c,torch.exp(z)-1,ec-1+ec*(z-self.c))
class PreCap(nn.Module):
 def __init__(self,c,a,g,s): super().__init__();self.c=c;self.a=a;self.g=g;self.s=s;self.cap=CapExp(c,1.)
 def forward(self,x):
  z=self.a*x;u=z+self.g*z*z;return self.cap(self.s*torch.tanh(u/self.s))
def act(n):
 c,s=PARAMS[n];return PreCap(c,A,GAMMA,s)
def fn(task,x,y):
 m=re.fullmatch(r'rot_ang(\d+)_f(\d+)_a(\d+)',task)
 if not m: raise ValueError(task)
 k=int(m.group(2))/100.;a=int(m.group(3))/100.
 c,sn={'53':(.6,.8),'00':(1.,0.)}[m.group(1)];r2=x*x+y*y;u=c*x+sn*y;v=sn*x-c*y
 return torch.sin(k*(2*(1-a)*r2+5*a*u))+.3*torch.cos(k*(3*(1-a)*torch.sqrt(r2+.05)+2*a*v))
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
  h=x;pres=[];posts=[]
  for layer in self.layers:
   pre=layer(h);post=self.af(pre);skip=h.mean(1,keepdim=True).expand_as(post) if h.shape[1]!=post.shape[1] else h;h=skip+post/math.sqrt(self.depth)
   if collect: pres.append(pre.detach()); posts.append(post.detach())
  y=self.out(h);return (y,pres,posts) if collect else y
def configs():
 for group,task in TASKS:
  for depth in DEPTHS:
   for seed in SEEDS:
    for name in ACTS: yield {"group":group,"task":task,"depth":depth,"seed":seed,"act":name}
def rid(c): return f"{c['task']}_d{c['depth']}_{c['act']}_s{c['seed']}"
def append(path,row):
 path.parent.mkdir(parents=True,exist_ok=True)
 new=not path.exists()
 with path.open('a',newline='') as f:
  w=csv.DictWriter(f,fieldnames=MAIN)
  if new:w.writeheader()
  w.writerow(row)
def qstats(tensors):
 x=torch.cat([t.flatten() for t in tensors]).float();qs=torch.quantile(x,torch.tensor([.01,.5,.99]));return tuple(float(v) for v in qs.tolist())
def p99(t): return float(torch.quantile(t.flatten().float(),torch.tensor(.99)))
def cap_hits(pres,cv,sv):
 hit=tot=0
 for pre in pres:
  v=sv*torch.tanh((A*pre+GAMMA*(A*pre)**2)/sv);hit+=int((v>cv).sum());tot+=v.numel()
 return hit,tot
def save(c,m,x,y,xt,yt,step,grad,peak,wid):
 m.eval()
 with torch.no_grad():
  pred,pres,posts=m(x,collect=True);tr=float(F.mse_loss(pred,y));te=float(F.mse_loss(m(xt),yt));cv,sv=PARAMS[c['act']]
  hit,tot=cap_hits(pres,cv,sv);cap=hit/tot if tot else 0.
  pre01,pre50,pre99=qstats(pres); post01,post50,post99=qstats(posts)
  pf,pm,pl=p99(pres[0]),p99(pres[len(pres)//2]),p99(pres[-1])
  append(OUTDIR/'workers'/f'w{wid}_checkpoints.csv',dict(run_id=rid(c),**c,step=step,train_mse=tr,test_mse=te,global_grad_norm=grad,hmax_peak=peak,cap_fraction=cap,cap_c=cv,soft_s=sv,layer_pre_p01=pre01,layer_pre_p50=pre50,layer_pre_p99=pre99,layer_post_p01=post01,layer_post_p50=post50,layer_post_p99=post99,pre_p99_first=pf,pre_p99_mid=pm,pre_p99_last=pl,timestamp=datetime.now().isoformat(timespec='seconds')))
def execute(c,wid):
 random.seed(c['seed']);np.random.seed(c['seed']);torch.manual_seed(c['seed']);x,y=data(c['task'],2048,c['seed']);xt,yt=data(c['task'],512,c['seed']+10000);dl=DataLoader(TensorDataset(x,y),batch_size=BATCH,shuffle=True);it=iter(dl);m=Net(c['depth'],c['act']);o=torch.optim.Adam(m.parameters(),lr=LR);peak=grad=0.;save(c,m,x,y,xt,yt,0,grad,peak,wid)
 for step in range(1,STEPS+1):
  try: xb,yb=next(it)
  except StopIteration: it=iter(dl); xb,yb=next(it)
  m.train();o.zero_grad(); loss=F.mse_loss(m(xb),yb); loss.backward(); grad=math.sqrt(sum(float(p.grad.detach().square().sum()) for p in m.parameters() if p.grad is not None)); o.step()
  with torch.no_grad(): peak=max(peak,float(m(x).abs().max()))
  if step in CHECKPOINTS: save(c,m,x,y,xt,yt,step,grad,peak,wid)
def ref_precap(x,c,s,g=GAMMA):
 u=x+g*x*x;v=s*np.tanh(u/s);ec=math.exp(c);return np.where(v<=c,np.exp(v)-1,ec-1+ec*(v-c))
def selfcheck():
 out={}
 xs=np.linspace(-6,6,2001);worst=0.;mono=True
 for n,(c,s) in PARAMS.items():
  m=PreCap(c,1.,GAMMA,s).double();t=torch.tensor(xs,dtype=torch.float64);got=m(t).numpy();ref=ref_precap(xs,c,s);worst=max(worst,float(np.abs(got-ref).max()));mono=mono and bool((np.diff(got)>=-1e-12).all())
 out['formula_max_abs_err']=worst;out['monotone_on_grid_pm6']=mono
 cap=CapExp(1.,1.).double();eps=1e-9;lo=cap(torch.tensor([1.-eps],dtype=torch.float64)).item();hi=cap(torch.tensor([1.+eps],dtype=torch.float64)).item()
 out['cap_continuity_gap']=abs(hi-lo)
 dl=(cap(torch.tensor([1.-1e-6],dtype=torch.float64))-cap(torch.tensor([1.-2e-6],dtype=torch.float64))).item()/1e-6;dr=(cap(torch.tensor([1.+2e-6],dtype=torch.float64))-cap(torch.tensor([1.+1e-6],dtype=torch.float64))).item()/1e-6
 out['cap_slope_left']=dl;out['cap_slope_right']=dr;out['cap_slope_gap']=abs(dl-dr);out['exp_c']=math.exp(1.)
 sup={}
 for n,(c,s) in PARAMS.items():
  bound=math.exp(c)-1+math.exp(c)*(s-c);big=torch.tensor([-1e3,-50.,50.,1e3],dtype=torch.float64);val=PreCap(c,1.,GAMMA,s).double()(big).abs().max().item();sup[n]={'analytic_sup':bound,'max_seen_at_extremes':val,'ok':val<=bound+1e-9}
 out['output_supremum']=sup
 g=torch.Generator().manual_seed(1);pre=torch.randn(512,64,generator=g)*2.;h1,t1=cap_hits([pre],1.,1.5);v=1.5*np.tanh((pre.numpy()+GAMMA*pre.numpy()**2)/1.5);out['cap_hits_match']=bool(h1==int((v>1.).sum()) and t1==pre.numel())
 tq=torch.tensor(np.random.RandomState(0).randn(1000),dtype=torch.float32);q=qstats([tq]);nq=np.quantile(tq.numpy(),[.01,.5,.99]);out['quantile_max_abs_err']=float(np.abs(np.array(q)-nq).max())
 tasks={}
 for _,t in TASKS:
  x,y=data(t,2048,6000);x2,y2=data(t,2048,6000);tasks[t]={'finite':bool(torch.isfinite(y).all()),'deterministic':bool(torch.equal(y,y2)),'y_min':float(y.min()),'y_max':float(y.max()),'y_std':float(y.std())}
 out['tasks_data']=tasks
 a=sorted(set(SEEDS)|{s+10000 for s in SEEDS});b=sorted(set(V10_SEEDS)|{s+10000 for s in V10_SEEDS});out['seed_overlap_with_v10']=sorted(set(a)&set(b))
 out['expected_runs']=len(list(configs()));out['expected_checkpoint_rows']=out['expected_runs']*len(CHECKPOINTS)
 return out
def integrity():
 rows=[]
 for f in sorted((OUTDIR/'workers').glob('w*_checkpoints.csv')): rows+=list(csv.DictReader(open(f)))
 cf=list(configs());exp=len(cf)*len(CHECKPOINTS); keys={(r['run_id'],r['step']) for r in rows}; full={r['run_id'] for r in rows if r['step']==str(STEPS)}; return {"checkpoint_rows":len(rows),"expected_checkpoint_rows":exp,"unique_checkpoint_keys":len(keys),"full_runs":len(full),"expected_full_runs":len(cf)}
def main():
 global STEPS,CHECKPOINTS,OUTDIR,SMOKE
 p=argparse.ArgumentParser(); p.add_argument('--worker',type=int); p.add_argument('--workers',type=int,default=1); p.add_argument('--selfcheck',action='store_true'); p.add_argument('--integrity',action='store_true'); p.add_argument('--smoke',action='store_true'); a=p.parse_args()
 if a.smoke: SMOKE=True;STEPS=10;CHECKPOINTS=[0,5,10];OUTDIR=ROOT/'smoke'
 if a.selfcheck: print(json.dumps({"configs":len(list(configs())),"selfcheck":selfcheck(),"tasks":TASKS,"depths":DEPTHS,"acts":ACTS,"seeds":SEEDS,"checkpoints":CHECKPOINTS},indent=2)); return
 if a.integrity: print(json.dumps(integrity(),indent=2)); return
 for i,c in enumerate(configs()):
  if SMOKE and not (c['depth']==16 and c['seed']==SEEDS[0]): continue
  if i%a.workers==a.worker: execute(c,a.worker); print('done',rid(c),flush=True)
if __name__=='__main__': main()
