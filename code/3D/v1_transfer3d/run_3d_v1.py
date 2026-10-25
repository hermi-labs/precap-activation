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
WIDTH=64; STEPS=3000; BATCH=128; INIT_K=.5; A=1.; GAMMA=.01
LRS=[1e-3,3e-3,1e-2]
ACTS=["precap_base","capexp_c1","silu","silu2x"]; SEEDS=list(range(30000,30030)); TUNE_SEEDS=SEEDS[:10]; CONF_SEEDS=SEEDS[10:]; CHECKPOINTS=[0,100,200,400,800,1200,2000,3000]
SCALES=["inv_sqrt"]
V10_SEEDS=list(range(400,410))+list(range(500,510))+list(range(900,905))+list(range(5000,5020))+list(range(6000,6020))+list(range(7000,7020))+list(range(8000,8030))+list(range(9000,9020))+list(range(9500,9600))+list(range(9700,9720))+list(range(9800,9820))+list(range(9900,9930))
TASKS=[("3d",n) for n in ["smooth_bowl3d","gyroid","torus_sdf","box_sdf","spheres_union"]]
NORMS=["none","layernorm"]; NTRAIN=4096; NTEST=1024; IN_DIM=3
DEPTHS=[8,16]; PARAMS={"precap_base":(1.,1.5),"precap_s225":(1.,2.25)}
SMOKE=False
PILOT2=False
# NOTE V16: column hmax_peak = max |network output| on the training set AT the checkpoint (no per-step full-train forward, ca. 5x faster).
MAIN=["run_id","group","task","depth","scale","norm","lr","act","seed","step","train_mse","test_mse","global_grad_norm","hmax_peak","cap_fraction","cap_c","soft_s","layer_pre_p01","layer_pre_p50","layer_pre_p99","layer_post_p01","layer_post_p50","layer_post_p99","pre_p99_first","pre_p99_mid","pre_p99_last","pre_rms","post_rms_scaled","frac_abs_pre_gt_half","residual_scale","run_status","timestamp"]
class CapExp(nn.Module):
 def __init__(self,c,a): super().__init__();self.c=c;self.a=a
 def forward(self,x):
  z=self.a*x;ec=math.exp(self.c);return torch.where(z<=self.c,torch.exp(z)-1,ec-1+ec*(z-self.c))
class PreCap(nn.Module):
 def __init__(self,c,a,g,s): super().__init__();self.c=c;self.a=a;self.g=g;self.s=s;self.cap=CapExp(c,1.)
 def forward(self,x):
  z=self.a*x;u=z+self.g*z*z;return self.cap(self.s*torch.tanh(u/self.s))
class Silu2x(nn.Module):
 # 3D-V1 Kontrolle: 2*SiLU hat am Ursprung Steigung 1 wie PreCap
 def forward(self,x): return 2.*F.silu(x)
def act(n):
 if n=="silu": return nn.SiLU()
 if n=="silu2x": return Silu2x()
 if n=="capexp_c1": return CapExp(1.,1.)
 if n=="gelu": return nn.GELU()
 c,s=PARAMS[n];return PreCap(c,A,GAMMA,s)
def fn(n,x,y,z):
 if n=='smooth_bowl3d': return 0.40*(x*x+y*y+z*z)+0.15*torch.sin(2*x-y+z)
 if n=='gyroid': w=2*math.pi*0.75; return 0.5*(torch.sin(w*x)*torch.cos(w*y)+torch.sin(w*y)*torch.cos(w*z)+torch.sin(w*z)*torch.cos(w*x))
 if n=='torus_sdf': q=torch.sqrt(x*x+z*z)-0.5; return torch.sqrt(q*q+y*y)-0.2
 if n=='box_sdf':
  q=torch.stack([x.abs(),y.abs(),z.abs()],-1)-0.5; return torch.linalg.norm(q.clamp(min=0),dim=-1)+q.max(-1).values.clamp(max=0)
 if n=='spheres_union':
  a=torch.sqrt((x+0.35)**2+y*y+z*z)-0.4; b=torch.sqrt((x-0.35)**2+(y-0.1)**2+z*z)-0.4; return torch.minimum(a,b)
 raise ValueError(n)
def data(task,n,seed):
 g=torch.Generator().manual_seed(seed);x=torch.empty(n,IN_DIM).uniform_(-1,1,generator=g);return x,fn(task,x[:,0],x[:,1],x[:,2]).unsqueeze(1)
class Net(nn.Module):
 # 3D-V1: wie 2D res_inv (h = skip + act(W h) * 1/sqrt(D)); optional Pre-LayerNorm vor jeder versteckten Schicht und vor dem Ausgang.
 def __init__(self,depth,name,scale,norm='none'):
  super().__init__();self.depth=depth;self.scale_name=scale;self.norm=norm;self.res_scale={"inv_sqrt":1/math.sqrt(depth),"inv_quart":depth**(-.25),"constant":1.}[scale];self.layers=nn.ModuleList();d=IN_DIM
  for _ in range(depth):self.layers.append(nn.Linear(d,WIDTH));d=WIDTH
  self.norms=nn.ModuleList([nn.LayerNorm(WIDTH) if (norm=='layernorm' and i>0) else nn.Identity() for i in range(depth)])
  self.out_norm=nn.LayerNorm(WIDTH) if norm=='layernorm' else nn.Identity()
  self.out=nn.Linear(WIDTH,1);self.af=act(name)
  for m in self.modules():
   if isinstance(m,nn.Linear):nn.init.normal_(m.weight,0,INIT_K/math.sqrt(m.in_features));nn.init.zeros_(m.bias)
 def forward(self,x,collect=False):
  h=x;pres=[];posts=[]
  for layer,nm in zip(self.layers,self.norms):
   pre=layer(nm(h));post=self.af(pre);skip=h.mean(1,keepdim=True).expand_as(post) if h.shape[1]!=post.shape[1] else h;h=skip+post*self.res_scale
   if collect: pres.append(pre.detach()); posts.append(post.detach())
  y=self.out(self.out_norm(h));return (y,pres,posts) if collect else y
def configs():
 for group,task in TASKS:
  for depth in DEPTHS:
   for scale in SCALES:
    for norm in NORMS:
     for lr in LRS:
      for seed in SEEDS:
       for name in ACTS: yield {"group":group,"task":task,"depth":depth,"scale":scale,"norm":norm,"lr":lr,"seed":seed,"act":name}
def rid(c): return f"{c['task']}_d{c['depth']}_{c['scale']}_{c['norm']}_lr{c['lr']:g}_{c['act']}_s{c['seed']}"
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
  pred,pres,posts=m(x,collect=True);tr=float(F.mse_loss(pred,y));te=float(F.mse_loss(m(xt),yt));cv,sv=PARAMS.get(c['act'],(float('nan'),float('nan')))
  hit,tot=cap_hits(pres,cv,sv);cap=hit/tot if tot else 0.
  if c['act']=='capexp_c1': cap=float(sum(int((p>1.).sum()) for p in pres))/max(1,sum(p.numel() for p in pres))
  pre01,pre50,pre99=qstats(pres); post01,post50,post99=qstats(posts)
  pf,pm,pl=p99(pres[0]),p99(pres[len(pres)//2]),p99(pres[-1])
  D=float(len(pres));pre_rms=float(torch.stack([p.float().square().mean() for p in pres]).mean().sqrt());post_rms=float(torch.stack([q.float().square().mean().sqrt() for q in posts]).mean())/math.sqrt(D);fh=float(torch.stack([(p.abs()>0.5).float().mean() for p in pres]).mean());hm=float(pred.abs().max())
  return dict(run_id=rid(c),**c,step=step,train_mse=tr,test_mse=te,global_grad_norm=grad,hmax_peak=hm,cap_fraction=cap,cap_c=cv,soft_s=sv,layer_pre_p01=pre01,layer_pre_p50=pre50,layer_pre_p99=pre99,layer_post_p01=post01,layer_post_p50=post50,layer_post_p99=post99,pre_p99_first=pf,pre_p99_mid=pm,pre_p99_last=pl,pre_rms=pre_rms,post_rms_scaled=post_rms,frac_abs_pre_gt_half=fh,residual_scale=m.res_scale,run_status="ok",timestamp=datetime.now().isoformat(timespec='seconds'))
def execute(c,wid):
 random.seed(c['seed']);np.random.seed(c['seed']);torch.manual_seed(c['seed']);x,y=data(c['task'],NTRAIN,c['seed']);xt,yt=data(c['task'],NTEST,c['seed']+10000);dl=DataLoader(TensorDataset(x,y),batch_size=BATCH,shuffle=True);it=iter(dl);m=Net(c['depth'],c['act'],c['scale'],c['norm']);o=torch.optim.Adam(m.parameters(),lr=c['lr']);peak=grad=0.;buf=[save(c,m,x,y,xt,yt,0,grad,peak,wid)]
 for step in range(1,STEPS+1):
  try: xb,yb=next(it)
  except StopIteration: it=iter(dl); xb,yb=next(it)
  m.train();o.zero_grad(); yp=m(xb); loss=F.mse_loss(yp,yb)
  if not torch.isfinite(loss):
   buf.append(dict(run_id=rid(c),**c,step=step,train_mse=float('nan'),test_mse=float('nan'),global_grad_norm=float('nan'),hmax_peak=float('nan'),cap_fraction=float('nan'),cap_c=PARAMS.get(c['act'],(float('nan'),float('nan')))[0],soft_s=PARAMS.get(c['act'],(float('nan'),float('nan')))[1],layer_pre_p01=float('nan'),layer_pre_p50=float('nan'),layer_pre_p99=float('nan'),layer_post_p01=float('nan'),layer_post_p50=float('nan'),layer_post_p99=float('nan'),pre_p99_first=float('nan'),pre_p99_mid=float('nan'),pre_p99_last=float('nan'),pre_rms=float('nan'),post_rms_scaled=float('nan'),frac_abs_pre_gt_half=float('nan'),residual_scale=m.res_scale,run_status='nonfinite_loss',timestamp=datetime.now().isoformat(timespec='seconds')))
   flush(buf,wid); return
  loss.backward(); grad=math.sqrt(sum(float(p.grad.detach().square().sum()) for p in m.parameters() if p.grad is not None)); o.step()
  if step in CHECKPOINTS: buf.append(save(c,m,x,y,xt,yt,step,grad,peak,wid))
 flush(buf,wid)
def flush(buf,wid):
 for row in buf: append(OUTDIR/'workers'/f'w{wid}_checkpoints.csv',row)
def finished_ids():
 done=set()
 for f in (OUTDIR/'workers').glob('w*_checkpoints.csv'):
  for r in csv.DictReader(open(f)):
   if r['step']==str(STEPS) or r['run_status']!='ok': done.add(r['run_id'])
 return done
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
  x,y=data(t,NTRAIN,6000);x2,y2=data(t,NTRAIN,6000);tasks[t]={'finite':bool(torch.isfinite(y).all()),'deterministic':bool(torch.equal(y,y2)),'y_min':float(y.min()),'y_max':float(y.max()),'y_std':float(y.std())}
 out['tasks_data']=tasks
 a=sorted(set(SEEDS)|{s+10000 for s in SEEDS});b=sorted(set(V10_SEEDS)|{s+10000 for s in V10_SEEDS});out['seed_overlap_with_v10']=sorted(set(a)&set(b))
 out['expected_runs']=len(list(configs()));out['expected_checkpoint_rows']=out['expected_runs']*len(CHECKPOINTS)
 return out
def integrity():
 rows=[]
 for f in sorted((OUTDIR/'workers').glob('w*_checkpoints.csv')): rows+=list(csv.DictReader(open(f)))
 cf=list(configs());exp=len(cf)*len(CHECKPOINTS); keys={(r['run_id'],r['step']) for r in rows}; full={r['run_id'] for r in rows if r['step']==str(STEPS)}; nonfinite={r['run_id'] for r in rows if r['run_status']!='ok'}; return {"checkpoint_rows":len(rows),"expected_checkpoint_rows":exp,"unique_checkpoint_keys":len(keys),"full_runs":len(full),"nonfinite_runs":len(nonfinite),"expected_full_runs":len(cf)}
def main():
 global STEPS,CHECKPOINTS,OUTDIR,SMOKE,PILOT2,PILOT2
 p=argparse.ArgumentParser(); p.add_argument('--worker',type=int); p.add_argument('--workers',type=int,default=1); p.add_argument('--selfcheck',action='store_true'); p.add_argument('--integrity',action='store_true'); p.add_argument('--smoke',action='store_true'); p.add_argument('--pilot2',action='store_true'); a=p.parse_args()
 if a.smoke: SMOKE=True;STEPS=10;CHECKPOINTS=[0,5,10];OUTDIR=ROOT/'smoke'
 if a.pilot2: PILOT2=True; OUTDIR=ROOT/'pilot2'
 if a.pilot2: PILOT2=True; OUTDIR=ROOT/'pilot2'
 if a.selfcheck: print(json.dumps({"configs":len(list(configs())),"selfcheck":selfcheck(),"tasks":TASKS,"depths":DEPTHS,"scales":SCALES,"norms":NORMS,"lrs":LRS,"acts":ACTS,"seeds":SEEDS,"checkpoints":CHECKPOINTS},indent=2)); return
 if a.integrity: print(json.dumps(integrity(),indent=2)); return
 skip=finished_ids()
 for i,c in enumerate(configs()):
  if SMOKE and not (c['depth']==8 and c['seed']==SEEDS[0] and c['lr']==1e-3): continue
  if PILOT2 and c['seed'] not in SEEDS[:2]: continue
  if i%a.workers==a.worker:
   if rid(c) in skip: print('skip (bereits fertig)',rid(c),flush=True); continue
   execute(c,a.worker); print('done',rid(c),datetime.now().strftime('%H:%M:%S'),flush=True)
if __name__=='__main__': main()
