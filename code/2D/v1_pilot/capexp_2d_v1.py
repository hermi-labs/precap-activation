import argparse,csv,math,os,random,time
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

torch.set_num_threads(max(1, min(8, (os.cpu_count() or 4)//2)))
DEVICE='cpu'
WIDTH=64
BATCH=128
STEPS=2500
LOG_STEPS={50,100,200,400,800,1600,2500}
SEEDS=list(range(300,305))
DEPTHS=[4,8,16]
ACTS=['precap','silu','gelu']
TASKS=['radial_ripple','ridge_mix','corner_kink']
FIELDS=['task','act','depth','arch','width','seed','timestamp','steps','bad','test_loss','grad_final','hmax_final','hmax_peak','cap_final']

class PreCap(nn.Module):
    def __init__(self,c=1.0,a=1.25,gamma=0.01,soft_s=1.5,knee=0.75):
        super().__init__(); self.c=c; self.a=a; self.gamma=gamma; self.soft_s=soft_s; self.knee=knee; self.last_cap_frac=0.0
    def forward(self,x):
        y=F.silu(x)*self.a
        ax=x.abs(); cap=F.softplus((ax-self.knee)*self.soft_s)/self.soft_s
        self.last_cap_frac=(ax>self.knee).float().mean().item()
        return y - self.c*cap*x.sign() + self.gamma*x

def make_act(name):
    if name=='precap': return PreCap()
    if name=='silu': return nn.SiLU()
    if name=='gelu': return nn.GELU()
    raise KeyError(name)

class Block(nn.Module):
    def __init__(self,w,act):
        super().__init__(); self.lin=nn.Linear(w,w); self.act=act
    def forward(self,x): return x + self.act(self.lin(x))

class Net(nn.Module):
    def __init__(self,depth,act_name,width=64):
        super().__init__(); self.inp=nn.Linear(2,width); self.blocks=nn.ModuleList([Block(width,make_act(act_name)) for _ in range(depth)]); self.out=nn.Linear(width,1)
    def forward(self,x):
        h=self.inp(x); hmax=0.0; cap=[]
        for b in self.blocks:
            h=b(h); hmax=max(hmax,h.abs().max().item());
            if hasattr(b.act,'last_cap_frac'): cap.append(b.act.last_cap_frac)
        return self.out(h), hmax, (sum(cap)/len(cap) if cap else 0.0)

def task_fn(name,x,y):
    if name=='radial_ripple': return torch.sin(3*(x*x+y*y)) + 0.3*x*y
    if name=='ridge_mix': return torch.sin(4*x)+0.5*torch.cos(3*y)+0.2*(x+y)**2
    if name=='corner_kink': return torch.relu(x+y-0.2)+0.3*torch.abs(x-y)
    raise KeyError(name)

def make_data(task,n,seed):
    g=torch.Generator().manual_seed(seed)
    pts=torch.empty(n,2).uniform_(-1,1, generator=g)
    z=task_fn(task,pts[:,0],pts[:,1]).unsqueeze(1)
    return pts,z

def run(cfg):
    seed=cfg['seed']; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    trX,trY=make_data(cfg['task'],2048,seed); teX,teY=make_data(cfg['task'],512,seed+10000)
    dl=DataLoader(TensorDataset(trX,trY),batch_size=BATCH,shuffle=True)
    m=Net(cfg['depth'],cfg['act'],cfg['width']).to(DEVICE)
    opt=torch.optim.Adam(m.parameters(),lr=1e-3)
    peak=0.0; rows=[]; bad=False
    it=iter(dl)
    for step in range(1,STEPS+1):
        try: xb,yb=next(it)
        except StopIteration: it=iter(dl); xb,yb=next(it)
        pred,hmax,cap=m(xb); loss=F.mse_loss(pred,yb)
        if not torch.isfinite(loss): bad=True; break
        opt.zero_grad(); loss.backward()
        grad=max((p.grad.detach().abs().max().item() for p in m.parameters() if p.grad is not None), default=0.0)
        if not math.isfinite(grad): bad=True; break
        opt.step(); peak=max(peak,hmax)
        if step in LOG_STEPS:
            with torch.no_grad():
                tp,th,tc=m(teX); tl=F.mse_loss(tp,teY).item()
            rows.append(dict(step=step,train_loss=float(loss.item()),test_loss=float(tl),grad_norm=float(grad),act_abs_max=float(th),cap_frac=float(tc)))
            print(f"checkpoint step={step} task={cfg['task']} act={cfg['act']} depth={cfg['depth']} train={loss.item():.6g} test={tl:.6g} grad={grad:.6g} hmax={th:.6g} cap={tc:.4f}", flush=True)
    with torch.no_grad():
        tp,th,tc=m(teX); tl=F.mse_loss(tp,teY).item()
    row=dict(task=cfg['task'],act=cfg['act'],depth=cfg['depth'],arch='res_inv',width=cfg['width'],seed=cfg['seed'],timestamp=datetime.now().isoformat(timespec='seconds'),steps=STEPS,bad=bool(bad),test_loss=float('inf') if bad else float(tl),grad_final=rows[-1]['grad_norm'] if rows else float('nan'),hmax_final=rows[-1]['act_abs_max'] if rows else float('nan'),hmax_peak=float(peak),cap_final=rows[-1]['cap_frac'] if rows else float('nan'))
    return rows,row

def configs(): return [dict(task=t,act=a,depth=d,arch='res_inv',width=WIDTH,seed=s) for s in SEEDS for t in TASKS for d in DEPTHS for a in ACTS]
def append(path,row):
    ex=os.path.exists(path)
    with open(path,'a',newline='') as f:
        w=csv.DictWriter(f,fieldnames=FIELDS)
        if not ex: w.writeheader()
        w.writerow(row)
def done(path):
    if not os.path.exists(path): return set()
    with open(path,newline='') as f: return {(r['task'],r['act'],r['depth'],r['arch'],r['width'],r['seed']) for r in csv.DictReader(f)}
def worker(shard,n):
    here=os.path.dirname(os.path.abspath(__file__)); out=f'{here}/results/w{shard}.csv'; have=done(out); cs=[c for i,c in enumerate(configs()) if i%n==shard]
    for j,c in enumerate(cs,1):
        k=(c['task'],c['act'],str(c['depth']),c['arch'],str(c['width']),str(c['seed']))
        if k in have: continue
        t=time.time(); _,row=run(c); append(out,row)
        with open(f'{here}/logs/progress.log','a') as f: f.write(f"{datetime.now():%F %T} w{shard} {j}/{len(cs)} {c['task']} {c['act']} d={c['depth']} seed={c['seed']} test={row['test_loss']:.4g} bad={row['bad']} {time.time()-t:.0f}s\n")
def merge():
    here=os.path.dirname(os.path.abspath(__file__)); rows=[]
    for i in range(7):
        p=f'{here}/results/w{i}.csv'
        if os.path.exists(p):
            with open(p,newline='') as f: rows+=list(csv.DictReader(f))
    out=f'{here}/v1_results.csv'
    with open(out,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
    ks={(r['task'],r['act'],r['depth'],r['arch'],r['width'],r['seed']) for r in rows}
    msg=f"rows={len(rows)} expected={len(configs())} unique={len(ks)} bad={sum(r['bad']=='True' for r in rows)}"
    print(msg); open(f'{here}/logs/integrity.txt','w').write(msg+'\n'); return len(rows)==len(configs()) and len(ks)==len(configs())
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('cmd',choices=['worker','merge','count','smoke']); ap.add_argument('--shard',type=int,default=0); ap.add_argument('--n',type=int,default=7); a=ap.parse_args()
    if a.cmd=='worker': worker(a.shard,a.n)
    elif a.cmd=='merge': raise SystemExit(0 if merge() else 1)
    elif a.cmd=='count': print('total',len(configs()))
    else:
        for act in ACTS: _,row=run(dict(task='radial_ripple',act=act,depth=4,arch='res_inv',width=64,seed=300)); print('SMOKE',act,row['test_loss'],row['bad'])
