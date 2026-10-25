import argparse,csv,math,os
from datetime import datetime
import torch
import torch.nn as nn

CHECKPOINTS={1,2,5,10,20,50,100,200,400,800,1600,3000}
KEYS=['target','act','c','a','gamma','soft_s','knee','mix_beta','depth','arch','init_k','lr','seed']

def target(name,x):
    if name=='sin3_quad': return torch.sin(3*x)+0.30*x*x
    if name=='sin_mix': return torch.sin(2*x)+0.35*torch.sin(7*x)
    if name=='abs_wave': return torch.abs(torch.sin(3*x))+0.20*x
    raise ValueError(name)

class CapExp(nn.Module):
    def __init__(self,c,a):
        super().__init__(); self.c=float(c); self.a=float(a)
    def forward(self,x):
        z=self.a*x; ec=math.exp(self.c)
        return torch.where(z<=self.c, torch.exp(z)-1, ec-1+ec*(z-self.c))

class PreCap(nn.Module):
    def __init__(self,c,a,g,s):
        super().__init__(); self.c=float(c); self.a=float(a); self.g=float(g); self.s=float(s); self.cap=CapExp(c,1.0)
    def forward(self,x):
        z=self.a*x
        u=z+self.g*z*z
        return self.cap(self.s*torch.tanh(u/self.s))

class PreCapWG(nn.Module):
    def __init__(self,c,a,g,s,knee):
        super().__init__(); self.c=float(c); self.a=float(a); self.g=float(g); self.s=float(s); self.knee=float(knee); self.cap=CapExp(c,1.0)
    def squish(self,u):
        k=max(self.knee,1e-6)
        au=u.abs()
        over=torch.relu(au-k)
        comp=k + self.s*torch.log1p(over/self.s)
        out=torch.where(au<=k, au, comp)
        return torch.sign(u)*out
    def forward(self,x):
        z=self.a*x
        u=z+self.g*z*z
        return self.cap(self.squish(u))

class HybridGate(nn.Module):
    def __init__(self,c,a,g,s,knee,beta):
        super().__init__(); self.a=float(a); self.knee=float(knee); self.beta=float(beta)
        self.capexp=CapExp(c,a)
        self.precap=PreCap(c,a,g,s)
    def gate(self,z):
        t=(z.abs()-self.knee)/max(self.beta,1e-6)
        return torch.sigmoid(t)
    def forward(self,x):
        z=self.a*x
        g=self.gate(z)
        y_open=self.capexp(x)
        y_safe=self.precap(x)
        return (1.0-g)*y_open + g*y_safe

def make_act(name,c,a,g,s,knee,beta):
    if name=='capexp': return CapExp(c,a)
    if name=='precap': return PreCap(c,a,g,s)
    if name=='precap_wg': return PreCapWG(c,a,g,s,knee)
    if name=='hybrid_gate': return HybridGate(c,a,g,s,knee,beta)
    if name=='silu': return nn.SiLU()
    if name=='gelu': return nn.GELU()
    if name=='relu': return nn.ReLU()
    raise ValueError(name)

class Net(nn.Module):
    def __init__(self,name,c,a,g,s,knee,beta,depth,arch,init_k,width=64):
        super().__init__(); self.arch=arch; self.depth=depth; self.layers=nn.ModuleList(); din=1
        for _ in range(depth): self.layers.append(nn.Linear(din,width)); din=width
        self.out=nn.Linear(width,1); self.af=make_act(name,c,a,g,s,knee,beta)
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.normal_(m.weight,0,init_k/math.sqrt(m.in_features)); nn.init.zeros_(m.bias)
    def forward(self,x,collect=False):
        h=x; stats=[]
        for l in self.layers:
            pre=l(h); y=self.af(pre)
            if self.arch=='plain': h=y
            else:
                hskip=h.mean(dim=1,keepdim=True).expand_as(y) if h.shape[1]!=y.shape[1] else h
                h=hskip + y/math.sqrt(self.depth)
            if collect: stats.append((pre.detach(),h.detach()))
        out=self.out(h)
        return (out,stats) if collect else out

def finite(x):
    return math.isfinite(float(x.detach() if hasattr(x,'detach') else x))

def run(cfg):
    torch.manual_seed(cfg['seed']); torch.set_num_threads(1)
    x=torch.linspace(-2,2,512).unsqueeze(1); y=target(cfg['target'],x)
    xt=torch.linspace(-2.1,2.1,1024).unsqueeze(1); yt=target(cfg['target'],xt)
    m=Net(cfg['act'],cfg['c'],cfg['a'],cfg['gamma'],cfg['soft_s'],cfg['knee'],cfg['mix_beta'],cfg['depth'],cfg['arch'],cfg['init_k'])
    opt=torch.optim.Adam(m.parameters(),lr=cfg['lr'])
    curve=[]; bad=False; fail=-1
    for step in range(1,3001):
        opt.zero_grad(); pred=m(x); loss=((pred-y)**2).mean()
        if not finite(loss): bad=True; fail=step; break
        loss.backward()
        gn=math.sqrt(sum(float(p.grad.detach().square().sum()) for p in m.parameters() if p.grad is not None))
        if not math.isfinite(gn): bad=True; fail=step; break
        opt.step()
        if step in CHECKPOINTS:
            with torch.no_grad():
                _,ss=m(x,collect=True); tl=((m(xt)-yt)**2).mean()
                hstd=sum(float(h.std()) for _,h in ss)/len(ss)
                hmax=max(float(h.abs().max()) for _,h in ss)
                cap=0.0
                if cfg['act'] in ('capexp','precap','precap_wg','hybrid_gate'):
                    cap=sum(float((z>cfg['c']).float().mean()) for z,_ in ss)/len(ss)
            train_loss=float(loss.detach())
            test_loss=float(tl.detach())
            curve.append({'step':step,'train_loss':train_loss,'test_loss':test_loss,'grad_norm':gn,'act_std_mean':hstd,'act_abs_max':hmax,'cap_frac':cap,'bad':False})
            print(f"checkpoint step={step} target={cfg['target']} act={cfg['act']} depth={cfg['depth']} arch={cfg['arch']} train={train_loss:.6g} test={test_loss:.6g} grad={gn:.6g} hmax={hmax:.6g} cap={cap:.4f}", flush=True)
    final=math.inf
    if not bad:
        with torch.no_grad(): final=float(((m(xt)-yt)**2).mean())
    row={k:cfg[k] for k in KEYS}; row.update(timestamp=datetime.now().isoformat(timespec='seconds'),steps=(fail if bad else 3000),bad=bad,test_loss=final)
    return curve,final,bad,fail,row

def configs(mode='full'):
    for target_name in ['sin3_quad','sin_mix','abs_wave']:
        for depth in [8,16]:
            for arch in ['plain','res_inv']:
                for seed in range(10 if mode=='full' else 3):
                    for init_k in [0.5,1.0]:
                        for lr in [0.001,0.003]:
                            yield dict(target=target_name,act='silu',c=0.0,a=1.0,gamma=0.0,soft_s=1.0,knee=0.75,mix_beta=0.25,depth=depth,arch=arch,init_k=init_k,lr=lr,seed=seed)
                            yield dict(target=target_name,act='gelu',c=0.0,a=1.0,gamma=0.0,soft_s=1.0,knee=0.75,mix_beta=0.25,depth=depth,arch=arch,init_k=init_k,lr=lr,seed=seed)
                    for init_k,lr in [(0.5,0.001),(1.0,0.001),(1.0,0.003)]:
                        yield dict(target=target_name,act='precap',c=1.0,a=1.25,gamma=0.01,soft_s=1.5,knee=0.75,mix_beta=0.25,depth=depth,arch=arch,init_k=init_k,lr=lr,seed=seed)
                        yield dict(target=target_name,act='precap_wg',c=1.0,a=1.25,gamma=0.01,soft_s=1.5,knee=0.5,mix_beta=0.25,depth=depth,arch=arch,init_k=init_k,lr=lr,seed=seed)
                        yield dict(target=target_name,act='precap_wg',c=1.0,a=1.25,gamma=0.01,soft_s=1.5,knee=0.75,mix_beta=0.25,depth=depth,arch=arch,init_k=init_k,lr=lr,seed=seed)
                        yield dict(target=target_name,act='hybrid_gate',c=1.0,a=1.25,gamma=0.01,soft_s=1.5,knee=0.5,mix_beta=0.20,depth=depth,arch=arch,init_k=init_k,lr=lr,seed=seed)
                        yield dict(target=target_name,act='hybrid_gate',c=1.0,a=1.25,gamma=0.01,soft_s=1.5,knee=0.75,mix_beta=0.30,depth=depth,arch=arch,init_k=init_k,lr=lr,seed=seed)
    for target_name in ['sin3_quad','sin_mix','abs_wave']:
        for depth in [32,64]:
            for seed in range(10 if mode=='full' else 3):
                for init_k in [0.5]:
                    for lr in [0.001]:
                        yield dict(target=target_name,act='precap',c=1.0,a=1.25,gamma=0.01,soft_s=1.5,knee=0.75,mix_beta=0.25,depth=depth,arch='res_inv',init_k=init_k,lr=lr,seed=seed)
                        yield dict(target=target_name,act='precap_wg',c=1.0,a=1.25,gamma=0.01,soft_s=1.5,knee=0.5,mix_beta=0.25,depth=depth,arch='res_inv',init_k=init_k,lr=lr,seed=seed)
                        yield dict(target=target_name,act='precap_wg',c=1.0,a=1.25,gamma=0.01,soft_s=1.5,knee=0.75,mix_beta=0.25,depth=depth,arch='res_inv',init_k=init_k,lr=lr,seed=seed)
                        yield dict(target=target_name,act='hybrid_gate',c=1.0,a=1.25,gamma=0.01,soft_s=1.5,knee=0.5,mix_beta=0.20,depth=depth,arch='res_inv',init_k=init_k,lr=lr,seed=seed)
                        yield dict(target=target_name,act='hybrid_gate',c=1.0,a=1.25,gamma=0.01,soft_s=1.5,knee=0.75,mix_beta=0.30,depth=depth,arch='res_inv',init_k=init_k,lr=lr,seed=seed)

def append_csv(path,row,fieldnames):
    write_header=not os.path.exists(path)
    with open(path,'a',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fieldnames)
        if write_header: w.writeheader()
        w.writerow(row)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',default='v11_results.csv')
    ap.add_argument('--shard',type=int,default=0)
    ap.add_argument('--nshards',type=int,default=1)
    ap.add_argument('--mode',choices=['mini','full'],default='full')
    args=ap.parse_args()
    all_cfgs=list(configs(args.mode))
    cfgs=[c for i,c in enumerate(all_cfgs) if i % args.nshards == args.shard]
    total=len(cfgs)
    for j,cfg in enumerate(cfgs,1):
        curve,final,bad,fail,row=run(cfg)
        append_csv(args.out,row,list(row.keys()))
        cp=os.path.splitext(args.out)[0]+'_curves.csv'
        for r in curve:
            rr={k:cfg[k] for k in KEYS}; rr.update(r); append_csv(cp,rr,list(rr.keys()))
        print(f"[{j}/{total}] {cfg['target']} {cfg['act']} d={cfg['depth']} {cfg['arch']} knee={cfg['knee']} beta={cfg['mix_beta']} test={final:.6g} bad={bad}", flush=True)

if __name__=='__main__':
    main()
