import argparse,csv,math,os,time
from datetime import datetime
import torch
import torch.nn as nn

CHECKPOINTS={1,2,5,10,20,50,100,200,400,800,1600,3000}
KEYS=['target','act','c','a','gamma','soft_s','depth','arch','init_k','lr','seed']


def target(name,x):
    if name=='sin3_quad': return torch.sin(3*x)+0.30*x*x
    if name=='sin_mix': return torch.sin(2*x)+0.35*torch.sin(7*x)
    if name=='abs_wave': return torch.abs(torch.sin(3*x))+0.20*x
    raise ValueError(name)


class CapExp(nn.Module):
    def __init__(self,c,a):
        super().__init__(); self.c=c; self.a=a
    def forward(self,x):
        z=self.a*x; ec=math.exp(self.c)
        return torch.where(z<=self.c, torch.exp(z)-1, ec-1+ec*(z-self.c))


class PreCap(nn.Module):
    def __init__(self,c,a,g,s):
        super().__init__(); self.c=c; self.a=a; self.g=g; self.s=s; self.cap=CapExp(c,1.0)
    def forward(self,x):
        z=self.a*x; u=z+self.g*z*z
        return self.cap(self.s*torch.tanh(u/self.s))


def make_act(name,c,a,g,s):
    if name=='capexp': return CapExp(c,a)
    if name=='precap': return PreCap(c,a,g,s)
    if name=='silu': return nn.SiLU()
    if name=='gelu': return nn.GELU()
    if name=='relu': return nn.ReLU()
    raise ValueError(name)


class Net(nn.Module):
    def __init__(self,name,c,a,g,s,depth,arch,init_k,width=64):
        super().__init__()
        self.arch=arch
        self.depth=depth
        self.layers=nn.ModuleList()
        din=1
        for _ in range(depth):
            self.layers.append(nn.Linear(din,width))
            din=width
        self.out=nn.Linear(width,1)
        self.af=make_act(name,c,a,g,s)
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.normal_(m.weight,0,init_k/math.sqrt(m.in_features))
                nn.init.zeros_(m.bias)

    def forward(self,x,collect=False):
        h=x
        stats=[]
        for l in self.layers:
            pre=l(h)
            y=self.af(pre)
            if self.arch=='plain':
                h=y
            else:
                if h.shape[1] != y.shape[1]:
                    hskip=h.mean(dim=1,keepdim=True).expand_as(y)
                else:
                    hskip=h
                h=hskip + y/math.sqrt(self.depth)
            if collect:
                stats.append((pre.detach(),h.detach()))
        out=self.out(h)
        return (out,stats) if collect else out


def finite(x):
    return math.isfinite(float(x))


def run(cfg):
    torch.manual_seed(cfg['seed'])
    torch.set_num_threads(1)
    x=torch.linspace(-2,2,512).unsqueeze(1)
    y=target(cfg['target'],x)
    xt=torch.linspace(-2.1,2.1,1024).unsqueeze(1)
    yt=target(cfg['target'],xt)
    m=Net(cfg['act'],cfg['c'],cfg['a'],cfg['gamma'],cfg['soft_s'],cfg['depth'],cfg['arch'],cfg['init_k'])
    opt=torch.optim.Adam(m.parameters(),lr=cfg['lr'])
    curve=[]; bad=False; fail=-1
    for step in range(1,3001):
        opt.zero_grad()
        pred=m(x)
        loss=((pred-y)**2).mean()
        if not finite(loss):
            bad=True; fail=step; break
        loss.backward()
        gn=math.sqrt(sum(float(p.grad.detach().square().sum()) for p in m.parameters() if p.grad is not None))
        if not finite(gn):
            bad=True; fail=step; break
        opt.step()
        if step in CHECKPOINTS:
            with torch.no_grad():
                _,ss=m(x,collect=True)
                tl=((m(xt)-yt)**2).mean()
                hstd=sum(float(h.std()) for _,h in ss)/len(ss)
                hmax=max(float(h.abs().max()) for _,h in ss)
                cap=0.0
                if cfg['act'] in ('capexp','precap'):
                    cap=sum(float((z>cfg['c']).float().mean()) for z,_ in ss)/len(ss)
            curve.append({'step':step,'train_loss':float(loss),'test_loss':float(tl),'grad_norm':gn,'act_std_mean':hstd,'act_abs_max':hmax,'cap_frac':cap,'bad':False})
    final = math.inf
    if not bad:
        with torch.no_grad():
            final=float(((m(xt)-yt)**2).mean())
    return curve,final,bad,fail


def configs():
    candidates=[
        ('silu',0.0,1.0,0.0,1.0,0.5,0.001),
        ('gelu',0.0,1.0,0.0,1.0,0.5,0.001),
        ('relu',0.0,1.0,0.0,1.0,0.5,0.0003),
        ('capexp',1.0,1.0,0.0,1.0,0.5,0.0003),
        ('capexp',1.5,1.0,0.0,1.0,0.5,0.0003),
        ('precap',0.5,1.5,0.0,1.5,0.5,0.003),
        ('precap',1.0,1.0,0.01,1.5,0.5,0.001),
    ]
    for target_name in ['sin3_quad','sin_mix','abs_wave']:
        for depth in [32,64]:
            for arch in ['plain','res_inv']:
                for act,c,a,g,s,init_k,lr in candidates:
                    for seed in range(10):
                        yield dict(target=target_name,act=act,c=c,a=a,gamma=g,soft_s=s,depth=depth,arch=arch,init_k=init_k,lr=lr,seed=seed)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--shard',type=int,default=0)
    p.add_argument('--nshards',type=int,default=1)
    p.add_argument('--out',default='v9_results.csv')
    args=p.parse_args()

    todo=list(configs())[args.shard::args.nshards]
    out=args.out.replace('.csv',f'_shard{args.shard}.csv')
    curveout=out.replace('.csv','_curves.csv')
    done=set()
    if os.path.exists(out):
        for r in csv.DictReader(open(out)):
            done.add(tuple(str(r[k]) for k in KEYS))

    total=len(todo)
    for i,cfg in enumerate(todo,1):
        key=tuple(str(cfg[k]) for k in KEYS)
        if key in done:
            continue
        t0=time.time()
        curve,final,bad,fail=run(cfg)
        row=cfg | {'timestamp':datetime.now().isoformat(timespec='seconds'),'test_loss':final,'bad':bad,'fail_step':fail,'duration_sec':time.time()-t0}
        nf=not os.path.exists(out)
        with open(out,'a',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(row.keys()))
            if nf: w.writeheader()
            w.writerow(row)
        if curve:
            cf=not os.path.exists(curveout)
            with open(curveout,'a',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list((cfg | curve[0]).keys()))
                if cf: w.writeheader()
                for pt in curve:
                    w.writerow(cfg | pt)
        print(f'[{i}/{total}] {cfg["target"]} {cfg["act"]} d={cfg["depth"]} {cfg["arch"]} seed={cfg["seed"]} test={final:.5g} bad={bad}', flush=True)


if __name__=='__main__':
    main()
