# Phase 3a: tiefe MLPs ohne Normalisierung und ohne Residualverbindung auf MNIST und FashionMNIST
# Frage: Bringt die Beschraenkung (PreCap0) bzw. die Selbstnormierung (SPreCap) in echten Daten etwas gegen Standardaktivierungen?
import argparse,csv,gzip,json,math,os,random,struct,time
from datetime import datetime
from pathlib import Path
import numpy as np,torch,torch.nn as nn,torch.nn.functional as F
torch.set_num_threads(1)
ROOT=Path(__file__).resolve().parent;OUTDIR=ROOT;DATA=Path('/Users/aaronschmitt/PyCharmMiscProject/data')
DATASETS=['mnist','fashion'];DEPTHS=[20,50,100];WIDTH=256;BATCH=128;STEPS=860;CHECKPOINTS=[0,215,430,860]
LRS=[1e-4,3e-4,1e-3,3e-3,1e-2,3e-2]
ACTS=['relu','gelu','silu','elu','selu','tanh','precap0','spreccap','spreccap_e03']
TUNE_SEEDS=list(range(40000,40005));CONF_SEEDS=list(range(40010,40025));SEEDS=TUNE_SEEDS+CONF_SEEDS
# Zweistufig: 'tune' = alle LRs x TUNE_SEEDS; danach --select (LR je dataset/depth/act); 'confirm' = gewaehlte LR x CONF_SEEDS
NVAL=5000
# --- Aktivierungen -----------------------------------------------------------
S_,C_=1.5,1.0;EC=math.exp(C_)
def capexp(w):return torch.where(w<=C_,torch.exp(torch.clamp(w,max=C_))-1,EC-1+EC*(w-C_))
def precap0(x):return capexp(S_*torch.tanh(x/S_))           # gamma=0 (Phase 1)
SP={'spreccap':(0.80676869644,2.39695130922,1.0),'spreccap_e03':(1.985552225,1.608136395,0.4118896262)}  # (lambda,alpha,b)
class Act(nn.Module):
    def __init__(s,n):super().__init__();s.n=n
    def forward(s,x):
        n=s.n
        if n=='relu':return F.relu(x)
        if n=='gelu':return F.gelu(x)
        if n=='silu':return F.silu(x)
        if n=='elu':return F.elu(x)
        if n=='selu':return F.selu(x)
        if n=='tanh':return torch.tanh(x)
        if n=='precap0':return precap0(x)
        l,a,b=SP[n];p=precap0(b*x);return l*torch.where(x>0,p,a*p)
        raise ValueError(n)
def gain(n):
    # varianzerhaltende Init fuer jede Aktivierung: W~N(0,g^2/fan_in), g=1/sqrt(E[f(Z)^2]), Z~N(0,1)
    # (ReLU -> sqrt(2)=He, SELU/SPreCap -> ca. 1=LeCun); Gauss-Hermite, 150 Knoten (400 laeuft in numpy ueber)
    X,Wt=np.polynomial.hermite_e.hermegauss(150);Wt=Wt/Wt.sum()
    with torch.no_grad():y=Act(n)(torch.tensor(X,dtype=torch.float64)).numpy()
    return float(1/np.sqrt(np.sum(Wt*y*y)))
GAINS={n:gain(n) for n in ACTS}
# --- Daten -------------------------------------------------------------------
def idx(p):
    with gzip.open(p,'rb') as f:
        mg=struct.unpack('>I',f.read(4))[0];nd=mg&0xff;sh=struct.unpack('>'+'I'*nd,f.read(4*nd));return np.frombuffer(f.read(),dtype=np.uint8).reshape(sh)
_CACHE={}
def load(ds):
    if ds in _CACHE:return _CACHE[ds]
    d=DATA/ds;xtr=idx(d/'train-images-idx3-ubyte.gz').reshape(-1,784).astype(np.float32)/255;ytr=idx(d/'train-labels-idx1-ubyte.gz').astype(np.int64)
    xte=idx(d/'t10k-images-idx3-ubyte.gz').reshape(-1,784).astype(np.float32)/255;yte=idx(d/'t10k-labels-idx1-ubyte.gz').astype(np.int64)
    xv,yv=xtr[-NVAL:],ytr[-NVAL:];xtr,ytr=xtr[:-NVAL],ytr[:-NVAL]           # feste Validierungsaufteilung: letzte 5000
    m,s=xtr.mean(),xtr.std();T=lambda a:torch.tensor((a-m)/s)
    _CACHE[ds]=(T(xtr),torch.tensor(ytr),T(xv),torch.tensor(yv),T(xte),torch.tensor(yte));return _CACHE[ds]
# --- Netz --------------------------------------------------------------------
class Net(nn.Module):
    def __init__(s,depth,n):
        super().__init__();dims=[784]+[WIDTH]*depth;s.hid=nn.ModuleList(nn.Linear(a,b) for a,b in zip(dims[:-1],dims[1:]));s.out=nn.Linear(WIDTH,10);s.af=Act(n);g=GAINS[n]
        for l in s.hid:nn.init.normal_(l.weight,0,g/math.sqrt(l.in_features));nn.init.zeros_(l.bias)
        nn.init.normal_(s.out.weight,0,1/math.sqrt(WIDTH));nn.init.zeros_(s.out.bias)
    def forward(s,x,collect=False):
        h=x;st=[]
        for l in s.hid:
            h=s.af(l(h))
            if collect:st.append(h)
        return (s.out(h),st) if collect else s.out(h)
# --- Ablauf ------------------------------------------------------------------
MAIN=['run_id','dataset','depth','act','lr','seed','step','train_loss','val_loss','val_acc','test_loss','test_acc','grad_norm','h_var_first','h_var_mid','h_var_last','h_mean_last','h_absmax_last','gain','run_status','sec','timestamp']
SELFILE=lambda:OUTDIR/'meta'/'lr_selection.json'
def configs(phase='all'):
    sel=None
    if phase in('confirm','all'):
        sel=json.load(open(SELFILE())) if SELFILE().exists() else None
        if phase=='confirm' and sel is None:raise SystemExit('lr_selection.json fehlt')
    for depth in DEPTHS:            # Tiefe aussen: 20 und 50 werden zuerst fertig
        for ds in DATASETS:
          for act in ACTS:
            if phase in('tune','all'):
                for lr in LRS:
                    for seed in TUNE_SEEDS:yield dict(dataset=ds,depth=depth,act=act,lr=lr,seed=seed)
            if phase in('confirm','all') and sel is not None:
                for seed in CONF_SEEDS:yield dict(dataset=ds,depth=depth,act=act,lr=sel[f'{ds}|{depth}|{act}'],seed=seed)
def select():
    # praeregistriert: hoechste mediane val_acc bei Schritt STEPS ueber TUNE_SEEDS; divergiert/nicht-endlich -> 0.1; Gleichstand -> kleinere LR
    it=integrity('tune')
    if it['missing'] or it['duplicates'] or it['unexpected']:raise SystemExit('Tuning unvollstaendig: '+json.dumps(it))
    acc={}
    for f in (OUTDIR/'workers').glob('w*.csv'):
        for r in csv.DictReader(open(f)):
            if int(r['seed']) in TUNE_SEEDS and r['step']==str(STEPS) and r['run_status']=='ok':acc[r['run_id']]=float(r['val_acc'])
    sel={}
    for depth in DEPTHS:
        for ds in DATASETS:
            for act in ACTS:
                best=None
                for lr in LRS:
                    v=[]
                    for s in TUNE_SEEDS:
                        k=rid(dict(dataset=ds,depth=depth,act=act,lr=lr,seed=s));v.append(acc.get(k,0.1))
                    m=float(np.median(v))
                    if best is None or m>best[0]:best=(m,lr)
                sel[f'{ds}|{depth}|{act}']=best[1]
    SELFILE().parent.mkdir(exist_ok=True);json.dump(sel,open(SELFILE(),'w'),indent=1);return sel
def rid(c):return f"{c['dataset']}_d{c['depth']}_{c['act']}_lr{c['lr']:g}_s{c['seed']}"
@torch.no_grad()
def evaluate(m,x,y):
    m.eval();lo=0.;ac=0
    for i in range(0,len(x),2500):
        o=m(x[i:i+2500]);lo+=float(F.cross_entropy(o,y[i:i+2500],reduction='sum'));ac+=int((o.argmax(1)==y[i:i+2500]).sum())
    return lo/len(x),ac/len(x)
@torch.no_grad()
def hstats(m,x):
    m.eval();_,st=m(x[:1000],collect=True);v=[float(h.var()) for h in st];L=st[-1]
    return v[0],v[len(v)//2],v[-1],float(L.mean()),float(L.abs().max())
def row(c,step,trl,m,data,gn,status,t0):
    xtr,ytr,xv,yv,xte,yte=data
    if status=='ok':
        vl,va=evaluate(m,xv,yv);tl,ta=evaluate(m,xte,yte);h=hstats(m,xv)
        if not all(math.isfinite(v) for v in (vl,tl)+h):status='nonfinite_eval'
    else:vl=va=tl=ta=float('nan');h=(float('nan'),)*5
    return dict(run_id=rid(c),**c,step=step,train_loss=trl,val_loss=vl,val_acc=va,test_loss=tl,test_acc=ta,grad_norm=gn,h_var_first=h[0],h_var_mid=h[1],h_var_last=h[2],h_mean_last=h[3],h_absmax_last=h[4],gain=GAINS[c['act']],run_status=status,sec=round(time.time()-t0,2),timestamp=datetime.now().isoformat(timespec='seconds'))
def execute(c,wid):
    t0=time.time();data=load(c['dataset']);xtr,ytr=data[0],data[1]
    random.seed(c['seed']);np.random.seed(c['seed']);torch.manual_seed(c['seed'])
    m=Net(c['depth'],c['act']);o=torch.optim.Adam(m.parameters(),lr=c['lr']);g=torch.Generator().manual_seed(c['seed']+777)
    buf=[row(c,0,float('nan'),m,data,float('nan'),'ok',t0)];perm=torch.randperm(len(xtr),generator=g);pos=0;run=[];gn=float('nan')
    for step in range(1,STEPS+1):
        if pos+BATCH>len(xtr):perm=torch.randperm(len(xtr),generator=g);pos=0
        b=perm[pos:pos+BATCH];pos+=BATCH
        m.train();o.zero_grad(set_to_none=True);loss=F.cross_entropy(m(xtr[b]),ytr[b])
        if not torch.isfinite(loss):buf.append(row(c,step,float('nan'),m,data,float('nan'),'nonfinite_loss',t0));break
        loss.backward();gn=float(torch.sqrt(sum(p.grad.square().sum() for p in m.parameters())));o.step();run.append(float(loss.detach()))
        if not math.isfinite(gn):buf.append(row(c,step,float('nan'),m,data,gn,'nonfinite_grad',t0));break
        if step in CHECKPOINTS:
            r=row(c,step,float(np.mean(run[-50:])),m,data,gn,'ok',t0);buf.append(r)
            if r['run_status']!='ok':break
    p=OUTDIR/'workers'/f'w{wid}.csv';p.parent.mkdir(parents=True,exist_ok=True);new=not p.exists()
    with p.open('a',newline='') as f:
        w=csv.DictWriter(f,fieldnames=MAIN)
        if new:w.writeheader()
        for r in buf:w.writerow(r)
def finished():
    done=set()
    for f in (OUTDIR/'workers').glob('w*.csv'):
        for r in csv.DictReader(open(f)):
            if r['step']==str(STEPS) or r['run_status']!='ok':done.add(r['run_id'])
    return done
def selfcheck():
    out={'gains':GAINS}
    X,Wt=np.polynomial.hermite_e.hermegauss(150);Wt=Wt/Wt.sum();z=torch.tensor(X,dtype=torch.float64)
    for n in ['selu','spreccap','spreccap_e03']:
        y=Act(n)(z).numpy();out[n+'_E_Var']=[float(np.sum(Wt*y)),float(np.sum(Wt*y*y)-np.sum(Wt*y)**2)]
    t=torch.linspace(-6,6,10001,dtype=torch.float64);p=precap0(t).numpy();ref=np.where(1.5*np.tanh(t.numpy()/1.5)<=1,np.exp(1.5*np.tanh(t.numpy()/1.5))-1,np.e-1+np.e*(1.5*np.tanh(t.numpy()/1.5)-1))
    out['precap0_max_abs_err']=float(np.abs(p-ref).max());out['precap0_monotone']=bool((np.diff(p)>=0).all());out['precap0_sup']=float(precap0(torch.tensor([1e4],dtype=torch.float64)))
    for ds in DATASETS:
        a=load(ds);out[ds]={'ntrain':len(a[0]),'nval':len(a[2]),'ntest':len(a[4]),'xmean':float(a[0].mean()),'xstd':float(a[0].std()),'classes':int(a[1].max())+1,'val_test_disjoint_from_train':True}
    out['configs_tune']=len(list(configs('tune')));out['configs']=out['configs_tune']+len(DATASETS)*len(DEPTHS)*len(ACTS)*len(CONF_SEEDS);out['expected_rows_max']=out['configs']*len(CHECKPOINTS)
    return out
def integrity(phase='all'):
    rows=[]
    for f in sorted((OUTDIR/'workers').glob('w*.csv')):rows+=list(csv.DictReader(open(f)))
    cf={rid(c) for c in configs(phase)};keys=[(r['run_id'],r['step']) for r in rows];ended={r['run_id'] for r in rows if r['step']==str(STEPS) or r['run_status']!='ok'}
    if phase=='tune':ended={e for e in ended if int(e.rsplit('_s',1)[1]) in TUNE_SEEDS}
    return dict(phase=phase,rows=len(rows),unique_keys=len(set(keys)),duplicates=len(keys)-len(set(keys)),ended_runs=len(ended),expected_runs=len(cf),missing=len(cf-ended),unexpected=len(ended-cf),nonfinite_runs=len({r['run_id'] for r in rows if r['run_status']!='ok'}))
def main():
    global STEPS,CHECKPOINTS,OUTDIR,TUNE_SEEDS,CONF_SEEDS,SEEDS,LRS,DEPTHS
    p=argparse.ArgumentParser();p.add_argument('--worker',type=int);p.add_argument('--workers',type=int,default=1);p.add_argument('--selfcheck',action='store_true');p.add_argument('--integrity',action='store_true');p.add_argument('--smoke',action='store_true');p.add_argument('--probe',action='store_true');p.add_argument('--phase',default='tune',choices=['tune','confirm']);p.add_argument('--select',action='store_true');a=p.parse_args()
    if a.smoke:STEPS=20;CHECKPOINTS=[0,10,20];OUTDIR=ROOT/'smoke';TUNE_SEEDS=[40000,40001];CONF_SEEDS=[40010];SEEDS=TUNE_SEEDS+CONF_SEEDS;LRS=[1e-3,1e-2];DEPTHS=[20,100]
    if a.selfcheck:print(json.dumps(selfcheck(),indent=2));return
    if a.integrity:print(json.dumps(integrity(a.phase if a.phase=='tune' else 'all'),indent=2));return
    if a.select:print(json.dumps(select(),indent=1));return
    if a.probe:
        # Zeitmessung: 50 Schritte je Tiefe und Aktivierung
        data=load('mnist');res={}
        for d in DEPTHS:
            for n in ['relu','precap0','spreccap']:
                m=Net(d,n);o=torch.optim.Adam(m.parameters(),1e-3);t=time.time()
                for i in range(50):
                    b=torch.randint(0,len(data[0]),(BATCH,));o.zero_grad();F.cross_entropy(m(data[0][b]),data[1][b]).backward();o.step()
                res[f'd{d}_{n}_ms_per_step']=round((time.time()-t)/50*1000,2)
        print(json.dumps(res));return
    skip=finished()
    for i,c in enumerate(configs(a.phase)):
        if i%a.workers!=a.worker:continue
        if rid(c) in skip:continue
        execute(c,a.worker);print('done',rid(c),datetime.now().strftime('%H:%M:%S'),flush=True)
if __name__=='__main__':main()
