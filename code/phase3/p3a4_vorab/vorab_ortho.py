# Explorativer Vorabtest 3a-4 (nicht praeregistriert): orthogonale Init, Gain var vs krit, Tiefe 100, MNIST, 600 Schritte
import sys,math,json,torch,torch.nn as nn,torch.nn.functional as F,numpy as np
sys.argv=['x'];sys.path.insert(0,'/tmp/p3a4v');import run_p3a3 as R
torch.set_num_threads(2)
GK={'precap0':0.9524,'spreccap_cs':0.7986,'selu':0.7002,'elu':1.0134}
GV={'precap0':1.0120,'spreccap_cs':1.0,'selu':1.0,'elu':1.2452}
def net(act,g,depth=100):
    torch.manual_seed(42000);m=R.Net.__new__(R.Net);nn.Module.__init__(m)
    dims=[784]+[256]*depth;m.hid=nn.ModuleList(nn.Linear(a,b) for a,b in zip(dims[:-1],dims[1:]));m.out=nn.Linear(256,10);m.afs=nn.ModuleList(R.Act(act) for _ in range(depth))
    for l in m.hid:nn.init.orthogonal_(l.weight,gain=g);nn.init.zeros_(l.bias)
    nn.init.normal_(m.out.weight,0,1/16);nn.init.zeros_(m.out.bias);return m
xtr,ytr,xv,yv,_,_=R.load('mnist')
res=[]
for act in GK:
  for lab,G in (('var',GV),('krit',GK)):
    for lr in (1e-4,5e-4):
      m=net(act,G[act]);opt=torch.optim.Adam(m.parameters(),lr=lr);gen=torch.Generator().manual_seed(1)
      # Init-Gradprofil
      out,st=m(xtr[:512],collect=True);F.cross_entropy(out,ytr[:512]).backward()
      g1=m.hid[0].weight.grad.norm().item();gL=m.hid[-1].weight.grad.norm().item();hv=st[-1].var().item();opt.zero_grad()
      for step in range(600):
        i=torch.randint(0,len(xtr),(128,),generator=gen);loss=F.cross_entropy(m(xtr[i]),ytr[i]);opt.zero_grad();loss.backward();opt.step()
      with torch.no_grad():acc=(m(xv).argmax(1)==yv).float().mean().item()
      r=dict(act=act,init=lab,lr=lr,log10_g1_gL=round(math.log10(g1/gL),3),hvar_last=float('%.3g'%hv),val_acc=round(acc,4));res.append(r);print(r,flush=True)
json.dump(res,open('code/phase3/p3a4_vorab/vorab_ortho.json','w'),indent=1)
