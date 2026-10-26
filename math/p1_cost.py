
import torch,time,math,torch.nn.functional as F
torch.set_num_threads(8)
def capexp(z,c=1.):ec=math.exp(c);return torch.where(z<=c,torch.exp(torch.clamp(z,max=c))-1,ec-1+ec*(z-c))
def precap(x,g=0.01,s=1.5):u=x+g*x*x;return capexp(s*torch.tanh(u/s))
def precap0(x,s=1.5):return capexp(s*torch.tanh(x/s))
L0,A0=0.80676869644,2.39695130922
def sprecap(x):p=precap0(x);return L0*torch.where(x>0,p,A0*p)
acts={"ReLU":F.relu,"SiLU":F.silu,"GELU":F.gelu,"ELU":F.elu,"SELU":F.selu,"CapExp":capexp,"PreCap":precap,"PreCap_g0":precap0,"SPreCap":sprecap}
res={}
for dev in ["cpu","mps"]:
  if dev=="mps" and not torch.backends.mps.is_available():continue
  x0=torch.randn(4096,1024,device=dev)
  for n,f in acts.items():
    ts=[]
    for rep in range(25):
      x=x0.clone().requires_grad_(True)
      if dev=="mps":torch.mps.synchronize()
      t=time.perf_counter();y=f(x);y.sum().backward()
      if dev=="mps":torch.mps.synchronize()
      ts.append(time.perf_counter()-t)
    ts=sorted(ts[5:]);res[(dev,n)]=ts[len(ts)//2]
  b=res[(dev,"ReLU")]
  print(dev,"  ".join(f"{n}={res[(dev,n)]*1e3:.2f}ms({res[(dev,n)]/b:.1f}x)" for n in acts))
print(torch.__version__)
