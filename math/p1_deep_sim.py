# Signal- und Gradientenfluss in tiefen MLPs ohne Normalisierung (Breite 256, Tiefe 100, W~N(0,1/n), kein Bias)
import numpy as np
L0,A0=0.80676869644,2.39695130922     # SPreCap-B, gamma=0
BA,LA=0.305560102648,1.0642096166      # SPreCap-A, gamma=0
ls,als=1.0507009873554805,1.6732632423543772
def cap(w,c=1.):return np.where(w<=c,np.exp(np.minimum(w,c))-1,np.e-1+np.e*(w-c))
def dcap(w,c=1.):return np.where(w<=c,np.exp(np.minimum(w,c)),np.e)
def P(x,g=0.,s=1.5):u=x+g*x*x;return cap(s*np.tanh(u/s))
def dP(x,g=0.,s=1.5):u=x+g*x*x;t=np.tanh(u/s);return dcap(s*t)*(1-t*t)*(1+2*g*x)
sig=lambda x:1/(1+np.exp(-x))
A={
 "SELU":(lambda x:ls*np.where(x>0,x,als*(np.exp(np.minimum(x,0))-1)),lambda x:ls*np.where(x>0,1,als*np.exp(np.minimum(x,0)))),
 "SPreCap":(lambda x:L0*np.where(x>0,P(x),A0*P(x)),lambda x:L0*np.where(x>0,dP(x),A0*dP(x))),
 "SPreCap_shift":(lambda x:LA*(P(x)-BA),lambda x:LA*dP(x)),
 "PreCap_ref":(lambda x:P(x,0.01),lambda x:dP(x,0.01)),
 "SiLU":(lambda x:x*sig(x),lambda x:sig(x)*(1+x*(1-sig(x)))),
 "2SiLU":(lambda x:2*x*sig(x),lambda x:2*sig(x)*(1+x*(1-sig(x)))),
 "ELU":(lambda x:np.where(x>0,x,np.exp(np.minimum(x,0))-1),lambda x:np.where(x>0,1,np.exp(np.minimum(x,0)))),
}
n,D,B=256,100,512
print(f"{'Akt':14s} {'start_var':>9s} | mean@10 var@10 | mean@50 var@50 | mean@100 var@100 | log10 |g0|/|gL| (Median 10 Seeds)")
for start in [1.0,4.0]:
  for nm,(f,df) in A.items():
    st=[];gr=[]
    for seed in range(10):
        r=np.random.default_rng(seed);h=r.standard_normal((B,n))*np.sqrt(start);Ws=[];zs=[];rec={}
        for l in range(1,D+1):
            W=r.standard_normal((n,n))/np.sqrt(n);z=h@W;h=f(z);Ws.append(W);zs.append(z)
            if l in(10,50,100):rec[l]=(h.mean(),h.var())
        g=r.standard_normal((B,n));g0=np.linalg.norm(g)
        for W,z in zip(reversed(Ws),reversed(zs)): g=(g*df(z))@W.T
        st.append([rec[10][0],rec[10][1],rec[50][0],rec[50][1],rec[100][0],rec[100][1]]);gr.append(np.log10(np.linalg.norm(g)/g0+1e-300))
    m=np.median(st,0)
    print(f"{nm:14s} {start:9.1f} | {m[0]:+.3f} {m[1]:7.3f} | {m[2]:+.3f} {m[3]:7.3f} | {m[4]:+.3f} {m[5]:8.3f} | {np.median(gr):+.2f}")
