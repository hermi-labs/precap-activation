# 100 Schichten, Breite 256, wie p1_deep_sim.py; Vergleich SPreCap (b=1) mit SPreCap-eps (b nach Chen et al.)
import numpy as np
def cap(w,c=1.):return np.where(w<=c,np.exp(np.minimum(w,c))-1,np.e-1+np.e*(w-c))
def dcap(w,c=1.):return np.where(w<=c,np.exp(np.minimum(w,c)),np.e)
def P(x,s=1.5):return cap(s*np.tanh(x/s))
def dP(x,s=1.5):t=np.tanh(x/s);return dcap(s*t)*(1-t*t)
def mk(l,a,b):return (lambda x:l*np.where(x>0,P(b*x),a*P(b*x)),lambda x:l*b*np.where(x>0,dP(b*x),a*dP(b*x)))
ls,als=1.0507009873554805,1.6732632423543772
A={"SELU":(lambda x:ls*np.where(x>0,x,als*(np.exp(np.minimum(x,0))-1)),lambda x:ls*np.where(x>0,1,als*np.exp(np.minimum(x,0)))),
   "SPreCap (b=1)":mk(0.80676869644,2.39695130922,1.0),
   "SPreCap-eps0.03":mk(1.985552225,1.608136395,0.4118896262),
   "SPreCap-eps0.01":mk(4.169105852,1.296919252,0.2124516684)}
n,D,B=256,100,512
for nm,(f,df) in A.items():
    st=[];gr=[];lin=[]
    for seed in range(10):
        r=np.random.default_rng(seed);h=r.standard_normal((B,n));Ws=[];zs=[]
        for l in range(D):
            W=r.standard_normal((n,n))/np.sqrt(n);z=h@W;h=f(z);Ws.append(W);zs.append(z)
        st.append([h.mean(),h.var()])
        g=r.standard_normal((B,n));g0=np.linalg.norm(g)
        for W,z in zip(reversed(Ws),reversed(zs)): g=(g*df(z))@W.T
        gr.append(np.log10(np.linalg.norm(g)/g0))
    # Nichtlinearitaet: Anteil der Varianz von f(Z), der NICHT durch lineare Regression auf Z erklaert wird
    z=np.random.default_rng(0).standard_normal(1_000_000);y=f(z);beta=np.cov(z,y)[0,1];nl=1-beta**2/y.var()
    m=np.median(st,0);print(f"{nm:18s} mean@100={m[0]:+.3f} var@100={m[1]:.3f} log10 Gradientenverstaerkung={np.median(gr):+.2f} nichtlinearer Varianzanteil={nl:.4f}")
