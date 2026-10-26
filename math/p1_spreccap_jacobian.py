# Stabilitaet des Fixpunkts (mu,nu)=(0,1) wie im SELU-Paper; Gauss-Hermite-Quadratur (Genauigkeit ca. 1e-3 an Knickstellen)
import numpy as np
from numpy.polynomial.hermite_e import hermegauss
X,Wt=hermegauss(201); Wt=Wt/Wt.sum()
def E(h,mu=0.,sd=1.): return float(np.sum(Wt*h(mu+sd*X)))
def capexp(w,c=1.):return np.where(w<=c,np.exp(np.minimum(w,c))-1,np.exp(c)-1+np.exp(c)*(w-c))
def P0(x,s=1.5):return capexp(s*np.tanh(x/s))
L0,A0=0.80676869644,2.39695130922; BA,LA=0.305560102648,1.0642096166
ls,als=1.0507009873554805,1.6732632423543772
F={"SPreCap":lambda t:L0*np.where(t>0,P0(t),A0*P0(t)),
   "SPreCap_shift":lambda t:LA*(P0(t)-BA),
   "SELU":lambda t:ls*np.where(t>0,t,als*(np.exp(np.minimum(t,0))-1))}
def mapping(f,mu,nu,om=0.,tau=1.):
    sd=np.sqrt(nu*tau);m=E(f,mu*om,sd);return m,E(lambda t:f(t)**2,mu*om,sd)-m*m
def jac(f,om=0.,tau=1.,h=1e-5):
    J=np.zeros((2,2));p=np.array([0.,1.])
    for j in range(2):
        e=np.zeros(2);e[j]=h;J[:,j]=(np.array(mapping(f,*(p+e),om,tau))-np.array(mapping(f,*(p-e),om,tau)))/(2*h)
    return J
for nm,f in F.items():
    ev=np.abs(np.linalg.eigvals(jac(f))).max()
    worst=max(np.linalg.norm(jac(f,om,ta),2) for om in np.linspace(-.1,.1,5) for ta in np.linspace(.8,1.25,5))
    print(f"{nm:14s} |eig|max={ev:.4f}  max ||J||_2 (om in [-0.1,0.1], tau in [0.8,1.25]) = {worst:.4f}")
