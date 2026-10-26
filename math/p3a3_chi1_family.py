# Phase 3a-3 Entwurf: SPreCap-Familie (b) mit exaktem Fixpunkt (Mittel 0, Varianz 1) und chi1 = E[f'(Z)^2]
# Integration mit scipy.quad, Knick bei 0 getrennt (Gauss-Hermite ist am Knick ungenau)
import numpy as np
from scipy.integrate import quad
from scipy.optimize import fsolve,brentq
C=1.0;E=np.e;phi=lambda x:np.exp(-x*x/2)/np.sqrt(2*np.pi)
def capexp(w):return np.exp(w)-1 if w<=C else E-1+E*(w-C)
def dcap(w):return np.exp(w) if w<=C else E
def pc(x,s):return capexp(s*np.tanh(x/s))
def dpc(x,s):
    t=x/s
    return 0.0 if abs(t)>350 else dcap(s*np.tanh(t))/np.cosh(t)**2
def I(h):return quad(lambda x:h(x)*phi(x),-np.inf,0,limit=200)[0]+quad(lambda x:h(x)*phi(x),0,np.inf,limit=200)[0]
def moments(b,s):  # Teilintegrale fuer x<0 und x>0
    neg=lambda k:quad(lambda x:pc(b*x,s)**k*phi(x),-np.inf,0,limit=200)[0]
    pos=lambda k:quad(lambda x:pc(b*x,s)**k*phi(x),0,np.inf,limit=200)[0]
    return neg(1),pos(1),neg(2),pos(2)
def solve(b,s):
    n1,p1,n2,p2=moments(b,s)        # mean: l*(a*n1+p1)=0 -> a=-p1/n1 ; var: l^2*(a^2 n2+p2)=1
    a=-p1/n1;l=1/np.sqrt(a*a*n2+p2);return l,a
def chi1(l,a,b,s):
    return l*l*b*b*(a*a*quad(lambda x:dpc(b*x,s)**2*phi(x),-np.inf,0,limit=200)[0]+quad(lambda x:dpc(b*x,s)**2*phi(x),0,np.inf,limit=200)[0])
def sup(l,b,s):return l*(E-1+E*(s-C)) if s>C else l*(np.exp(s)-1)
def inf_(l,a,s):return -l*a*(1-np.exp(-s))
if __name__=='__main__':
    l,a=solve(1.0,1.5);print('Kontrolle SPreCap b=1 s=1.5: lam=%.11f alpha=%.11f chi1=%.4f'%(l,a,chi1(l,a,1,1.5)))
    l,a=solve(0.4118896262,1.5);print('Kontrolle SPreCap-e03: lam=%.9f alpha=%.9f chi1=%.4f'%(l,a,chi1(l,a,0.4118896262,1.5)))
    la,al=1.0507009873554805,1.6732632423543772
    cs=la*la*(al*al*quad(lambda x:np.exp(2*x)*phi(x),-np.inf,0)[0]+0.5);print('SELU chi1 exakt=%.4f'%cs)
    print('--- b-Sweep (s=1.5) ---')
    for b in [1.2,1.0,0.8,0.7,0.6,0.5,0.4118896262,0.3]:
        l,a=solve(b,1.5);print(f"b={b:.4f} lam={l:.6f} alpha={a:.6f} chi1={chi1(l,a,b,1.5):.4f} Schranken [{inf_(l,a,1.5):.3f}, {sup(l,b,1.5):.3f}]")
    for target in [cs,1.04]:
        bt=brentq(lambda b:chi1(*solve(b,1.5),b,1.5)-target,0.3,1.0,xtol=1e-10);l,a=solve(bt,1.5)
        print(f"chi1={target:.4f}: b={bt:.8f} lam={l:.9f} alpha={a:.9f} Schranken [{inf_(l,a,1.5):.3f}, {sup(l,bt,1.5):.3f}]")
