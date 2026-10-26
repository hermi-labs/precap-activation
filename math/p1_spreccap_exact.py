# SPreCap-Konstanten mit adaptiver Integration (aufgeteilt an Knickstellen) + Gradientenfluss chi
import numpy as np, mpmath as mp
mp.mp.dps=30
S,C=mp.mpf('1.5'),mp.mpf(1)
def capexp(w,c=C): return mp.e**w-1 if w<=c else mp.e**c-1+mp.e**c*(w-c)
def dcap(w,c=C): return mp.e**w if w<=c else mp.e**c
def P(x,g): u=x+g*x*x; return capexp(S*mp.tanh(u/S))
def dP(x,g): u=x+g*x*x; return dcap(S*mp.tanh(u/S))*mp.sech(u/S)**2*(1+2*g*x)
def xc(g):  # Uebergang v=c
    uc=S*mp.atanh(C/S); return uc if g==0 else (-1+mp.sqrt(1+4*g*uc))/(2*g)
phi=lambda x:mp.e**(-x*x/2)/mp.sqrt(2*mp.pi)
def E(h,pts): return mp.quad(lambda t:h(t)*phi(t),pts)
def solveB(g):
    pts=[-mp.inf,0,xc(g),mp.inf]
    Ip=E(lambda t:P(t,g) if t>0 else 0,pts); In=E(lambda t:P(t,g) if t<=0 else 0,pts)
    Ip2=E(lambda t:P(t,g)**2 if t>0 else 0,pts); In2=E(lambda t:P(t,g)**2 if t<=0 else 0,pts)
    al=-Ip/In; lam=1/mp.sqrt(Ip2+al**2*In2)   # Mittel 0 => alpha; Var 1 => lambda
    chi=lam**2*(E(lambda t:dP(t,g)**2 if t>0 else 0,pts)+al**2*E(lambda t:dP(t,g)**2 if t<=0 else 0,pts))
    return lam,al,chi
def solveA(g):
    pts=[-mp.inf,0,xc(g),mp.inf]
    m=E(lambda t:P(t,g),pts); v=E(lambda t:P(t,g)**2,pts)-m*m; lam=1/mp.sqrt(v)
    chi=lam**2*E(lambda t:dP(t,g)**2,pts); return m,lam,chi
# SELU-Kontrolle: publizierte Konstanten muessen E=0, Var=1 liefern
l0,a0=mp.mpf('1.0507009873554804934193349852946'),mp.mpf('1.6732632423543772848170429916717')
selu=lambda t:l0*t if t>0 else l0*a0*(mp.e**t-1); dselu=lambda t:l0 if t>0 else l0*a0*mp.e**t
m=E(selu,[-mp.inf,0,mp.inf]); v=E(lambda t:selu(t)**2,[-mp.inf,0,mp.inf])-m*m
print("SELU-Kontrolle: E=",mp.nstr(m,5),"Var=",mp.nstr(v,12),"chi=E[f'^2]=",mp.nstr(E(lambda t:dselu(t)**2,[-mp.inf,0,mp.inf]),8))
for g in [mp.mpf('0.01'),mp.mpf(0)]:
    lam,al,chi=solveB(g); print(f"SPreCap-B gamma={g}: lambda={mp.nstr(lam,12)} alpha={mp.nstr(al,12)} chi={mp.nstr(chi,8)}")
    m,lamA,chiA=solveA(g); print(f"SPreCap-A gamma={g}: beta={mp.nstr(m,12)} lambda={mp.nstr(lamA,12)} chi={mp.nstr(chiA,8)}")
for nm,f,df in [("SiLU",lambda t:t/(1+mp.e**-t),None),("2SiLU",lambda t:2*t/(1+mp.e**-t),None)]:
    d=lambda t,f=f:mp.diff(f,t); print(nm,"E=",mp.nstr(E(f,[-mp.inf,0,mp.inf]),6),"Var=",mp.nstr(E(lambda t:f(t)**2,[-mp.inf,0,mp.inf])-E(f,[-mp.inf,0,mp.inf])**2,6),"chi=",mp.nstr(E(lambda t:d(t)**2,[-mp.inf,0,mp.inf]),6))
