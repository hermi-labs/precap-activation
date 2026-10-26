# Phase 3a-4 Vorbereitung: kritischer Gain g_c (chi1=1) fuer orthogonale Init, Integration mit quad (Knick bei 0 getrennt)
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
S,C=1.5,1.0;E=np.e
def capexp(w):return np.exp(w)-1 if w<=C else E-1+E*(w-C)
def dcap(w):return np.exp(w) if w<=C else E
def pc(x):return capexp(S*np.tanh(x/S))
def dpc(x):return dcap(S*np.tanh(x/S))/np.cosh(x/S)**2
def mk(l,a,b):
    f=lambda x:l*(pc(b*x) if x>0 else a*pc(b*x));df=lambda x:l*b*(1 if x>0 else a)*dpc(b*x);return f,df
la,al=1.0507009873554805,1.6732632423543772
selu=lambda x:la*(x if x>0 else al*(np.exp(x)-1));dselu=lambda x:la*(1 if x>0 else al*np.exp(x))
elu=lambda x:x if x>0 else np.exp(x)-1;delu=lambda x:1.0 if x>0 else np.exp(x)
phi=lambda z:np.exp(-z*z/2)/np.sqrt(2*np.pi)
def Eg(h,q):
    s=np.sqrt(q);return quad(lambda z:h(s*z)*phi(z),-np.inf,0,limit=200)[0]+quad(lambda z:h(s*z)*phi(z),0,np.inf,limit=200)[0]
def qstar(f,g):
    # Fixpunkt der Varianzabbildung q=g^2 E[f^2] (Bias 0)
    q=1.0
    for _ in range(400):
        qn=g*g*Eg(lambda x:f(x)**2,q)
        if abs(qn-q)<1e-12:break
        q=qn
    return q
def chi1(f,df,g):q=qstar(f,g);return q,g*g*Eg(lambda x:df(x)**2,q)
acts={'precap0':(pc,dpc),'spreccap_cs':mk(1.102702500,2.053776267,0.71188112),'spreccap':mk(0.80676869644,2.39695130922,1.0),'selu':(selu,dselu),'elu':(elu,delu)}
for n,(f,df) in acts.items():
    g0=1/np.sqrt(Eg(lambda x:f(x)**2,1.0))  # Varianz-erhaltender Gain bei q=1
    q0,c0=chi1(f,df,g0)
    row=[(round(g,3),)+tuple(round(v,4) for v in chi1(f,df,g)) for g in (0.7,0.85,0.95,1.0,1.1)]
    try:
        gc=brentq(lambda g:chi1(f,df,g)[1]-1,0.5,g0*1.2,xtol=1e-6);qc,cc=chi1(f,df,gc)
        mu=Eg(f,qc);m2=Eg(lambda x:f(x)**2,qc)
        print(f"{n:12s} g_var={g0:.4f} (q*={q0:.3f}, chi1={c0:.4f}) | g_krit={gc:.4f} q*={qc:.4f} chi1={cc:.5f} mean/rms={mu/np.sqrt(m2):.3f}")
    except Exception as e:print(n,'kein g_krit',e)
    print('   g,q*,chi1:',row)
# Streuung der Ableitung am kritischen Punkt: kappa=E[f'^4]/E[f'^2]^2-1 (0 = perfekte Isometrie-Voraussetzung, Pennington 2017)
import warnings;warnings.filterwarnings('ignore')
print('--- Ableitungs-Streuung am kritischen Punkt ---')
for n,(f,df) in acts.items():
    g0=1/np.sqrt(Eg(lambda x:f(x)**2,1.0));gc=brentq(lambda g:chi1(f,df,g)[1]-1,0.5,g0*1.2,xtol=1e-6);qc=qstar(f,gc)
    for lab,g,q in (('g_krit',gc,qc),('g_var',g0,qstar(f,g0))):
        m2=Eg(lambda x:df(x)**2,q);m4=Eg(lambda x:df(x)**4,q)
        print(f"{n:12s} {lab:6s} g={g:.4f} q*={q:.4f} kappa={m4/m2**2-1:.4f} f'(0-)={df(-1e-9):.3f} f'(0+)={df(1e-9):.3f}")
