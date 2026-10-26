import numpy as np
from numpy.polynomial.hermite_e import hermegauss
X,W=hermegauss(150);W=W/W.sum()
S,C=1.5,1.0;E=np.e
def capexp(w):return np.where(w<=C,np.exp(np.minimum(w,C))-1,E-1+E*(w-C))
def dcap(w):return np.where(w<=C,np.exp(np.minimum(w,C)),E)
def pc(x):return capexp(S*np.tanh(x/S))
def dpc(x):return dcap(S*np.tanh(x/S))/np.cosh(x/S)**2
def sp(x,l=0.80676869644,a=2.39695130922,b=1.0):p=pc(b*x);return l*np.where(x>0,p,a*p)
def dsp(x,l=0.80676869644,a=2.39695130922,b=1.0):return l*b*np.where(x>0,1,a)*dpc(b*x)
la,al=1.0507009873554805,1.6732632423543772
def selu(x):return la*np.where(x>0,x,al*(np.exp(np.minimum(x,0))-1))
def dselu(x):return la*np.where(x>0,1,al*np.exp(np.minimum(x,0)))
def relu(x):return np.maximum(x,0)
def drelu(x):return (x>0)*1.0
def tanh(x):return np.tanh(x)
def dtanh(x):return 1/np.cosh(x)**2
def fixed(f,g,q=1.0):
    for _ in range(3000):q=g*g*np.sum(W*f(np.sqrt(q)*X)**2)
    return q
def chi(f,df,g):
    q=fixed(f,g);z=np.sqrt(q)*X
    return q,g*g*np.sum(W*df(z)**2),np.sum(W*f(z)),np.sum(W*f(z)**2)
gain=lambda f:1/np.sqrt(np.sum(W*f(X)**2))
for n,f,df in [('precap0',pc,dpc),('spreccap',sp,dsp),('selu',selu,dselu),('relu',relu,drelu),('tanh',tanh,dtanh)]:
    g=gain(f);q,c1,m,s2=chi(f,df,g)
    print(f"{n:9s} g={g:.3f} q*={q:.3f} chi1={c1:.4f} chi1^100={c1**100:.2e} mean f={m:.3f} mean/rms={m/np.sqrt(s2):.2f}")
# kritischer Gain fuer precap0 (chi1=1)
from scipy.optimize import brentq
h=lambda g:chi(pc,dpc,g)[1]-1
gs=np.linspace(0.8,3,45);vals=[h(g) for g in gs];print('precap0 chi1-1 ueber g:',[(round(a,2),round(b,3)) for a,b in zip(gs[::4],vals[::4])])
try:
    gc=brentq(h,0.8,3);q,c1,m,s2=chi(pc,dpc,gc);print('g_krit',round(gc,4),'q*',round(q,3),'chi1',round(c1,4),'mean/rms',round(m/np.sqrt(s2),2))
except Exception as e:print('kein g_krit',e)
# zentrierte Variante f-mu
print('--- Korrelation zweier Eingaben ueber 100 Schichten (Start c=0.2) ---')
X2,W2=hermegauss(60);W2=W2/W2.sum();Z1,Z2=np.meshgrid(X2,X2);WW=np.outer(W2,W2)
def cmap(f,g,q,c):
    u1=np.sqrt(q)*Z1;u2=np.sqrt(q)*(c*Z1+np.sqrt(max(1-c*c,0))*Z2)
    return g*g*np.sum(WW*f(u1)*f(u2))/q
for n,f,g in [('precap0',pc,gain(pc)),('precap0_zentriert',lambda x:pc(x)-0.306/1.012*0+0,None),('spreccap',sp,1.0),('selu',selu,1.0),('relu',relu,np.sqrt(2)),('tanh',tanh,gain(tanh))]:
    if g is None:continue
    q=fixed(f,g);c=0.2;tr=[]
    for L in range(100):
        c=cmap(f,g,q,c)
        if L+1 in (1,10,20,50,100):tr.append((L+1,round(c,5)))
    print(f"{n:9s}",tr)
# zentrierte PreCap0: f - E[f] mit varianz-gain
mu=np.sum(W*pc(X));fc=lambda x:pc(x)-mu;gc=1/np.sqrt(np.sum(W*fc(X)**2))
q=fixed(fc,gc);c=0.2;tr=[]
for L in range(100):
    c=cmap(fc,gc,q,c)
    if L+1 in (1,10,20,50,100):tr.append((L+1,round(c,5)))
print('precap0 zentriert g=%.3f q*=%.3f chi1=%.3f'%(gc,q,chi(fc,dpc,gc)[1]),tr)
