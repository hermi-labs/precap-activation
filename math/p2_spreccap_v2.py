# SPreCap v2: drei Konstanten (lambda, alpha, Eingangsskala b), Bedingungen nach Chen et al. (2021/2024):
# E[f]=0, E[f^2]=1, E[f'^2]=1+eps ; f(x)=lambda*PreCap0(b x) fuer x>0, lambda*alpha*PreCap0(b x) fuer x<=0
import mpmath as mp
mp.mp.dps=25
S,C=mp.mpf('1.5'),mp.mpf(1)
cap=lambda w:mp.e**w-1 if w<=C else mp.e**C-1+mp.e**C*(w-C)
dcap=lambda w:mp.e**w if w<=C else mp.e**C
P=lambda x:cap(S*mp.tanh(x/S)); dP=lambda x:dcap(S*mp.tanh(x/S))*(1-mp.tanh(x/S)**2)
phi=lambda t:mp.e**(-t*t/2)/mp.sqrt(2*mp.pi)
xc=S*mp.atanh(C/S)
def mom(b):
    pts=[-mp.inf,0,xc/b,mp.inf]
    I=lambda h:mp.quad(lambda t:h(t)*phi(t),pts)
    return (I(lambda t:P(b*t) if t>0 else 0),I(lambda t:P(b*t) if t<=0 else 0),
            I(lambda t:P(b*t)**2 if t>0 else 0),I(lambda t:P(b*t)**2 if t<=0 else 0),
            I(lambda t:(b*dP(b*t))**2 if t>0 else 0),I(lambda t:(b*dP(b*t))**2 if t<=0 else 0))
def consts(b):
    Ip,In,Ip2,In2,Dp,Dn=mom(b); al=-Ip/In; lam=1/mp.sqrt(Ip2+al**2*In2); chi=lam**2*(Dp+al**2*Dn); return lam,al,chi
print(" b      lambda     alpha      chi=E[f'^2]")
for b in ['0.25','0.5','0.75','1','1.5','2','3']:
    l,a,c=consts(mp.mpf(b)); print(f" {b:5s} {mp.nstr(l,8):10s} {mp.nstr(a,8):10s} {mp.nstr(c,8)}")
for eps in ['0','0.01','0.03']:
    try:
        b=mp.findroot(lambda bb:consts(bb)[2]-(1+mp.mpf(eps)),mp.mpf('2'))
        l,a,c=consts(b); print(f"eps={eps}: b={mp.nstr(b,10)} lambda={mp.nstr(l,10)} alpha={mp.nstr(a,10)} chi={mp.nstr(c,10)}")
    except Exception as e: print("eps",eps,"keine Loesung:",e)
