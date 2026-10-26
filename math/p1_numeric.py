import numpy as np, mpmath as mp
mp.mp.dps=40
A,G,S,C=1.0,0.01,1.5,1.0
def capexp(w,c=C): return np.where(w<=c,np.exp(np.minimum(w,c))-1,np.exp(c)-1+np.exp(c)*(w-c))
def precap(x,a=A,g=G,s=S,c=C):
    z=a*x;u=z+g*z*z;return capexp(s*np.tanh(u/s),c)
def dprecap(x,a=A,g=G,s=S,c=C):
    z=a*x;u=z+g*z*z;v=s*np.tanh(u/s);dc=np.where(v<=c,np.exp(np.minimum(v,c)),np.exp(c))
    return dc*(1/np.cosh(u/s)**2)*(1+2*g*z)*a
sup=np.exp(C)-1+np.exp(C)*(S-C); inf=np.exp(-S)-1
print("Supremum analytisch e^c-1+e^c(s-c) =",sup)
print("Infimum (nur bei gamma=0) e^-s - 1 =",inf)
# Minimum der quadratischen Vorstufe
zmin=-1/(2*G); umin=-1/(4*G); print("u minimal bei x=",zmin," u_min=",umin," f(x_min)=",float(precap(zmin)))
print("u=0 erneut bei x=",-1/G," f(-100)=",float(precap(-100.0)))
for xv in [-1e4,-1e3,-200,-150,-120,-100,-80,-60,-50,-30,-10,-5,-3,-2,-1,0,1,2,3,5,10]:
    print(f"x={xv:>8}: f={float(precap(xv)):+.6f}  f'={float(dprecap(xv)):+.3e}")
# Monotonie numerisch
xs=np.linspace(-49.9,60,2_000_001); f=precap(xs); print("monoton auf (-50,60):",bool((np.diff(f)>=-1e-15).all()))
xs2=np.linspace(-1000,-50.1,200001); print("monoton fallend auf (-1000,-50):",bool((np.diff(precap(xs2))<=1e-15).all()))
# Uebergangspunkt v=c
u_c=S*np.arctanh(C/S); x_c=(-1+np.sqrt(1+4*G*u_c))/(2*G); print("Uebergang zum linearen Ast bei u=",u_c," x=",x_c)
# maximale Steigung (Lipschitz) und wo
xx=np.linspace(-10,10,2_000_001); d=dprecap(xx); i=d.argmax(); print("max f' =",d[i],"bei x=",xx[i])
# Sprung der 2. Ableitung bei x_c (C1 aber nicht C2)
h=1e-5; d2=lambda t:(dprecap(t+h)-dprecap(t-h))/(2*h)
print("f'' links/rechts von x_c:",float(d2(x_c-1e-3)),float(d2(x_c+1e-3)))
# Ableitungsstetigkeit numerisch
print("f' links/rechts x_c:",float(dprecap(x_c-1e-9)),float(dprecap(x_c+1e-9)))
# Variante gamma=0, s=sqrt2
print("== Varianten ==")
for nm,(g,s,c) in {"Referenz":(0.01,1.5,1.0),"gamma0_s1.5":(0,1.5,1.0),"gamma0_s=sqrt2":(0,np.sqrt(2),1.0)}.items():
    print(nm,"sup",np.exp(c)-1+np.exp(c)*(s-c) if s>c else np.exp(s)-1,"inf",np.exp(-s)-1)
