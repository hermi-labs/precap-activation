# Phase 1: symbolische und numerische Eigenschaften von PreCap
import sympy as sp, numpy as np, mpmath as mp
x=sp.symbols('x',real=True); a,g,s,c=sp.symbols('a gamma s c',positive=True)
z=a*x; u=z+g*z**2; v=s*sp.tanh(u/s)
lo=sp.exp(v)-1; hi=(sp.exp(c)-1)+sp.exp(c)*(v-c)
print("== Taylor am Ursprung (unterer Ast, gilt solange v<c) ==")
T=sp.series(lo,x,0,5).removeO(); T=sp.expand(T)
for k in range(1,5): print(f"x^{k}:",sp.simplify(T.coeff(x,k)))
P={a:1,g:sp.Rational(1,100),s:sp.Rational(3,2),c:1}
Tn=sp.expand(T.subs(P)); print("Referenz a=1,g=0.01,s=1.5:",[float(Tn.coeff(x,k)) for k in range(1,5)])
print("2*SiLU:",sp.series(2*x/(1+sp.exp(-x)),x,0,5))
print("GELU*2:",[float(sp.series(x*(1+sp.erf(x/sp.sqrt(2))),x,0,5).removeO().coeff(x,k)) for k in range(1,5)])
print("== Ableitungen ==")
d1lo=sp.diff(lo,x); print("f'(x) unterer Ast =",sp.simplify(d1lo))
d1hi=sp.diff(hi,x); print("f'(x) oberer Ast =",sp.simplify(d1hi))
# Glattheit am Uebergang v=c: zweite Ableitung von CapExp springt von e^c auf 0
w=sp.symbols('w'); print("CapExp'' links/rechts bei w=c:",sp.diff(sp.exp(w)-1,w,2).subs(w,c),0)
