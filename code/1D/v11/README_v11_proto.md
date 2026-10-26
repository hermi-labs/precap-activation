# v11 Proto - Schutzmechanismus im Motorsteuergerät

## Ziel
v11 testet eine neue Regler-Idee, ohne den stabilen Kern von PreCap direkt aufzugeben. Ausgangspunkt ist, dass PreCap in tiefen Netzen stark war, während flache Netze in v10 Teil A eher von SiLU oder GELU profitierten.

## Ausgangsaktivierungen
### CapExp
Mit Voraktivierung $z = a x$:

$$
\mathrm{CapExp}_{c,a}(x)=
\begin{cases}
\exp(z)-1, & z \le c \cr 
\exp(c)-1 + \exp(c)(z-c), & z>c
\end{cases}
$$

### PreCap
Mit $z=ax$ und $u=z+\gamma z^2$:

$$
\mathrm{PreCap}_{c,a,\gamma,s}(x)=\mathrm{CapExp}_{c,1}\left(s\tanh\left(\frac{u}{s}\right)\right)
$$

Die $\tanh$-Stufe ist der Schutzmechanismus: sie komprimiert schon vor dem CapExp-Zweig.

## Neue v11-Kandidaten
### 1. PreCapWG
Bis zu einer Schwelle $k$ bleibt das Signal fast frei; erst oberhalb davon wird weich komprimiert.

Mit $u=z+\gamma z^2$ und $r=|u|$:

$$
\phi_{k,s}(u)=\mathrm{sign}(u)
\begin{cases}
r, & r\le k \cr 
k + s\log\left(1+\frac{r-k}{s}\right), & r>k
\end{cases}
$$

und dann

$$
\mathrm{PreCapWG}(x)=\mathrm{CapExp}_{c,1}(\phi_{k,s}(u))
$$

### 2. HybridGate
Mit Gate

$$
g(z)=\sigma\left(\frac{|az|-k}{\beta}\right)
$$

und Mischung

$$
\mathrm{Hybrid}(x)=\bigl(1-g(z)\bigr)\thinspace \mathrm{CapExp}_{c,a}(x)+g(z)\thinspace \mathrm{PreCap}_{c,a,\gamma,s}(x)
$$

Für kleine Signale dominiert der offene Ast, für größere der sichere PreCap-Ast.
