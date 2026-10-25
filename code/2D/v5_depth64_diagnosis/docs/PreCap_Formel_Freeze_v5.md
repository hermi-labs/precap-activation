# PreCap-v9 — Formel-Freeze für v5

v5 übernimmt PreCap-v9 unverändert aus v3 und v4:

\[
z=ax,\quad u=z+\gamma z^2,\quad v=s\tanh(u/s)
\]
\[
\operatorname{PreCap}(x)=\operatorname{CapExp}(v),\qquad
\operatorname{CapExp}(v)=\begin{cases}e^v-1,&v\le c\\e^c-1+e^c(v-c),&v>c\end{cases}
\]

Fixe Parameter: \(c=1\), \(a=1\), \(\gamma=0.01\), \(s=1.5\).

Selfcheck: maximale absolute Abweichung zur Referenz auf dichtem Gitter kleiner als \(10^{-6}\).
