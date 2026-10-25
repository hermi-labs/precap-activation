# PreCap-v9 — Formel-Freeze für v4

Diese diagnostische Studie übernimmt PreCap-v9 unverändert aus v3.

\[
z=a x,\qquad u=z+\gamma z^2,\qquad v=s\tanh(u/s)
\]
\[
\operatorname{CapExp}(v)=
\begin{cases}
 e^v-1,&v\le c\\
 e^c-1+e^c(v-c),&v>c
\end{cases}
\]
\[
\operatorname{PreCap-v9}(x)=\operatorname{CapExp}(v)
\]

Parameter: \(c=1\), \(a=1\), \(\gamma=0.01\), \(s=1.5\).

Der Runner-Selfcheck vergleicht die Implementierung auf einem dichten Gitter gegen diese Referenz und fordert maximale absolute Abweichung kleiner als \(10^{-6}\).
