# PreCap-v9 — Freeze für die große 2D-v3-Studie

Die Aktivierungsformel ist für v3 identisch zur validierten Referenz aus 1D-v9, 2D-v1b und 2D-v2:

\[
\operatorname{PreCap}(x)=x-\gamma\,\log\!\left(1+\exp\!\left(\frac{x-(c+a)}{s}\right)\right)
\]

mit

\[
c=1.0,\qquad a=1.0,\qquad \gamma=0.01,\qquad s=1.5.
\]

Damit konkret:

\[
\operatorname{PreCap}(x)=x-0.01\,\log\!\left(1+\exp\!\left(\frac{x-2}{1.5}\right)\right).
\]

Die Ableitung ist

\[
\operatorname{PreCap}'(x)=1-\frac{0.01}{1.5}\,\sigma\!\left(\frac{x-2}{1.5}\right),
\]

also stets im Intervall \([0.993,1]\). Die Implementierung verwendet numerisch stabil `F.softplus`, und der v3-Runner muss vor Start dieselbe Referenzprüfung mit `formula_max_abs_err=0` bestehen.
