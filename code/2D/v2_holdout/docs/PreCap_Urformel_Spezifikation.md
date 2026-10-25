# PreCap — Urformel und eingefrorene Spezifikation

**Status:** Referenzdefinition für CapExp 1D-v9, 2D-v1b und den unabhängigen 2D-v2-Holdout.  
**Version:** `precap-v9`, eingefroren vor Auswertung des v2-Holdouts.  
**Zweck:** Eine glatte, asymptotisch begrenzte Korrektur der Identität, die große positive Aktivierungen selektiv dämpft und kleine Werte nahezu unverändert lässt.

## Exakte Formel

Für einen skalaren Voraktivierungswert \(x\) gilt

\[
\operatorname{PreCap}(x)=x-\gamma\,\operatorname{softplus}\!\left(\frac{x-(c+a)}{s}\right).
\]

Dabei ist

\[
\operatorname{softplus}(z)=\log(1+e^z).
\]

Die eingefrorenen Parameter lauten

\[
c=1.0,\qquad a=1.0,\qquad \gamma=0.01,\qquad s=1.5.
\]

Damit lautet die konkret verwendete Urformel

\[
\boxed{\operatorname{PreCap}(x)=x-0.01\,\log\!\left(1+\exp\!\left(\frac{x-2}{1.5}\right)\right).}
\]

Die Implementierung nutzt PyTorch `F.softplus`; dies ist die numerisch stabile Auswertung der oben definierten Softplus-Funktion.

## Verhalten

- Für \(x\ll 2\) ist der Softplus-Term nahezu null, also \(\operatorname{PreCap}(x)\approx x\).
- Im Übergangsbereich um \(x=2\) setzt die Dämpfung glatt ein.
- Für \(x\gg 2\) ist \(\operatorname{softplus}((x-2)/1.5)\approx(x-2)/1.5\); daher ist die asymptotische Steigung \(1-\gamma/s=0.993\), nicht null.
- PreCap ist folglich **kein harter Cap** und keine Sättigungsfunktion: Die Aktivierung bleibt monoton steigend, aber große positive Werte werden leicht zurückgenommen.

## Ableitung

Mit der Sigmoidfunktion \(\sigma(z)=1/(1+e^{-z})\) ist

\[
\frac{d}{dx}\operatorname{PreCap}(x)=1-\frac{\gamma}{s}\,\sigma\!\left(\frac{x-(c+a)}{s}\right).
\]

Für die eingefrorenen Werte liegt die Ableitung strikt im Intervall

\[
0.993\leq\operatorname{PreCap}'(x)\leq1.
\]

Damit ist PreCap glatt, streng monoton und nahezu identitätsnah. Die Wirkung entsteht nicht durch Gradientensperren, sondern durch eine kleine, kontinuierliche Rücknahme hoher positiver Voraktivierungen.

## Messgröße der Intervention

Der während des Trainings protokollierte Cap-Anteil ist

\[
\operatorname{cap\_fraction}=\mathbb{E}\left[\mathbf{1}\{\operatorname{PreCap}(x)\neq x\}\right],
\]

wobei die Implementierung eine numerische Schwelle von \(10^{-7}\) verwendet. Er zeigt, bei welchem Anteil der Aktivierungen die glatte Korrektur messbar von der Identität abweicht; er misst **nicht** die Stärke der Korrektur pro Wert.

## Validierungsnachweis

Die 2D-v1b- und 2D-v2-Implementierung wurden gegen die 1D-v9-Referenzformel auf 10.001 Werten im Bereich \([-20,20]\) geprüft. Das Ergebnis war `formula_max_abs_err=0`, also bitgenaue Übereinstimmung der verwendeten Auswertung mit der Referenzimplementierung.

## Geltungsbereich

Diese Formel und Parameter sind für den laufenden unabhängigen v2-Holdout eingefroren. Änderungen an \(c,a,\gamma,s\), Modell, Aufgaben, Seeds, Optimierer, Trainingsdauer oder Auswertung wären ein neues Experiment und dürfen nicht in den v2-Holdout einfließen.
