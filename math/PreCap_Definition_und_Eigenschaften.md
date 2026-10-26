# PreCap – Definition und Eigenschaften (Phase 1)

Stand: 26.10.2026. Alle Aussagen sind symbolisch (sympy) hergeleitet und numerisch geprüft (numpy, mpmath mit 30 Stellen). Die Skripte und ihre Ausgaben liegen in diesem Ordner (`p1_*.py`, `p1_*.out`).

## 1. Definition

$$
z=a\thinspace x,\qquad u=z+\gamma z^2,\qquad v=s\tanh\left(\frac{u}{s}\right),\qquad
\mathrm{PreCap}(x)=\mathrm{CapExp}_c(v)
$$

$$
\mathrm{CapExp}_c(w)=\begin{cases}e^{w}-1,& w\le c\cr  (e^{c}-1)+e^{c}(w-c),& w>c\end{cases}
$$

Referenzparameter aller Läufe: $a=1,\ \gamma=0{,}01,\ s=1{,}5,\ c=1$.

Bezug zu ELU (exakt, Fehler unter 4e-15): $\mathrm{CapExp}_c(w)=e^{c}\thinspace \mathrm{ELU}(w-c)+(e^{c}-1)$. CapExp ist also ein verschobenes und skaliertes ELU (Clevert et al. 2015, https://arxiv.org/abs/1511.07289). Neu an PreCap ist nur die vorgeschaltete Stufe $s\tanh(\cdot/s)$ samt dem quadratischen Term.

## 2. Ableitung

$$
\mathrm{PreCap}'(x)=a\thinspace (1+2\gamma a x)\thinspace \mathrm{sech}^2\left(\frac{u}{s}\right)\cdot
\begin{cases}e^{v},& v\le c\cr  e^{c},& v>c\end{cases}
$$

Für die Umsetzung sollte man $\mathrm{sech}^2=1-\tanh^2$ verwenden. `1/cosh²` läuft für große $|u|$ über.

## 3. Glattheit

- PreCap ist **stetig differenzierbar ($C^1$)**, aber nicht $C^2$.
- Der Übergang zum linearen Ast liegt bei $v=c$, also bei $u_c=s\thinspace \mathrm{artanh}(c/s)=1{,}2071$ bzw. $x_c=1{,}1928$.
- Dort ist die erste Ableitung stetig (1,54618 links wie rechts), die zweite springt dagegen von −0,50 auf −1,38.
- Grund: $\mathrm{CapExp}''$ springt bei $c$ von $e^c$ auf 0. ELU mit $\alpha=1$ ist ebenfalls nur $C^1$, SELU hat sogar einen Knick bei 0.

## 4. Wertebereich, Monotonie und ein Formelfehler

- **Obere Schranke (exakt):** $\sup \mathrm{PreCap}=e^{c}-1+e^{c}(s-c)=3{,}0774$, weil $v<s$ gilt.
- **Untere Schranke:** $e^{-s}-1=-0{,}7769$.
- **Maximale Steigung:** 1,621 bei $x\approx0{,}905$, der globale Lipschitz-Wert.
- **Monoton steigend** auf $(-50,\infty)$, geprüft auf 2 Mio. Punkten.

**Formelfehler durch $\gamma z^2$:**

- $u=z+\gamma z^2$ hat sein Minimum bei $z=-1/(2\gamma)=-50$ ($u=-25$). Für $x<-50$ steigt $u$ wieder an.
- Bei $x=-100$ ist $u=0$ und damit PreCap$(-100)=0$, mit Steigung −1.
- Für $x<-120$ liefert PreCap den **Höchstwert 3,077**.

| x | −1000 | −120 | −100 | −80 | −50 | −10 | −1 | 0 | 1 | 3 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| PreCap | 3,077 | 3,077 | 0,000 | −0,777 | −0,777 | −0,777 | −0,580 | 0 | 1,413 | 2,947 | 3,077 |

In allen bisherigen Läufen lagen die Voraktivierungen weit innerhalb von ±10, praktisch hat das also nichts verändert. Für eine Veröffentlichung ist es aber ein Fehler. Der Beitrag von γ ist im normalen Bereich winzig: Er ändert nur den Koeffizienten von $x^2$ um 0,01.

**Korrektur: $\gamma=0$.** Dann ist PreCap auf ganz ℝ streng monoton, und es gilt

$$
\mathrm{PreCap}_0(x)=\mathrm{CapExp}_c\left(s\tanh\frac{x}{s}\right),\qquad
\mathrm{PreCap}_0(x)\in\left(e^{-s}-1,\ e^{c}-1+e^{c}(s-c)\right).
$$

## 5. Taylor-Entwicklung am Ursprung

Allgemein, solange $v<c$ gilt:

$$
\mathrm{PreCap}(x)=a\thinspace x+a^2\left(\gamma+\tfrac12\right)x^2+\frac{a^3\big(s^2(1+6\gamma)-2\big)}{6s^2}x^3+\frac{a^4\big(s^2(12\gamma^2+12\gamma+1)-24\gamma-8\big)}{24s^2}x^4+\dots
$$

| Funktion | $x$ | $x^2$ | $x^3$ | $x^4$ |
|---|---|---|---|---|
| PreCap (Referenz) | 1 | 0,51 | 0,0285 | −0,1059 |
| PreCap$_0$, $s=1{,}5$ | 1 | 0,5 | 0,0185 | −0,1065 |
| PreCap$_0$, $s=\sqrt2$ | 1 | 0,5 | **0** | −0,125 |
| 2·SiLU | 1 | 0,5 | 0 | −0,0417 |
| 2·GELU | 1 | 0,798 | 0 | −0,133 |
| SiLU | 0,5 | 0,25 | 0 | −0,0208 |

**Kernaussage:** Nahe null ist PreCap bis zur dritten Ordnung fast identisch mit 2·SiLU. Mit $\gamma=0$ und $s=\sqrt2$ stimmen beide bis $x^3$ exakt überein. In den Experimenten lagen 70–97 % der Voraktivierungen unter 0,5. Das erklärt mathematisch, warum der Vorteil bei fester Lernrate verschwindet, sobald die Lernrate fair gewählt wird (V18, 3D-V1). PreCap und SiLU unterscheiden sich erst bei $|x|\gtrsim1$, nämlich durch die Begrenzung.

## 6. Parameter reduzieren

- **a** ist bei Netzen mit Gewichten überflüssig, weil $a$ in die Gewichte $W$ wandert. Nur die Initialisierung ändert sich. Daher $a=1$.
- **γ** ist überflüssig und sogar schädlich (Abschnitt 4). Daher $\gamma=0$.
- **c und s** lassen sich nicht zusammenfassen, beide bestimmen die Form. Es gibt aber zwei natürliche Einparameter-Familien:
  - **c = 1 fest, s frei.** Für $s\le c$ wird der lineare Ast nie erreicht, und PreCap ist $e^{s\tanh(x/s)}-1$. Diese Form ist beliebig oft differenzierbar ($C^\infty$) und liegt in $(e^{-s}-1,\ e^{s}-1)$.
  - **s = √2 fest, c frei.** Das ist die Ankopplung an 2·SiLU bis zur dritten Ordnung.
- **Vorschlag für Phase 3:** PreCap$_0$ mit $c=1$ und $s\in\lbrace 1{,}5;\ \sqrt2\rbrace $. Damit bleibt ein einziger freier Parameter.

## 7. SPreCap – selbstnormierendes PreCap

Der Ansatz folgt SELU (Klambauer et al. 2017, https://arxiv.org/abs/1706.02515):

$$
\mathrm{SPreCap}(x)=\lambda\cdot\begin{cases}\mathrm{PreCap}_0(x),& x>0\cr  \alpha\thinspace \mathrm{PreCap}_0(x),& x\le0\end{cases}
$$

Gesucht sind λ und α so, dass für $Z\sim\mathcal N(0,1)$ gilt: $\mathbb E[\mathrm{SPreCap}(Z)]=0$ und $\mathrm{Var}=1$. Die Konstanten sind exakt berechnet (mpmath, Integration an den Knickstellen 0 und $x_c$ aufgeteilt). Zur Kontrolle liefern die veröffentlichten SELU-Konstanten mit demselben Verfahren $\mathbb E=-1{,}5\cdot10^{-32}$ und Var = 1,0000000000.

| Variante (c = 1, s = 1,5) | λ | α | χ = E[f′²] |
|---|---|---|---|
| **SPreCap (γ = 0)** | **0,806768696** | **2,396951309** | 1,1305 |
| SPreCap (γ = 0,01) | 0,798686312 | 2,430216844 | 1,1319 |
| Alternative λ·(PreCap₀ − β), β = 0,305560 | 1,064209617 | – | 1,1762 |
| SELU zum Vergleich | 1,050700987 | 1,673263242 | 1,0716 |

**Eigenschaften von SPreCap:**

- $f(0)=0$, streng monoton, Werte in $(-1{,}502;\ 2{,}483)$.
- Steigung bei 0 links 1,934, rechts 0,807. Wie SELU hat SPreCap einen Knick bei 0.

**Stabilität des Fixpunkts (Mittelwert 0, Varianz 1):**

- Die Abbildung $(\mu,\nu)\mapsto(\tilde\mu,\tilde\nu)$ hat am Fixpunkt Jacobi-Eigenwerte vom Betrag höchstens 0,64. Bei SELU sind es 0,78.
- Im Bereich $\omega\in[-0{,}1;0{,}1]$, $\tau\in[0{,}8;1{,}25]$ bleibt die Spektralnorm unter 0,70, bei SELU unter 0,94.
- Der Fixpunkt zieht also stärker an als bei SELU. Diese Werte stammen aus Gauß-Hermite-Quadratur und sind auf etwa 1e-3 genau.
- **Einschränkung:** Für eine begrenzte Funktion ist eine starke Anziehung der Varianz fast selbstverständlich. Das ist kein so starkes Ergebnis wie bei SELU, das nach oben unbeschränkt ist.

### Simulation: 100 Schichten ohne Normalisierung

Aufbau: Breite 256, $W\sim\mathcal N(0,1/n)$, kein Bias, 512 Eingaben, Median über 10 Seeds (`p1_deep_sim.out`).

| Aktivierung | Var nach 10 | Var nach 50 | Var nach 100 | Mittel nach 100 | log10 Gradientenverstärkung über 100 Schichten |
|---|---|---|---|---|---|
| SELU | 0,988 | 0,987 | 1,013 | +0,002 | +1,57 |
| **SPreCap** | 0,992 | 0,995 | 0,993 | +0,000 | +2,70 |
| SPreCap (Verschiebung) | 0,983 | 0,982 | 0,995 | +0,001 | +3,55 |
| PreCap (Referenz, ohne Konstanten) | 0,867 | 0,879 | 0,801 | +0,297 | +1,34 |
| ELU | 0,058 | 0,002 | 0,001 | +0,000 | −1,54 |
| SiLU | ≈0 | ≈0 | ≈0 | 0 | −29,95 (verschwindet) |
| 2·SiLU | 246 | 1,9e14 | 9,5e28 | 1,9e14 | +14,86 (explodiert) |

Startet man mit Eingangsvarianz 4 statt 1, ergibt sich dasselbe Bild: SPreCap liegt nach 10 Schichten bei 1,000, SELU bei 1,118.

**Deutung:**

- SPreCap ist selbstnormierend wie SELU, und zwar etwas schneller.
- Der Gradient wird über 100 Schichten aber stärker verstärkt (Faktor etwa 500 gegenüber etwa 37 bei SELU), weil $\chi=1{,}13$ über 1,07 liegt. Bei sehr tiefen Netzen kann das eine kleinere Lernrate oder Gradient-Clipping nötig machen.
- Bemerkenswert ist das schlichte PreCap: Es bleibt ohne jede Normalisierung in einem festen Bereich (Varianz etwa 0,8). SiLU verschwindet dagegen, 2·SiLU explodiert. Das ist die mathematische Seite des V17-Befunds.

## 8. Rechenaufwand

Messung auf dem MacBook Pro M1 Pro mit torch 2.8, Tensor 4096 × 1024, Vorwärts- und Rückwärtsdurchlauf, ohne `torch.compile` (`p1_cost.out`):

| | ReLU | SiLU | GELU | ELU | SELU | CapExp | PreCap | PreCap₀ | SPreCap |
|---|---|---|---|---|---|---|---|---|---|
| CPU (ms) | 1,07 | 3,62 | 7,53 | 6,01 | 6,51 | 8,46 | 17,92 | 14,63 | 19,35 |
| GPU/MPS (ms) | 1,39 | 1,50 | 1,51 | 1,42 | 1,46 | 6,48 | 12,94 | 9,95 | 13,87 |

ReLU, SiLU, GELU, ELU und SELU laufen als fertige Einzelkernel. PreCap besteht in Python aus etwa 8 Einzeloperationen und ist deshalb auf der GPU etwa 7–9-mal langsamer. Für Phase 3 sollte man PreCap mit `torch.compile` oder als eigenen Kernel zusammenfassen, sonst ist der Zeitvergleich unfair. Rechnerisch braucht PreCap₀ ein tanh, ein exp und einen Vergleich, also etwa so viel wie GELU in der tanh-Näherung.

## 9. Zusammenfassung und Folgen

1. PreCap ist $C^1$, nach oben begrenzt (3,077) und mit $\gamma=0$ streng monoton. Die bisherige Form mit $\gamma=0{,}01$ hat einen Fehler für $x<-50$. Er war in der Praxis folgenlos, ist aber zu korrigieren.
2. Nahe null ist PreCap ≈ 2·SiLU bis zur dritten Ordnung. Das erklärt die Ergebnisse von V18 und 3D-V1 vollständig.
3. Der eigentliche Unterschied ist die Begrenzung. Sie macht PreCap ohne Normalisierung stabil (V17 und Simulation), kostet aber Ausdrucksstärke bei großen Voraktivierungen (gyroid in 3D-V1).
4. **SPreCap** mit λ = 0,806769 und α = 2,396951 ist eine echte selbstnormierende Variante. Es ist der aussichtsreichste eigene Beitrag für das Paper und sollte in Phase 3 gegen SELU in tiefen Netzen ohne Normalisierung antreten.
5. In Phase 3 gilt deshalb: PreCap₀ (γ = 0) statt der alten Referenz, SPreCap gegen SELU, PreCap mit `torch.compile` für faire Zeitmessung und ein Lernraten-Raster bis 3e-2.

Konfidenz: Die mathematischen Aussagen in den Abschnitten 2–7 sind nachgerechnet, etwa 0,95. Dass SPreCap in echten Netzen hilft, ist dagegen offen, etwa 0,3.
