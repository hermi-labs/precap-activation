# Phase 2 – Literaturrecherche und Einordnung

Stand: 26.10.2026. Ziel ist, für jeden Baustein von PreCap festzustellen, was es schon gibt und was neu sein könnte.

## 1. Vorgehen (Protokoll)

- **Akademische Suche:** 10 Suchanfragen, unter anderem zu selbstnormierenden beschränkten Aktivierungen, SERLU, bidirektionaler Selbstnormierung, skaliertem tanh nach LeCun, Verkettungen von exp und tanh, beschränktem ELU, SELU-Varianten, Edge of Chaos und Stabilität ohne Normalisierung. Rohdaten: `literatur/p2_academic.json`.
- **Survey:** Volltextsuche im Survey von Kunc & Kléma mit 400 Aktivierungsfunktionen (https://arxiv.org/abs/2402.09092). Gesucht wurde nach `exp(tanh`, `e^tanh`, `tanh(e`, bounded/clipped ELU, self-normalizing, scaled/penalized tanh, soft clipping, CELU und tanh in der Nähe von ELU.
- **Gelesene Arbeiten:** TeLU, DyT, NF-ResNets, bSNN, SERLU, Large-scale SNN, Redefining Self-Normalization, SaluNet, Z-Plane, Hayou et al. sowie die Funde vom 25.10. (TERELU, TanhExp, Soft Exponential, xIELU).
- **Patente:** nur eine einzige Suchanfrage, ohne relevanten Treffer. Das ist **keine** ausreichende Patentrecherche.

## 2. Ergebnis je Baustein

| Baustein | Was es gibt | Bewertung |
|---|---|---|
| **CapExp** | Exakt \(e^c\,\mathrm{ELU}(x-c)+(e^c-1)\), also ELU (Clevert et al. 2015, https://arxiv.org/abs/1511.07289) | nicht neu (sicher) |
| **tanh vor der Aktivierung** (\(s\tanh(x/s)\)) | DyT ersetzt LayerNorm durch \(\gamma\tanh(\alpha x)+\beta\), damit Transformer ohne Normalisierung gleich gut oder besser laufen (Zhu et al. 2025, https://arxiv.org/abs/2503.10622). Soft Clipping steht als beschränkte Aktivierung im Survey (https://arxiv.org/abs/2402.09092). | Die Idee, Extremwerte mit tanh weich abzuschneiden, ist bekannt und aktuell sehr gefragt |
| **Verkettung exp und tanh** | TanhExp/TeLU: \(x\tanh(e^x)\) (https://arxiv.org/abs/2402.02790). Mish: \(x\tanh(\mathrm{softplus}(x))\). Variante \(x\ln(1+\tanh(e^x))\) im Survey. | Verwandt, aber andere Reihenfolge und Form |
| **PreCap selbst** \(\mathrm{ELU}\)-artig\((s\tanh(x/s))\) | Im Survey keine Form \(e^{s\tanh(x/s)}-1\) oder ELU∘tanh gefunden. Auch die Suchen fanden nichts. | wahrscheinlich in dieser Form nicht veröffentlicht, Konfidenz etwa 0,6. Als Beitrag aber klein, weil es eine naheliegende Verkettung ist |
| **Stabilität durch Beschränkung** | Liew et al. 2016, „Bounded activation functions for enhanced training stability“ (https://doi.org/10.1016/j.neucom.2016.08.037). SaluNet 2026 mit beschränkter, lernbarer SALU als Ersatz für Normalisierung (https://arxiv.org/abs/2606.02927). Z-Plane 2026 mit beschränkter Aktivierung und 100-Schichten-MLP ohne Normalisierung (https://arxiv.org/abs/2606.15669). | **nicht neu.** Unser V17-Befund bestätigt eine bekannte Idee |
| **SPreCap** (λ, α nach SELU) | SELU (https://arxiv.org/abs/1706.02515) ist das Rezept. NF-ResNets skalieren jede Aktivierung mit \(\gamma=1/\sqrt{\mathrm{Var}\,g(z)}\), Abschnitt 4.3 (https://arxiv.org/abs/2101.08692). bSNN gibt affine Konstanten für tanh, ELU und GELU an, mit \(\mathbb E[f^2]=\mathbb E[f'^2]=1\) (https://arxiv.org/abs/2006.12169). Chen et al. definieren Selbstnormierung über \(\mathbb E[f'^2]=1+\varepsilon\) (https://openreview.net/forum?id=gfwfOskyzSx), und die ImageNet-Version erreicht 75,23 % mit bSELU-BSTD (https://www.sciopen.com/article/10.1016/j.jai.2024.05.001). | **Das Verfahren ist bekannt.** Neu wäre nur die konkrete Anwendung auf eine beschränkte ELU-Form. Kleiner Beitrag |

## 3. Was die Literatur für SPreCap bedeutet

Nach Chen et al. braucht eine selbstnormierende Funktion zusätzlich \(\mathbb E[f'(z)^2]\approx1+\varepsilon\) mit \(\varepsilon\approx1/L\) bei Tiefe \(L\). SPreCap aus Phase 1 hat \(\chi=1{,}13\). Das erklärt die starke Gradientenverstärkung in unserer Simulation.

Deshalb habe ich SPreCap um eine Eingangsskala \(b\) erweitert: \(f(x)=\lambda\,\mathrm{PreCap}_0(bx)\) für \(x>0\) und \(\lambda\alpha\,\mathrm{PreCap}_0(bx)\) für \(x\le0\). Es gelten drei Bedingungen: \(\mathbb E f=0\), \(\mathbb E f^2=1\) und \(\mathbb E f'^2=1+\varepsilon\). Skripte: `math/p2_spreccap_v2.py` und `math/p2_deep_sim_v2.py`.

| Variante | b | λ | α | χ | Gradientenverstärkung über 100 Schichten (log10) | nichtlinearer Varianzanteil |
|---|---|---|---|---|---|---|
| SELU | – | 1,0507 | 1,6733 | 1,072 | +1,57 | 0,028 |
| SPreCap (b = 1) | 1 | 0,8068 | 2,3970 | 1,130 | +2,70 | 0,046 |
| SPreCap-ε 0,03 | 0,4119 | 1,9856 | 1,6081 | 1,030 | +0,68 | 0,014 |
| SPreCap-ε 0,01 | 0,2125 | 4,1691 | 1,2969 | 1,010 | +0,24 | 0,004 |

Mit ε = 0 gibt es keine Lösung, außer im Grenzfall einer linearen Funktion. Das passt genau zum Satz aus bSNN: Eine nichtlineare Funktion mit Mittelwert 0 kann nicht gleichzeitig \(\mathbb E f^2=\mathbb E f'^2=1\) erfüllen.

**Kernaussage:** Weniger Gradientenverstärkung kostet Nichtlinearität. Bei SELU ist das genauso, es ist eine Eigenschaft des Prinzips und kein Nachteil von PreCap. SPreCap mit b = 1 ist nichtlinearer als SELU (0,046 gegenüber 0,028), verstärkt Gradienten aber stärker.

## 4. Gesamteinordnung

- **Nicht neu:** CapExp (= ELU), Stabilität durch Beschränkung, das Rezept für selbstnormierende Konstanten.
- **Wahrscheinlich neu, aber klein:** die konkrete Verkettung \(\mathrm{CapExp}(s\tanh(x/s))\) und ihre selbstnormierende Version SPreCap samt den Konstanten.
- **Bisher eigener Befund mit Wert:** die saubere, präregistrierte Negativ-Studie. Der scheinbare Vorteil einer neuen Aktivierung verschwindet mit fairem Lernraten-Tuning und lässt sich vollständig durch die Steigung am Ursprung (PreCap ≈ 2·SiLU) erklären. Solche methodischen Ergebnisse sind in der Literatur zu Aktivierungsfunktionen selten und nützlich, denn viele der 400 Funktionen im Survey wurden ohne Tuning der Lernrate verglichen.

**Empfehlung für das Paper:**

- Kein Titel der Art „eine neue, bessere Aktivierung“.
- Stattdessen etwa: „Bounded exponential activations: a preregistered study showing that apparent gains vanish under fair learning-rate tuning, with a self-normalizing variant (SPreCap)“.
- Phase 3 sollte SPreCap fair gegen SELU, bSNN-tanh und DyT-artige Varianten in tiefen Netzen ohne Normalisierung testen.

Konfidenz in diese Einordnung: etwa 0,75. Hauptgrund für die Unsicherheit sind die fehlende Patentrecherche und die Tatsache, dass der Survey nur per Volltextsuche und nicht Eintrag für Eintrag geprüft wurde.

## 5. Offene Punkte

- Patentrecherche gezielt nach „bounded exponential activation“ und „tanh preprocessing activation“.
- Survey-Abschnitte zu beschränkten und zusammengesetzten Funktionen vollständig lesen, nicht nur per Volltextsuche.
- Die bSNN-Konstanten für tanh (a = 1,4674, b = 0,3885) als Vergleichskandidaten in Phase 3 aufnehmen.
