# Phase 3a-3 und 3b – Entwurf (nicht eingefroren)

Stand: 26.10.2026. Das ist ein Entwurf. Er wird erst nach der Auswertung von 3a-2 festgelegt, weil das Lernraten-Raster und die Wahl der Vergleichsfunktionen vom Ergebnis abhängen. Eingefroren wird wie bisher per SHA-256 vor dem Start.

## Ausgangspunkt

- SPreCap erfüllt nur zwei Bedingungen: Mittelwert 0 und Varianz 1 bleiben über die Schichten erhalten.
- Der Gradientenfaktor pro Schicht ist $\chi_1=\mathbb{E}[f'(Z)^2]$. Bei SPreCap liegt er bei 1,1305, bei SELU bei 1,0716. Beide Werte sind exakt mit `scipy.quad` gerechnet, mit getrenntem Knick bei 0 (`math/p3a3_chi1_family.py`).
- Korrektur: Die früheren Werte 1,124 und 1,065 stammen aus einer Gauss-Hermite-Rechnung, die am Knick bei 0 ungenau ist. Die eingefrorenen Konstanten von SPreCap sind trotzdem korrekt (Mittelwert −4,8e−10, Varianz 1,0000000000).
- Über 100 Schichten wächst ein Faktor 1,13 auf $1{,}13^{100}\approx 2\cdot10^5$, ein Faktor 1,07 auf etwa $10^3$. SPreCap startet also deutlich "chaotischer" als SELU. Das passt dazu, dass SPreCap bei 50 Schichten etwa 1,2 Prozentpunkte schlechter war.

## Theoretische Grenze

Nach der Gaußschen Poincaré-Ungleichung gilt $\mathrm{Var}[f(Z)] \le \mathbb{E}[f'(Z)^2]$. Bei Varianz 1 folgt daraus $\chi_1 \ge 1$, mit Gleichheit nur für lineare Funktionen.

Ein selbstnormierendes, nichtlineares Netz liegt also immer leicht auf der chaotischen Seite. Die Stellschraube heißt deshalb nicht "$\chi_1 = 1$", sondern "Nichtlinearität gegen $\chi_1$ abwägen". Der Parameter $b$ (Eingangsskalierung) steuert genau das.

## SPreCap-Familie (s = 1,5, exakt gelöst)

| Variante | b | λ | α | χ₁ | Wertebereich |
|---|---|---|---|---|---|
| spreccap (bisher) | 1,0 | 0,806768696 | 2,396951309 | 1,1305 | [−1,50; 2,48] |
| spreccap_chi_selu (neu) | 0,71188112 | 1,102702500 | 2,053776267 | 1,0716 (= SELU) | [−1,76; 3,39] |
| spreccap_chi104 (neu) | 0,49626398 | 1,613729530 | 1,739903932 | 1,0400 | [−2,18; 4,97] |
| spreccap_e03 (bisher) | 0,41188963 | 1,985552225 | 1,608136397 | 1,0300 | [−2,48; 6,11] |

Hinweis: In 3a hatte spreccap_e03 den besseren mittleren Rang (3,7) als spreccap (4,8). Das spricht dafür, dass ein kleineres $\chi_1$ hilft.

## 3a-3: Hauptfrage

Ist SPreCap mit gleichem $\chi_1$ wie SELU (`spreccap_chi_selu`) mindestens gleich gut wie SELU? Damit wäre der Vergleich fair, und es bliebe nur der Unterschied "beschränkt gegen unbeschränkt".

| Punkt | Vorschlag |
|---|---|
| Zellen | MNIST, FashionMNIST × Tiefe 50, 100 (4 Zellen), Netz wie 3a-2 |
| Aktivierungen | spreccap, spreccap_chi_selu, spreccap_chi104, spreccap_e03, selu, spreccap_chi_selu_ls (6) |
| Lernbares s | `spreccap_chi_selu_ls`: wie spreccap_chi_selu, aber s je Schicht lernbar, $s=1{,}5\cdot e^{\rho}$ mit $\rho=0$ beim Start. λ, α und b bleiben fest. Beim Start ist die Variante also identisch mit spreccap_chi_selu, der Fixpunkt gilt nur dort. Zusätzlich wird das gelernte s je Schicht an den Checkpoints protokolliert. Hypothese: kein Vorteil (Konfidenz 0,6, wie Aaron vermutet). |
| LR-Raster | nach 3a-2 festlegen (Bereich um die dort gewählten LRs, 6 Stufen) |
| Seeds | Tuning 42000–42003, Bestätigung 42010–42024 (15) |
| Schritte | 1720, Checkpoints wie 3a-2, inklusive Gradientennorm erste/mittlere/letzte Schicht |
| Primärtest | spreccap_chi_selu gegen selu, 4 Zellen, Wilcoxon, Holm über 4. Für einen Vorteil müssen mindestens 3 von 4 Zellen besser sein und keine schlechter. Für "nicht schlechter" darf keine Zelle signifikant schlechter sein und der Median der Differenz muss bei ≤ 0,5 pp liegen (vorher festlegen). |
| Sekundär | Dosis-Wirkung: Testfehler gegen $\chi_1$ über die 4 SPreCap-Varianten (Spearman je Zelle), Divergenzrate, Breite des LR-Fensters |
| Umfang | etwa 576 Tuning- und 360 Bestätigungsruns, etwa 3–3,5 h auf dem M1 Pro |
| Sekundär (s) | spreccap_chi_selu_ls gegen spreccap_chi_selu, eigene Holm-Familie über 4 Zellen |

Erwartungen (vorläufig):

| Frage | Erwartung | Konfidenz |
|---|---|---|
| spreccap_chi_selu ist besser als spreccap | ja | 0,6 |
| spreccap_chi_selu ist nicht schlechter als selu | ja | 0,5 |
| Vorteil gegen selu bestätigt | nein | 0,7 |

## 3b: Weniger Normierung (Skizze)

Frage: Kann man mit SPreCap LayerNorm seltener einsetzen oder ganz weglassen und dabei weniger verlieren als mit GELU?

- **Aufbau:** Residual-MLP oder kleiner Transformer, LayerNorm in jeder k-ten Schicht, k ∈ {1, 4, 16, ∞}.
- **Aktivierungen:** spreccap(_chi_selu), selu, gelu. Zusätzlich gelu mit DyT statt LayerNorm als Referenz für bestehende normfreie Ansätze.
- **Kennzahlen:** Testfehler, Divergenzrate, Laufzeit pro Schritt (CPU und MPS).
- **Erwartung:** Bei k = 1 etwa gleichauf (Konfidenz 0,7). Bei großem k fällt SPreCap weniger ab als GELU (Konfidenz 0,5).
- **Laufzeit:** Die Ersparnis ist nach der DyT-Arbeit klein. Ohne Optimierung waren es etwa 8 % für das ganze Modell, nach Kompilierung kein Unterschied (Zhu et al. 2025, https://arxiv.org/html/2503.10622v2, Anhang C). Ein Laufzeitvorteil ist daher nur auf Hardware zu erwarten, auf der Reduktionen der Engpass sind.

## Test über MNIST hinaus (Skizze)

MNIST und FashionMNIST überzeugen öffentlich kaum. Vor einer breiten Veröffentlichung sollte mindestens ein Test laufen, der näher an der Praxis liegt:

1. **Kleiner Transformer auf Text** (zeichenweise, Tiny Shakespeare, etwa 1 MB): Pre-LN mit LayerNorm gegen ohne LayerNorm gegen DyT, jeweils mit SPreCap, SELU und GELU im MLP-Teil. Das verbindet sich direkt mit 3b und mit der aktuellen Debatte um normfreie Transformer.
2. **CIFAR-10** mit tiefem MLP oder kleinem CNN ohne Normierung, also ein schwereres Bildproblem als MNIST.

Rechnen: Der Engpass ist im Moment die CPU mit einem Thread je Worker. Für den Transformer bringt die GPU des M1 Pro (MPS oder MLX) mehr als ein eigener Kernel. Der Kernel lohnt sich vor allem, damit SPreCap im Training nicht langsamer ist als GELU. Plan: zuerst `torch.compile` oder MLX `mx.compile` als Basis messen, dann den Metal-Kernel bauen und gegen diese Basis prüfen (Ergebnisgleichheit auf 1e-6, Laufzeit).

## Literatur

- Klambauer et al. 2017, Self-Normalizing Neural Networks: https://papers.nips.cc/paper_files/paper/2017/hash/5d44ee6f2c3f71b73125876103c8f6c4-Abstract.html
- Zhu et al. 2025, Transformers without Normalization (DyT): https://arxiv.org/html/2503.10622v2
- Stronger Normalization-Free Transformers (Derf): https://arxiv.org/pdf/2512.10938v1
- SALU / SaluNet 2026: https://arxiv.org/pdf/2606.02927
- Brock et al. 2021, NFNets: https://arxiv.org/abs/2102.06171v1
