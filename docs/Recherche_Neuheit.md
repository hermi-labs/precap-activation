# Recherche: Neuheit und Name (Abschluss vor v0.1)

Stand: 26.10.2026. Ergänzt die Phase-2-Literaturrecherche (`literatur/Phase2_Literaturrecherche.md`) um die Prüfung vor der Veröffentlichung.

## Ergebnis in einem Satz

Die konkrete Formel von PreCap/SPreCap habe ich in der Literatur nicht gefunden. Die einzelnen Bausteine sind aber bekannt: weiche tanh-Begrenzung, ELU-Form, SELU-artige Konstanten und auch ein leicht überkritischer Gradientenfaktor $\chi_1>1$. Der Beitrag ist deshalb klein und klar umrissen.

Sicherheit: etwa 0,8, dass es keine identische veröffentlichte Formel gibt. Nicht auszuschließen sind unveröffentlichte oder schlecht auffindbare Varianten in den über 400 bekannten Aktivierungsfunktionen ([Kunc & Kléma 2024](https://arxiv.org/abs/2402.09092)).

## Formel (zur Abgrenzung)

$$
y(x)=\lambda\cdot\begin{cases}1 & x>0\cr  \alpha & x\le 0\end{cases}\cdot\mathrm{capexp}\big(s\tanh(bx/s)\big),\qquad
\mathrm{capexp}(w)=\begin{cases}e^w-1 & w\le c\cr  e^c-1+e^c(w-c) & w>c\end{cases}
$$

mit $s=1{,}5$, $c=1$. Die Ausgabe ist beidseitig beschränkt, zum Beispiel für SPreCap-e03 auf $[-2{,}48;\ 6{,}11]$.

## Nächste Verwandte

| Arbeit | Formel / Idee | Gemeinsamkeit | Unterschied |
|---|---|---|---|
| SELU, Klambauer et al. 2017 ([arXiv:1706.02515](https://arxiv.org/pdf/1706.02515)) | $\lambda$ mal ELU mit Faktor $\alpha$, Fixpunkt Mittel 0 / Varianz 1 | Rezept für $\lambda,\alpha$ | SELU ist nach oben unbeschränkt |
| bSELU, Chen et al. 2024, „Large-scale self-normalizing neural networks“, J. Automation and Intelligence 3, 101–110 ([PDF](https://www.sciopen.com/local/article_pdf/10.1016/j.jai.2024.05.001.pdf)) | $\lambda\thinspace (x$ bzw. $\alpha e^{\beta x}-\alpha)$, Konstanten über $\mathbb E[f]=\varepsilon_\mu$, $\mathbb E[f^2]=1-\varepsilon_q$, $\phi(1)=1+\varepsilon_\phi$ | **Sehr eng:** Auch dort wird der Gradientenfaktor bewusst leicht über 1 gelegt (bSELU-FSTD mit $\varepsilon_\phi=1/L$). Bestes Ergebnis auf einem 56-Schichten-CNN ohne Shortcuts: 88,54 % gegen SELU 84,81 % (CIFAR-10) ([PDF](https://www.sciopen.com/local/article_pdf/10.1016/j.jai.2024.05.001.pdf)) | Unbeschränkt nach oben, keine tanh-Vorstufe. Unsere $\chi_1$-Familie ist also eine Anwendung derselben Idee auf eine beschränkte Form, keine neue Idee |
| Bidirectionally Self-Normalizing NNs, Lu, Gould, Ajanthan 2020/21 ([arXiv:2006.12169](https://www.arxiv.org/abs/2006.12169)) | Gauß-Poincaré-normierte Aktivierungen + orthogonale Gewichte | Beschränkte, selbstnormierende Aktivierung (z. B. tanh-Variante) möglich | Andere Funktion, andere Normierungsbedingung |
| SaluNet, Zaied 2026 ([arXiv:2606.02927](https://arxiv.org/abs/2606.02927)) | $\mathrm{SALU}(x)=ax/\sqrt{1+abx^2}$, beschränkt und lernbar | Beschränkte Aktivierung als Ersatz für Normalisierung | Keine Exponentialform, keine Selbstnormierungs-Fixpunkte; getestet auf ResNet und Transformer |
| DyT, Zhu et al., CVPR 2025 ([arXiv:2503.10622](https://arxiv.org/abs/2503.10622)) | $\tanh(\alpha x)$ als Ersatz für LayerNorm | tanh-Begrenzung wie unsere Vorstufe | Ersetzt die Normalisierung, nicht die Aktivierung |
| Derf, Chen et al. 2025, „Stronger Normalization-Free Transformers“ ([arXiv:2512.10938](https://arxiv.org/abs/2512.10938)) | $\mathrm{erf}(\alpha x+s)$ | Punktweise, beschränkt, normalisierungsfrei | Wie DyT: Normierungsersatz |
| TanhExp, Liu & Di 2020 ([arXiv:2003.09855](https://arxiv.org/abs/2003.09855)) | $x\tanh(e^x)$ | tanh und exp kombiniert | Umgekehrte Verkettung, unbeschränkt |
| Criticality versus uniformity, Bukva et al. 2023 ([arXiv:2304.04784](https://arxiv.org/abs/2304.04784)) | tanh am Rand des Chaos | Sättigung kann trotz kritischer Init das Training bremsen | Passt zu unserem 3a-4-Befund, dass der kritische Punkt bei PreCap0 fragil ist |
| Edge of Chaos, Hayou et al. 2019 ([arXiv:1902.06853](https://arxiv.org/html/1902.06853v2)) | Init auf der EOC-Kurve | Grundlage für $\chi_1$ und Phase 3a-4 | Theorie, keine neue Funktion |

## Was davon wirklich neu ist

1. **Die konkrete Form:** CapExp (exakt ein verschobenes ELU) hinter einer weichen tanh-Begrenzung, mit SELU-artigen Konstanten. Damit entsteht eine glatte, monotone, beidseitig beschränkte Aktivierung mit Selbstnormierung.
2. **Der empirische Befund:** In 50 und 100 Schichten tiefen MLPs ohne Normalisierung ist SPreCap-cs gleichauf mit SELU (Phase 3a-3, 4 von 4 Zellen innerhalb ±0,2 pp), obwohl die Ausgabe beschränkt ist.
3. **Die $\chi_1$-Dosis-Wirkung bei einer beschränkten Funktion:** Der Fehler sinkt in allen 4 Zellen streng mit $\chi_1$ Richtung 1,03. SPreCap-e03 ist bei MNIST mit 100 Schichten signifikant besser als SELU (5,28 % gegen 6,36 %, Holm p = 0,0064). Das passt zur bSELU-Idee und stützt sie unabhängig.
4. **Der Zielkonflikt:** Höheres $\chi_1$ verträgt höhere Lernraten, liefert aber ein schlechteres Optimum.
5. **Kernel:** CUDA (Jiterator) und MLX (Metal), die so schnell sind wie natives SELU.

## Was ich nicht behaupten darf

- "Erste beschränkte selbstnormierende Aktivierung": nein (bSNN, SaluNet).
- "Neue Idee, $\chi_1$ leicht über 1 zu legen": nein (bSELU).
- "Besser als SELU allgemein": nein. Nur eine Zelle ist signifikant besser, und nur für e03.
- "Robuster gegenüber der Lernrate": nein (3a-3: bei LR 5e-4 lernen SELU/ELU, PreCap nicht).
- Der Vorteil durch begrenzten Wertebereich für fp16/fp8/int8 ist plausibel ([Studie "FP8 versus INT8 for efficient deep learning inference"](https://arxiv.org/html/2303.17951v2): der Hauptunterschied der Formate liegt im Umgang mit Ausreißern), aber **nicht gemessen**.

## Pflicht-Zitate im Paper

SELU, ELU ([arXiv:1511.07289](https://arxiv.org/pdf/1511.07289v3)), bSELU, bSNN, SaluNet, DyT, Derf, TanhExp, Hayou et al., Bukva et al., Kunc & Kléma.

## Name

| Prüfung | Ergebnis |
|---|---|
| PyPI `precap`, `precap-activation`, `pre-cap`, `spreccap` | alle frei (HTTP 404, geprüft am 26.10.2026) |
| GitHub-Repositorys "precap" | nur fachfremde Projekte (Uni-Kurse "Precapstone", Captcha-Bot) |
| ML-Literatur | "PreCaps" ist der Name einer Schicht in RobustCaps, einem Capsule-Netz ([arXiv:2210.11092](https://arxiv.org/pdf/2210.11092v1)). Das ist keine Aktivierungsfunktion, eine Verwechslung ist unwahrscheinlich. |

Empfehlung: Den Namen PreCap behalten und im Paper immer "PreCap activation" schreiben. Das Paket heißt `precap`.

## Folgen für die Weiterentwicklung

- e03 als Tiefen-Variante weiterführen. Als Nächstes wäre eine Familie mit $\chi_1=1+k/L$ sinnvoll, die von der Tiefe abhängt, wie bei bSELU. Das wäre ein direkter Test, ob die beschränkte Form das bSELU-Prinzip übernimmt.
- 3a-4 (kritische Init) bleibt sinnvoll, Bukva et al. sagen aber voraus, dass Sättigung trotz kritischer Init bremsen kann.
- Messung des Quantisierungsvorteils (int8/fp8 nach dem Training) wäre der einfachste Weg zu einem echten Alleinstellungsmerkmal.
