# PreCap / SPreCap – beschränkte, selbstnormierende Aktivierungsfunktion

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23676432.svg)](https://doi.org/10.5281/zenodo.23676432)

Forschungsprojekt von Aaron Schmitt (ORCID [0009-0009-8689-5848](https://orcid.org/0009-0009-8689-5848)), 14.–26.10.2026. Lizenz: Apache-2.0.

English summary: [see below](#english-summary).

## Formel

$$
y(x)=\lambda\cdot\begin{cases}1 & x>0\cr  \alpha & x\le 0\end{cases}\cdot\mathrm{capexp}\big(s\tanh(bx/s)\big),\qquad
\mathrm{capexp}(w)=\begin{cases}e^w-1 & w\le c\cr  e^c-1+e^c\thinspace (w-c) & w>c\end{cases}
$$

mit $s=1{,}5$ und $c=1$. CapExp allein ist exakt ein verschobenes ELU. Erst die weiche tanh-Begrenzung davor macht daraus eine eigene Funktion: glatt, monoton und beidseitig beschränkt. Die Konstanten $\lambda,\alpha$ setzen wie bei SELU Mittelwert 0 und Varianz 1 als Fixpunkt. $b$ stellt den Gradientenfaktor $\chi_1=\mathbb E[f'(z)^2]$ ein.

| Preset | λ | α | b | χ₁ | Wertebereich | Einsatz |
|---|---|---|---|---|---|---|
| `spreccap_cs` (Standard) | 1,102702500 | 2,053776267 | 0,71188112 | 1,0716 (wie SELU) | [−1,76; 3,39] | Hauptvariante v0.1 |
| `spreccap_e03` | 1,985552225 | 1,608136397 | 0,4118896262 | 1,0300 | [−2,48; 6,11] | sehr tiefe Netze (100 Schichten) |
| `spreccap` | 0,80676869644 | 2,39695130922 | 1 | 1,1305 | [−1,50; 2,48] | historisch, schlechter |
| `precap0` | 1 | 1 | 1 | – | [−0,78; 3,08] | Grundform, nicht selbstnormierend |

## Ergebnisse (ehrlich)

Phase 3a-3: MLP mit Breite 256, Tiefe 50 und 100, **ohne Normalisierung und ohne Residual**, Adam, eigenes Lernraten-Tuning je Funktion, 15 Bestätigungs-Seeds, präregistriert. Medianer Testfehler:

| Zelle | SPreCap-cs | SPreCap-e03 | SELU | ELU |
|---|---|---|---|---|
| FashionMNIST, 50 | 14,92 % | 14,57 % | 14,69 % | 14,67 % |
| FashionMNIST, 100 | 16,98 % | 16,61 % | 17,29 % | 16,62 % |
| MNIST, 50 | 4,42 % | 4,05 % | 4,31 % | 3,75 % |
| MNIST, 100 | 6,27 % | **5,28 %** | 6,36 % | 5,78 % |

- **SPreCap-cs ist gleichauf mit SELU,** in keiner Zelle signifikant schlechter (Differenz −0,12 bis +0,20 Prozentpunkte). Damit ist die Entscheidungsregel für v0.1 erfüllt. Besser als SELU ist cs nicht.
- **SPreCap-e03 ist bei MNIST mit 100 Schichten signifikant besser als SELU** (Holm p = 0,0064). Das ist ein Sekundärtest und nur eine von vier Zellen.
- **ELU ist genauso stark:** Mittlerer Rang e03 1,5, ELU 1,75, SPreCap-cs und SELU je 4,25. Einen klaren Vorsprung vor ELU hat keine PreCap-Variante.
- **Je näher χ₁ an 1, desto besser:** In allen 4 Zellen sinkt der Fehler streng mit χ₁. Dieselbe Idee steckt in bSELU (Chen et al. 2024). Unsere Daten stützen sie unabhängig für eine beschränkte Funktion.
- **Nicht robuster gegenüber der Lernrate:** Bei LR 5e-4 und Tiefe 50 lernen SELU und ELU, die PreCap-Varianten nicht.
- **Frühere Vorteile in 1D/2D hingen an fester Lernrate** und verschwanden mit fairem Tuning (V18). Das ist dokumentiert.

Paper: `paper/PreCap_paper.pdf`. Details: `analysis/P3A3_Auswertung.md`, Einordnung: `docs/Recherche_Neuheit.md`, alles zusammen: `docs/PROJEKT_GESAMTDOKUMENTATION.md`.

## Installation und Nutzung

```bash
pip install "precap[torch] @ git+https://github.com/hermi-labs/precap-activation"   # PyTorch (CPU oder CUDA)
pip install "precap[mlx] @ git+https://github.com/hermi-labs/precap-activation"     # Apple Silicon
```

```python
import torch
from precap import PreCap
net = torch.nn.Sequential(torch.nn.Linear(784, 256), PreCap('spreccap_cs'), torch.nn.Linear(256, 10))
# Auf CUDA-Tensoren wird automatisch der fusionierte Kernel genutzt, sonst die Referenz.
# PreCap(backend='ref') erzwingt die Referenz.
```

```python
from precap.mlx import PreCap   # Metal-Kernel, rechnet intern in float32
act = PreCap('spreccap_e03')
```

Für die Selbstnormierung braucht das Netz wie bei SELU eine Init $W\sim N(0,1/\mathrm{fan}_{\mathrm{in}})$ und standardisierte Eingaben.

Es gibt keine festen Pfade. Der Kernel-Code ist im Paket enthalten, kein CUDA-Toolkit und kein Xcode nötig.

## Kernel

| Gerät | PreCap-Kernel | Referenz (einzelne Operationen) | natives SELU |
|---|---|---|---|
| RTX 4070, CUDA 12.8, torch 2.11 | 0,909 ms | 10,46 ms | 0,89 ms |
| M1 Pro, MLX 0.32.4 | 4,15 ms | 34,1 ms | 11,47 ms |

Gemessen: vorwärts und rückwärts über 16M Elemente in float32. Die MLX-Messung lief parallel zu einem Trainingslauf und ist daher eher zu langsam. Tests: 17/17 (CUDA), 12/12 (MLX), Abweichung zur Referenz in fp32 höchstens 1,1e-6.

## Reproduzieren

- Jede Studie hat eine Präregistrierung (`docs/`), eingefrorene Skripte mit SHA-256 (`code/**/meta/frozen_sha256.txt`), Rohdaten oder deren Hashes und eine Auswertung (`analysis/`).
- **Pfade in den eingefrorenen Skripten:** Die Skripte enthalten feste Mac-Pfade wie `DATA=Path('/Users/aaronschmitt/PyCharmMiscProject/data')` (bei `run_p3a3.py` in Zeile 10). Sie dürfen nicht verändert werden, sonst stimmen die Hashes nicht. Zum Nachrechnen eine Kopie anlegen und nur diese Zeile auf den eigenen MNIST-Ordner setzen.
- Paket-Tests: `pip install -e ".[torch,test]" && pytest tests`. Kernel-Tests: `kernels/README.md`.

## Struktur

| Ordner | Inhalt |
|---|---|
| `paper/` | Preprint als PDF (`PreCap_paper.pdf`), Grafiken und Quellskripte |
| `src/precap/` | installierbares Paket (PyTorch + CUDA-Kernel, MLX + Metal-Kernel) |
| `kernels/` | eigenständige Kernel mit Tests und Messergebnissen |
| `code/` | alle Studien 1D, 2D, 3D, Phase 3 mit Hashes |
| `analysis/` | Auswertungen |
| `docs/` | Präregistrierungen, Gesamtdoku, Fahrplan, Recherche, [Danksagung](docs/Danksagung_und_Werkzeuge.md) |
| `math/` | Herleitungen der Konstanten und Mean-Field-Rechnungen |
| `literatur/` | Literaturrecherche |
| `notizen/` | handschriftliche Notizen |

## Zitieren

Siehe `CITATION.cff`. Die DOI folgt mit dem ersten Release über Zenodo.

---

## English summary

**PreCap** is a smooth, monotonic, bounded activation: an exponential-linear unit (CapExp, exactly a shifted ELU) applied after a soft tanh clip, $y=\lambda\thinspace [x>0\thinspace ?\thinspace 1:\alpha]\cdot\mathrm{capexp}(s\tanh(bx/s))$ with $s=1.5,\ c=1$. **SPreCap** chooses $\lambda,\alpha$ for the SELU fixed point (mean 0, variance 1) and $b$ to set the gradient gain $\chi_1$.

In a preregistered study on 50- and 100-layer MLPs without normalization or residuals (MNIST, FashionMNIST, per-activation LR tuning, 15 seeds), **SPreCap-cs matches SELU** (no cell significantly worse, differences −0.12 to +0.20 pp). It is not better overall. The variant **SPreCap-e03 ($\chi_1=1.03$) beats SELU on MNIST at depth 100** (5.28 % vs 6.36 %, Holm p = 0.0064, secondary test). Plain ELU is about as strong (mean rank e03 1.5, ELU 1.75). The test error decreases monotonically as $\chi_1\to1$, consistent with bSELU (Chen et al. 2024). PreCap is **not** more robust to the learning rate. Earlier advantages at a fixed learning rate disappeared under fair tuning. Negative results are documented.

Fused kernels: CUDA via `torch.cuda.jiterator` (RTX 4070, 16M elements fwd+bwd: 0.91–1.06 ms vs 0.73–0.89 ms for native SELU, i.e. up to about 1.5×; fp16/bf16 work but go through fp32) and MLX/Metal (4.15 ms vs MLX SELU 11.47 ms on an M1 Pro). Install with `pip install "precap[torch] @ git+https://github.com/hermi-labs/precap-activation"` and use `from precap import PreCap`.
