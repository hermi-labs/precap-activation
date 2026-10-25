# v9b: sin3_quad Zwischenauswertung

Runs: 4200. Overflow-Ausfaelle: 904. Konstante-Ausgabe-Kollaps: 769 (Referenz-Loss: 0.655996).

## Beste Konfiguration je Familie

Auswahl auf Seeds 0-4, unabhaengige Bewertung auf Seeds 5-9.

### Tiefe 16

| Familie | Holdout-Median | 95%-CI | Tote Holdout-Runs | Einstellung | Tote Runs, alle Seeds |
|---|---:|---:|---:|---|---:|
| GELU plain | 3.24e-05 | 2.56e-05 - 5.99e-05 | 0/5 | lr=0.0003, k=1.0 | 0% |
| CapExp res | 3.57e-05 | 8.13e-06 - 2.36e-04 | 0/5 | lr=0.001, k=0.5, c=1.5, a=1.0, gamma=0.0 | 0% |
| SiLU plain | 3.61e-05 | 8.58e-06 - 9.37e-05 | 0/5 | lr=0.0003, k=1.0 | 0% |
| PreCap res | 4.04e-05 | 1.81e-05 - 1.69e-03 | 0/5 | lr=0.0003, k=0.5, c=1.0, a=1.0, gamma=0.01 | 0% |
| PreCap plain | 4.44e-05 | 3.24e-05 - 1.71e-04 | 0/5 | lr=0.0003, k=0.5, c=1.0, a=1.0, gamma=0.01 | 0% |
| CapExp plain | 8.45e-05 | 3.53e-05 - 1.39e-04 | 0/5 | lr=0.0003, k=0.5, c=0.75, a=1.0, gamma=0.0 | 0% |
| GELU res | 5.43e-04 | 1.61e-04 - 1.04e-03 | 0/5 | lr=0.003, k=0.5 | 0% |
| ReLU plain | 6.08e-04 | 4.44e-04 - 9.86e-04 | 0/5 | lr=0.0003, k=0.5 | 0% |
| SiLU res | 8.92e-04 | 3.42e-05 - 1.23e-02 | 0/5 | lr=0.003, k=1.0 | 0% |
| ReLU res | 9.19e-04 | 4.97e-04 - 9.15e-03 | 0/5 | lr=0.0003, k=0.5 | 0% |

### Tiefe 32

| Familie | Holdout-Median | 95%-CI | Tote Holdout-Runs | Einstellung | Tote Runs, alle Seeds |
|---|---:|---:|---:|---|---:|
| PreCap res | 1.66e-05 | 9.96e-06 - 1.14e-04 | 0/5 | lr=0.001, k=1.0, c=1.0, a=1.0, gamma=0.01 | 0% |
| CapExp res | 2.37e-05 | 3.71e-06 - 6.45e-04 | 0/5 | lr=0.001, k=0.25, c=1.5, a=1.0, gamma=0.0 | 0% |
| SiLU plain | 4.46e-05 | 1.32e-05 - 1.57e-04 | 0/5 | lr=0.0003, k=1.0 | 0% |
| SiLU res | 5.47e-05 | 3.60e-05 - 4.13e-03 | 0/5 | lr=0.003, k=1.0 | 0% |
| GELU plain | 6.22e-05 | 8.48e-06 - 9.34e-05 | 0/5 | lr=0.0003, k=1.0 | 0% |
| CapExp plain | 1.37e-04 | 3.17e-05 - 4.02e-04 | 0/5 | lr=0.0003, k=0.5, c=0.75, a=1.0, gamma=0.0 | 0% |
| PreCap plain | 2.57e-04 | 1.14e-04 - 1.14e-03 | 0/5 | lr=0.0003, k=0.5, c=1.0, a=1.25, gamma=0.01 | 0% |
| GELU res | 4.09e-04 | 1.37e-04 - 1.40e-03 | 0/5 | lr=0.003, k=0.5 | 0% |
| ReLU plain | 1.15e-03 | 4.40e-04 - 2.35e-03 | 0/5 | lr=0.0003, k=0.5 | 0% |
| ReLU res | 1.40e-03 | 7.85e-04 - 2.71e-03 | 0/5 | lr=0.0003, k=0.5 | 0% |

### Tiefe 64

| Familie | Holdout-Median | 95%-CI | Tote Holdout-Runs | Einstellung | Tote Runs, alle Seeds |
|---|---:|---:|---:|---|---:|
| PreCap res | 1.34e-05 | 1.12e-05 - 1.88e-03 | 0/5 | lr=0.001, k=0.5, c=1.0, a=1.0, gamma=0.01 | 0% |
| CapExp res | 2.31e-05 | 1.30e-05 - 2.92e-05 | 0/5 | lr=0.001, k=0.25, c=1.5, a=1.0, gamma=0.0 | 0% |
| GELU res | 1.46e-04 | 1.04e-04 - 3.80e-04 | 0/5 | lr=0.003, k=0.5 | 0% |
| SiLU res | 2.21e-04 | 3.50e-05 - 2.99e-04 | 0/5 | lr=0.003, k=0.5 | 0% |
| ReLU plain | 1.41e-03 | 1.30e-03 - 2.70e-03 | 0/5 | lr=0.0003, k=1.0 | 0% |
| ReLU res | 2.01e-03 | 1.40e-03 - 2.91e-03 | 0/5 | lr=0.003, k=0.5 | 0% |
| SiLU plain | dead | dead - dead | 5/5 | lr=0.0003, k=0.5 | 100% |
| GELU plain | dead | dead - dead | 5/5 | lr=0.0003, k=0.5 | 100% |
| CapExp plain | dead | dead - dead | 5/5 | lr=0.0001, k=0.25, c=0.75, a=1.0, gamma=0.0 | 100% |
| PreCap plain | dead | 5.17e-05 - dead | 3/5 | lr=0.0003, k=0.5, c=1.0, a=1.25, gamma=0.01 | 40% |

## Stabiler Betriebsbereich

| Tiefe | Familie | Configs | Stabile Configs | Gute Configs | Tote Runs | Kollaps | Overflow |
|---|---|---:|---:|---:|---:|---:|---:|
| 16 | SiLU plain | 6 | 100% | 33% | 0% | 0 | 0 |
| 16 | PreCap res | 16 | 100% | 25% | 0% | 0 | 0 |
| 16 | GELU plain | 6 | 100% | 17% | 0% | 0 | 0 |
| 16 | CapExp res | 36 | 67% | 17% | 28% | 0 | 99 |
| 16 | SiLU res | 6 | 100% | 0% | 0% | 0 | 0 |
| 16 | GELU res | 6 | 100% | 0% | 0% | 0 | 0 |
| 16 | ReLU plain | 6 | 100% | 0% | 0% | 0 | 0 |
| 16 | ReLU res | 6 | 100% | 0% | 0% | 0 | 0 |
| 16 | CapExp plain | 36 | 28% | 0% | 50% | 0 | 181 |
| 16 | PreCap plain | 16 | 94% | 0% | 1% | 2 | 0 |
| 32 | PreCap res | 16 | 100% | 50% | 0% | 0 | 0 |
| 32 | SiLU plain | 6 | 50% | 17% | 50% | 30 | 0 |
| 32 | CapExp res | 36 | 56% | 11% | 37% | 0 | 134 |
| 32 | SiLU res | 6 | 100% | 0% | 0% | 0 | 0 |
| 32 | GELU plain | 6 | 50% | 0% | 50% | 30 | 0 |
| 32 | GELU res | 6 | 100% | 0% | 0% | 0 | 0 |
| 32 | ReLU plain | 6 | 67% | 0% | 32% | 19 | 0 |
| 32 | ReLU res | 6 | 100% | 0% | 0% | 0 | 0 |
| 32 | CapExp plain | 36 | 8% | 0% | 85% | 130 | 177 |
| 32 | PreCap plain | 16 | 44% | 0% | 28% | 44 | 0 |
| 64 | PreCap res | 16 | 100% | 56% | 0% | 0 | 0 |
| 64 | CapExp res | 36 | 39% | 19% | 53% | 0 | 190 |
| 64 | SiLU res | 6 | 100% | 17% | 0% | 0 | 0 |
| 64 | SiLU plain | 6 | 0% | 0% | 100% | 60 | 0 |
| 64 | GELU plain | 6 | 0% | 0% | 100% | 60 | 0 |
| 64 | GELU res | 6 | 100% | 0% | 0% | 0 | 0 |
| 64 | ReLU plain | 6 | 17% | 0% | 72% | 43 | 0 |
| 64 | ReLU res | 6 | 100% | 0% | 0% | 0 | 0 |
| 64 | CapExp plain | 36 | 0% | 0% | 98% | 231 | 123 |
| 64 | PreCap plain | 16 | 6% | 0% | 75% | 120 | 0 |

## Bestes PreCap gegen Baselines

Mann-Whitney auf Holdout-Seeds; p-Werte Holm-korrigiert.

| Tiefe | PreCap | Baseline | Median PreCap | Median Baseline | p Holm | PreCap besser |
|---|---|---|---:|---:|---:|---|
| 16 | PreCap res | SiLU plain | 4.04e-05 | 3.61e-05 | 1.000 | nein |
| 16 | PreCap res | SiLU res | 4.04e-05 | 8.92e-04 | 1.000 | ja |
| 16 | PreCap res | GELU plain | 4.04e-05 | 3.24e-05 | 1.000 | nein |
| 16 | PreCap res | GELU res | 4.04e-05 | 5.43e-04 | 1.000 | ja |
| 16 | PreCap res | ReLU plain | 4.04e-05 | 6.08e-04 | 1.000 | ja |
| 16 | PreCap res | ReLU res | 4.04e-05 | 9.19e-04 | 0.667 | ja |
| 32 | PreCap res | SiLU plain | 1.66e-05 | 4.46e-05 | 1.000 | ja |
| 32 | PreCap res | SiLU res | 1.66e-05 | 5.47e-05 | 1.000 | ja |
| 32 | PreCap res | GELU plain | 1.66e-05 | 6.22e-05 | 1.000 | ja |
| 32 | PreCap res | GELU res | 1.66e-05 | 4.09e-04 | 0.135 | ja |
| 32 | PreCap res | ReLU plain | 1.66e-05 | 1.15e-03 | 0.135 | ja |
| 32 | PreCap res | ReLU res | 1.66e-05 | 1.40e-03 | 0.135 | ja |
| 64 | PreCap res | SiLU plain | 1.34e-05 | dead | 0.135 | ja |
| 64 | PreCap res | SiLU res | 1.34e-05 | 2.21e-04 | 1.000 | ja |
| 64 | PreCap res | GELU plain | 1.34e-05 | dead | 0.135 | ja |
| 64 | PreCap res | GELU res | 1.34e-05 | 1.46e-04 | 1.000 | ja |
| 64 | PreCap res | ReLU plain | 1.34e-05 | 1.41e-03 | 0.667 | ja |
| 64 | PreCap res | ReLU res | 1.34e-05 | 2.01e-03 | 0.413 | ja |