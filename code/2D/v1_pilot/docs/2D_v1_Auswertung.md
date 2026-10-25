# CapExp 2D v1 — Pilotauswertung

Integrität: 135/135 Runs, bad=True: 0. Die Hauptgröße ist d = log10(MSE_PreCap/MSE_Referenz); d<0 bedeutet PreCap besser. Klammern zeigen PreCap-Siege pro 5 gepaarten Seeds.

## Ergebnis nach Aufgabe und Tiefe

| Aufgabe | Tiefe | vs SiLU | vs GELU |
|---|---:|---|---|
| radial_ripple | 4 | +1.135 (0/5) | +1.109 (0/5) |
| radial_ripple | 8 | +1.451 (0/5) | +1.491 (0/5) |
| radial_ripple | 16 | +1.558 (0/5) | +0.685 (0/5) |
| ridge_mix | 4 | +0.768 (0/5) | +1.142 (1/5) |
| ridge_mix | 8 | +1.148 (0/5) | +1.054 (0/5) |
| ridge_mix | 16 | +1.576 (1/5) | +1.631 (0/5) |
| corner_kink | 4 | +0.552 (0/5) | +0.812 (0/5) |
| corner_kink | 8 | +0.643 (0/5) | +0.945 (0/5) |
| corner_kink | 16 | +0.634 (0/5) | +0.938 (0/5) |

## Zusammenfassung je Aufgabe (über Tiefen und Seeds)

| Aufgabe | vs SiLU | vs GELU | PASS gegen beide? |
|---|---|---|---|
| radial_ripple | +1.340 (0/15) | +1.202 (0/15) | False |
| ridge_mix | +1.101 (1/15) | +1.172 (1/15) | False |
| corner_kink | +0.595 (0/15) | +0.930 (0/15) | False |

## Zusammenfassung je Tiefe (über Aufgaben und Seeds)

| Tiefe | vs SiLU | vs GELU |
|---:|---|---|
| 4 | +0.811 (0/15) | +1.084 (1/15) |
| 8 | +1.101 (0/15) | +1.054 (0/15) |
| 16 | +0.885 (1/15) | +0.962 (0/15) |

## Stabilität

| Referenz | hmax_peak PreCap/Ref | grad_final PreCap/Ref |
|---|---:|---:|
| silu | 1.565 | 4.392 |
| gelu | 1.922 | 6.437 |

Median finaler Cap-Anteil: 0.443.

## Gepaarte Seed-Vorzeichentests

| Referenz | PreCap besser | p, einseitig |
|---|---:|---:|
| silu | 1/45 | 1 |
| gelu | 1/45 | 1 |

## Pilotentscheidung

- Aufgaben mit negativem Median d gegen beide Referenzen: 0/3.
- Bad-Rate: 0/135.
- **PASS nach der vorab festgelegten Leistungsregel: False.**

Interpretation: Dies ist ein kleiner Pilot mit fünf Seeds und drei Tiefen. Ein PASS rechtfertigt einen unabhängigen 2D-Holdout, nicht die Behauptung universeller Überlegenheit.