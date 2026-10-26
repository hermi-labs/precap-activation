# CapExp 2D v1b — Auswertung der korrigierten PreCap-Formel

Integrität: 135/135 Runs, `bad=True`: 0. Diese Auswertung ist gültig, weil v1b die 1D-v9-PreCap-Formel auf 10.001 Werten exakt reproduziert hat.

Primärgröße: $d=\log_{10}(\mathrm{MSE}_{PreCap}/\mathrm{MSE}_{Ref})$. $d<0$ heißt: PreCap besser. Klammern: PreCap-Siege / gepaarte Seeds.

## Nach Aufgabe und Tiefe

| Aufgabe | Tiefe | gegen SiLU | gegen GELU |
|---|---:|---|---|
| radial_ripple | 4 | -0.894 (5/5) | -0.765 (5/5) |
| radial_ripple | 8 | -0.796 (5/5) | -1.032 (5/5) |
| radial_ripple | 16 | +0.174 (0/5) | -0.442 (4/5) |
| ridge_mix | 4 | -0.685 (5/5) | -0.415 (4/5) |
| ridge_mix | 8 | -0.426 (5/5) | -0.412 (5/5) |
| ridge_mix | 16 | -0.355 (3/5) | -0.201 (3/5) |
| corner_kink | 4 | -0.248 (5/5) | +0.064 (2/5) |
| corner_kink | 8 | -0.466 (5/5) | -0.012 (3/5) |
| corner_kink | 16 | -0.294 (4/5) | +0.035 (2/5) |

## Pilotentscheidung pro Aufgabe

| Aufgabe | gegen SiLU | gegen GELU | PASS gegen beide |
|---|---|---|---|
| radial_ripple | -0.674 (10/15) | -0.638 (14/15) | True |
| ridge_mix | -0.623 (13/15) | -0.405 (12/15) | True |
| corner_kink | -0.441 (14/15) | +0.013 (7/15) | False |

## Nach Tiefe (alle Aufgaben und Seeds)

| Tiefe | gegen SiLU | gegen GELU |
|---:|---|---|
| 4 | -0.719 (15/15) | -0.663 (11/15) |
| 8 | -0.502 (15/15) | -0.392 (13/15) |
| 16 | +0.005 (7/15) | -0.148 (9/15) |

## Stabilität

| Referenz | Median hmax_peak PreCap/Ref | Median grad_final PreCap/Ref |
|---|---:|---:|
| silu | 0.719 | 0.757 |
| gelu | 0.909 | 0.762 |

Median finaler Cap-Anteil (PreCap): 0.025.

## Gepaarte Seed-Vorzeichentests

| Referenz | PreCap besser | p, einseitig |
|---|---:|---:|
| silu | 37/45 | 7.69e-06 |
| gelu | 33/45 | 0.00123 |

## Fazit

- Aufgaben mit PreCap-Median besser gegen beide Referenzen: 2/3.
- Keine bad-Runs: True.
- **PASS nach vorab festgelegter Regel: True.**

Die Entscheidung wird nur aus diesem korrigierten v1b-Pilot abgeleitet. Der frühere v1-Lauf bleibt als Implementierungsfehler markiert und fließt nicht ein.