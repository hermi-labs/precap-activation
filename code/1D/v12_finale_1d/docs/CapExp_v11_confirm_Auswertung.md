# v11 confirm — Auswertung nach Manifest

Integrität: 2400/2400 Runs, bad=True: 1.
- bad: abs_wave hybrid_gate Tiefe 64 res_inv init_k=0.5 lr=0.001 seed=102 test_loss=inf

## Regeln gegen PreCap

| Kriterium | Wert | Schwelle | erfüllt |
|---|---|---|---|
| 1 tief: schlechtester Zellmedian d | +0.359 (Faktor 2.28) | <= 0.176 | False |
| 2 flach: Zellen mit Median d<0 | 6/18 | >= 11 | False |
| 2 flach: gepoolter Median d / Win-Rate | +0.0214 / 46.1% (n=540) | <0 / >50% | False |
| 3 Stabilität: bad-Runs des Kandidaten | 1 | 0 | False |
| 3 Stabilität: Median hmax_peak-Verhältnis | 1.00 | <= 2 | True |
| Alternative: tief Zellen d<0 / Win-Rate | 3/6 / 51.7% (n=60) | >=4 / >=66,7% | False |

**Gesamt: NICHT BESTANDEN**

## Median d pro Tiefe (alle Zellen gepoolt)

| Tiefe | vs PreCap | vs SiLU | vs GELU |
|---:|---|---|---|
| 4 | +0.003 (47%, n=180) | -0.162 (57%, n=180) | -0.197 (59%, n=180) |
| 8 | +0.016 (48%, n=180) | +0.083 (47%, n=180) | -0.121 (57%, n=180) |
| 16 | +0.053 (43%, n=180) | -0.014 (51%, n=180) | -0.045 (53%, n=180) |
| 32 | -0.049 (63%, n=30) | -0.342 (70%, n=30) | -0.474 (73%, n=30) |
| 64 | +0.039 (40%, n=30) | -0.240 (67%, n=30) | -0.388 (77%, n=30) |

## Zellen: Median d (Seed-Paare besser)

| Aufgabe | Tiefe | Arch | vs PreCap | vs SiLU | vs GELU |
|---|---:|---|---|---|---|
| abs_wave | 4 | plain | -0.12 (17/30) | -0.22 (18/30) | -0.23 (19/30) |
| abs_wave | 4 | res_inv | +0.03 (12/30) | -1.44 (30/30) | -1.96 (30/30) |
| abs_wave | 8 | plain | +0.03 (13/30) | +0.54 (7/30) | +0.56 (10/30) |
| abs_wave | 8 | res_inv | +0.06 (13/30) | -0.59 (26/30) | -1.66 (29/30) |
| abs_wave | 16 | plain | -0.10 (20/30) | +0.36 (11/30) | +0.55 (5/30) |
| abs_wave | 16 | res_inv | +0.20 (10/30) | -0.55 (25/30) | -1.28 (29/30) |
| abs_wave | 32 | res_inv | -0.05 (7/10) | -0.62 (8/10) | -1.05 (8/10) |
| abs_wave | 64 | res_inv | +0.12 (4/10) | -0.30 (8/10) | -0.74 (8/10) |
| sin3_quad | 4 | plain | +0.24 (11/30) | +1.21 (1/30) | +0.82 (4/30) |
| sin3_quad | 4 | res_inv | +0.12 (13/30) | -0.58 (24/30) | -0.95 (30/30) |
| sin3_quad | 8 | plain | +0.10 (11/30) | +1.55 (0/30) | +1.10 (0/30) |
| sin3_quad | 8 | res_inv | +0.12 (9/30) | -0.70 (20/30) | -1.10 (28/30) |
| sin3_quad | 16 | plain | +0.11 (9/30) | +1.02 (3/30) | +1.09 (1/30) |
| sin3_quad | 16 | res_inv | +0.16 (8/30) | -0.03 (16/30) | -0.69 (23/30) |
| sin3_quad | 32 | res_inv | +0.36 (3/10) | +0.07 (4/10) | +0.03 (5/10) |
| sin3_quad | 64 | res_inv | +0.24 (3/10) | +0.11 (4/10) | -0.25 (6/10) |
| sin_mix | 4 | plain | -0.02 (19/30) | +0.17 (8/30) | +0.15 (6/30) |
| sin_mix | 4 | res_inv | +0.00 (13/30) | -0.11 (22/30) | -0.06 (17/30) |
| sin_mix | 8 | plain | -0.01 (16/30) | +0.37 (3/30) | +0.22 (9/30) |
| sin_mix | 8 | res_inv | -0.05 (24/30) | -0.28 (28/30) | -0.28 (27/30) |
| sin_mix | 16 | plain | -0.02 (16/30) | +0.64 (9/30) | +0.28 (11/30) |
| sin_mix | 16 | res_inv | +0.00 (15/30) | -0.38 (28/30) | -0.33 (26/30) |
| sin_mix | 32 | res_inv | -0.06 (9/10) | -0.35 (9/10) | -0.33 (9/10) |
| sin_mix | 64 | res_inv | -0.01 (5/10) | -0.25 (8/10) | -0.25 (9/10) |

## Seed-Ebene (Median über Paare je Zelle und Seed), Vorzeichentest einseitig

| Bereich | Vergleich | Seed-Einheiten besser | p |
|---|---|---|---:|
| flach | precap | 82/180 | 0.898 |
| flach | silu | 93/180 | 0.355 |
| flach | gelu | 98/180 | 0.132 |
| tief | precap | 31/60 | 0.449 |
| tief | silu | 41/60 | 0.00311 |
| tief | gelu | 45/60 | 6.73e-05 |

## Sensitivität ohne bad-Runs

Flach ohne betroffene Paare: Median d +0.0214, Win-Rate 46.1%; tief: Median d -0.023, Win-Rate 52.5%.