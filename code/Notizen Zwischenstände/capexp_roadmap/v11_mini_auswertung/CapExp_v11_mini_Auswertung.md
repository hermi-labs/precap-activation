# v11 mini — Auswertung

Integrität: 918/918 Zeilen, 918 eindeutig, 0 Duplikate, bad=True: 0, alle test_loss endlich und positiv: True.

Metrik: d = log10(MSE_Variante / MSE_Vergleich), gepaart je Aufgabe, Tiefe, Architektur, init_k, lr, Seed. d<0 heißt Variante besser. Toleranz tief: Faktor 1,5, also d <= 0.176.

## Gate-B-Prüfung gegen PreCap

| Variante | flach: Zellen besser | flach: Median d | flach: Win-Rate | tief: Zellen besser | tief: Median d | tief: schlechtester Zellmedian d | tief erhalten | flach verbessert | tief verbessert | Gate B |
|---|---|---:|---:|---|---:|---:|---|---|---|---|
| precap_wg_k0.5 | 0/12 | +0.232 | 24% | 2/6 | +0.296 | +0.800 | False | False | False | FAIL |
| precap_wg_k0.75 | 1/12 | +0.325 | 24% | 2/6 | +0.273 | +0.480 | False | False | False | FAIL |
| hybrid_k0.5_b0.2 | 9/12 | -0.026 | 60% | 2/6 | +0.007 | +0.798 | False | True | False | FAIL |
| hybrid_k0.75_b0.3 | 7/12 | -0.016 | 54% | 3/6 | +0.026 | +0.221 | False | True | False | FAIL |

## Flache Netze gegen SiLU und GELU (Median d, Win-Rate)

| Aktivierung | vs SiLU | vs GELU |
|---|---|---|
| precap | -0.083 (53%) | -0.129 (56%) |
| precap_wg_k0.5 | +0.096 (47%) | -0.025 (52%) |
| precap_wg_k0.75 | +0.209 (46%) | -0.038 (51%) |
| hybrid_k0.5_b0.2 | -0.189 (58%) | -0.112 (56%) |
| hybrid_k0.75_b0.3 | -0.170 (55%) | -0.145 (55%) |

## Zellen: Median d gegen PreCap (Seed-Paare besser)

| Aufgabe | Tiefe | Arch | precap_wg_k0.5 | precap_wg_k0.75 | hybrid_k0.5_b0.2 | hybrid_k0.75_b0.3 |
|---|---:|---|---|---|---|---|
| abs_wave | 8 | plain | +0.43 (2/9) | +0.49 (2/9) | -0.03 (5/9) | +0.05 (4/9) |
| abs_wave | 8 | res_inv | +0.42 (2/9) | +0.42 (2/9) | -0.08 (7/9) | -0.01 (5/9) |
| abs_wave | 16 | plain | +0.17 (3/9) | +0.33 (1/9) | -0.06 (5/9) | +0.13 (4/9) |
| abs_wave | 16 | res_inv | +0.54 (1/9) | +0.62 (1/9) | -0.05 (6/9) | -0.09 (5/9) |
| abs_wave | 32 | res_inv | +0.74 (1/3) | +0.48 (0/3) | +0.20 (1/3) | +0.09 (1/3) |
| abs_wave | 64 | res_inv | +0.80 (0/3) | +0.48 (1/3) | +0.69 (1/3) | +0.08 (1/3) |
| sin3_quad | 8 | plain | +0.35 (1/9) | +0.66 (2/9) | +0.08 (3/9) | +0.31 (2/9) |
| sin3_quad | 8 | res_inv | +0.11 (3/9) | +0.14 (4/9) | -0.11 (6/9) | +0.06 (4/9) |
| sin3_quad | 16 | plain | +0.07 (2/9) | +0.37 (0/9) | -0.06 (7/9) | -0.04 (5/9) |
| sin3_quad | 16 | res_inv | +0.27 (2/9) | +0.35 (2/9) | -0.26 (7/9) | +0.08 (3/9) |
| sin3_quad | 32 | res_inv | -0.21 (2/3) | -0.31 (2/3) | -0.55 (2/3) | -0.42 (2/3) |
| sin3_quad | 64 | res_inv | +0.40 (1/3) | +0.11 (1/3) | +0.80 (1/3) | +0.22 (1/3) |
| sin_mix | 8 | plain | +0.16 (2/9) | -0.06 (5/9) | -0.00 (5/9) | -0.07 (8/9) |
| sin_mix | 8 | res_inv | +0.14 (3/9) | +0.06 (4/9) | -0.03 (7/9) | -0.04 (7/9) |
| sin_mix | 16 | plain | +0.10 (4/9) | +0.31 (1/9) | +0.05 (3/9) | -0.03 (5/9) |
| sin_mix | 16 | res_inv | +0.23 (1/9) | +0.37 (2/9) | +0.00 (4/9) | -0.04 (6/9) |
| sin_mix | 32 | res_inv | -0.11 (2/3) | -0.14 (3/3) | -0.11 (3/3) | -0.18 (2/3) |
| sin_mix | 64 | res_inv | +0.22 (0/3) | +0.39 (0/3) | +0.01 (0/3) | -0.01 (2/3) |

## Sekundär: Best-of je Seed (gleiches Budget: je Variante und PreCap 3 Initialisierung-/Lernraten-Paare)

| Variante | Median d vs PreCap | Win-Rate | n |
|---|---:|---:|---:|
| precap_wg_k0.5 | +0.297 | 24% | 54 |
| precap_wg_k0.75 | +0.335 | 22% | 54 |
| hybrid_k0.5_b0.2 | -0.015 | 54% | 54 |
| hybrid_k0.75_b0.3 | +0.063 | 46% | 54 |