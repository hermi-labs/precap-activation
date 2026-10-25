# v12 — Auswertung Plain-Mechanismus

Integrität: 1620/1620 Runs, bad=True: 0. d = log10(MSE_PreCap / MSE_Referenz); d<0 heißt PreCap besser. Klammer: Seed-Paare mit PreCap besser (Tiefen 4/8/16 × 10 Seeds = 30).

## Architektur plain: Median d je Aufgabe und Regime (gepoolt über Tiefen und Seeds)

| Aufgabe | Regime | vs SiLU | vs GELU |
|---|---|---|---|
| sin3_quad | base | +0.39 (8/30) | +0.59 (7/30) |
| sin3_quad | high_lr | +0.62 (4/30) | +0.61 (4/30) |
| sin3_quad | large_init | +1.44 (1/30) | +1.07 (0/30) |
| sin_mix | base | +0.12 (10/30) | +0.02 (13/30) |
| sin_mix | high_lr | +0.06 (11/30) | +0.14 (13/30) |
| sin_mix | large_init | +0.27 (3/30) | +0.19 (5/30) |
| abs_wave | base | -0.27 (23/30) | -0.08 (16/30) |
| abs_wave | high_lr | +0.31 (9/30) | +0.42 (8/30) |
| abs_wave | large_init | +0.25 (7/30) | +0.60 (5/30) |

## Architektur res_inv: Median d je Aufgabe und Regime (gepoolt über Tiefen und Seeds)

| Aufgabe | Regime | vs SiLU | vs GELU |
|---|---|---|---|
| sin3_quad | base | -0.76 (26/30) | -1.06 (29/30) |
| sin3_quad | high_lr | -0.13 (18/30) | -0.40 (23/30) |
| sin3_quad | large_init | -1.12 (25/30) | -1.40 (30/30) |
| sin_mix | base | -0.10 (24/30) | +0.01 (14/30) |
| sin_mix | high_lr | -0.23 (26/30) | -0.17 (27/30) |
| sin_mix | large_init | -0.32 (28/30) | -0.46 (28/30) |
| abs_wave | base | -0.84 (30/30) | -2.05 (30/30) |
| abs_wave | high_lr | -0.69 (25/30) | -0.99 (26/30) |
| abs_wave | large_init | -1.00 (27/30) | -1.92 (28/30) |

## Regel aus dem Manifest (Plain)

| Regime | Aufgaben mit d<0 gegen SiLU und GELU |
|---|---|
| base | 1/3 |
| high_lr | 0/3 |
| large_init | 0/3 |

Anzahl Plain-Regime mit Erholung (>=2/3 Aufgaben): 0/3. res_inv: 8/9 Aufgabe-Regime-Zellen mit PreCap besser gegen beide Referenzen.

## Plain nach Tiefe (Median d, alle Aufgaben und Regime)

| Tiefe | vs SiLU | vs GELU |
|---:|---|---|
| 4 | +0.13 (29/90) | +0.23 (32/90) |
| 8 | +0.29 (19/90) | +0.38 (20/90) |
| 16 | +0.42 (28/90) | +0.53 (19/90) |

## Stabilität: Median-Verhältnis PreCap / Referenz (Plain und res_inv)

| Arch | Referenz | hmax_peak | grad_final |
|---|---|---:|---:|
| plain | silu | 0.36 | 4.60 |
| plain | gelu | 0.51 | 3.36 |
| res_inv | silu | 0.50 | 0.12 |
| res_inv | gelu | 0.59 | 0.03 |

Median PreCap-Cap-Anteil am Ende: plain 0.067, res_inv 0.150.

## Seed-Ebene Vorzeichentest (Median über Tiefen je Aufgabe, Regime, Seed)

| Arch | Referenz | Einheiten PreCap besser | p (einseitig PreCap besser) |
|---|---|---|---:|
| plain | silu | 17/90 | 1 |
| plain | gelu | 15/90 | 1 |
| res_inv | silu | 81/90 | 6.4e-16 |
| res_inv | gelu | 83/90 | 6.58e-18 |

## Befund

- Optimierungsempfindlichkeit gestützt: False.
- PreCap in plain unter allen 9 Aufgabe-Regime-Zellen nicht besser gegen beide Referenzen: False.