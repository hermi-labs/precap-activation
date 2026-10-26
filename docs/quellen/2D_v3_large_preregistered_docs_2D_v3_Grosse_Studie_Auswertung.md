# CapExp v3 — große 2D-Studie: Endauswertung

Integrität: 900/900 Runs, 300/300 vollständige Paare, bad=0.

Metrik: $d=\log_{10}(MSE_{PreCap}/MSE_{Ref})$; negativ begünstigt PreCap.

## Nach Aufgabe und Tiefe

| Aufgabe | Tiefe | vs SiLU | vs GELU |
|---|---:|---|---|
| radial_ripple | 4 | -0.857 (10/10) | -0.819 (10/10) |
| radial_ripple | 8 | -0.978 (9/10) | -0.963 (9/10) |
| radial_ripple | 16 | -0.314 (7/10) | -0.459 (9/10) |
| radial_ripple | 32 | -0.519 (7/10) | -0.265 (8/10) |
| radial_ripple | 64 | +0.213 (5/10) | +0.045 (4/10) |
| ridge_mix | 4 | -0.960 (10/10) | -0.722 (10/10) |
| ridge_mix | 8 | -0.642 (10/10) | -0.502 (10/10) |
| ridge_mix | 16 | -0.213 (6/10) | -0.323 (6/10) |
| ridge_mix | 32 | -0.380 (8/10) | -0.237 (7/10) |
| ridge_mix | 64 | +0.294 (4/10) | +0.115 (5/10) |
| corner_kink | 4 | -0.660 (10/10) | -0.336 (8/10) |
| corner_kink | 8 | -0.557 (10/10) | -0.131 (7/10) |
| corner_kink | 16 | -0.563 (10/10) | -0.096 (7/10) |
| corner_kink | 32 | -0.580 (10/10) | -0.223 (8/10) |
| corner_kink | 64 | -0.333 (9/10) | +0.169 (0/10) |
| smooth_bowl | 4 | -0.070 (6/10) | -0.034 (6/10) |
| smooth_bowl | 8 | -0.016 (6/10) | -0.045 (5/10) |
| smooth_bowl | 16 | -0.477 (7/10) | -0.535 (6/10) |
| smooth_bowl | 32 | -0.409 (9/10) | -0.397 (8/10) |
| smooth_bowl | 64 | -0.118 (6/10) | +0.035 (5/10) |
| highfreq_grid | 4 | -0.008 (7/10) | -0.007 (7/10) |
| highfreq_grid | 8 | -0.009 (7/10) | -0.004 (6/10) |
| highfreq_grid | 16 | -0.007 (8/10) | -0.001 (5/10) |
| highfreq_grid | 32 | +0.007 (3/10) | +0.017 (3/10) |
| highfreq_grid | 64 | -0.013 (8/10) | -0.007 (7/10) |
| angled_kink | 4 | -0.792 (10/10) | -0.666 (9/10) |
| angled_kink | 8 | -0.689 (10/10) | -0.900 (9/10) |
| angled_kink | 16 | -0.511 (8/10) | -0.712 (10/10) |
| angled_kink | 32 | -0.213 (7/10) | -0.696 (8/10) |
| angled_kink | 64 | -0.053 (7/10) | -0.159 (6/10) |

## Kernaufgaben: primäres Gate

| Aufgabe | vs SiLU | vs GELU | PASS |
|---|---|---|---|
| radial_ripple | -0.708 (38/50) | -0.642 (40/50) | True |
| ridge_mix | -0.540 (38/50) | -0.426 (38/50) | True |
| corner_kink | -0.481 (49/50) | -0.078 (30/50) | True |

## Nach Tiefe: alle Aufgaben

| Tiefe | vs SiLU | vs GELU |
|---:|---|---|
| 4 | -0.671 (53/60) | -0.507 (50/60) |
| 8 | -0.468 (52/60) | -0.330 (46/60) |
| 16 | -0.406 (46/60) | -0.290 (43/60) |
| 32 | -0.331 (44/60) | -0.199 (42/60) |
| 64 | -0.038 (39/60) | +0.019 (27/60) |

## Entscheidung

- Core PASS-Aufgaben: 3/3.
- Präregistriertes Gate: PASS.
- Neue Aufgaben werden separat als Generalisierungsbefund berichtet.
