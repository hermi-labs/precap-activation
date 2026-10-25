# V6 — Auswertung der c-/s-Ablation (21.10.2026)

Integrität: 105/105 Runs, 630/630 Checkpoint-Zeilen, eindeutige Schlüssel, keine NaN, Formel-Selfcheck = 0.
Metrik: d = log10(MSE_Variante / MSE_base); negativ = besser als precap_base. 5 Seeds (900–904), Schritt 1200.

## Kriterium (eingefroren): niedrigere Median-MSE UND >= 4/5 Seeds besser als base
| Variante | radial_ripple d64 | ridge_mix d64 | Anker radial d16 |
|---|---|---|---|
| c075 | +0.570 (1/5) | -0.122 (3/5) | +0.149 (1/5) |
| c125 | +0.124 (2/5) | -0.041 (5/5) | -0.027 (3/5) |
| s075 | +0.495 (1/5) | +0.193 (1/5) | +0.299 (1/5) |
| s225 | +0.151 (1/5) | -0.016 (4/5) | -0.277 (3/5) |

Formal erfüllt: c125 und s225 nur in ridge_mix d64. Nicht erfüllt in radial_ripple d64. Effekte klein (Median -0.04 bzw. -0.02), bei beiden trägt Seed 901 (d = -0.62 bei c125, -0.35 bei s225) viel; dieser Seed ist in ridge_mix d64 auch der schwächste Seed der Basis. Bewertung: kein belastbarer Treffer.

## Basis gegen Referenzen
- radial d16: base besser als SiLU und GELU in 5/5 Seeds (Median -0.99 / -0.68).
- radial d64: 4/5 (Median -0.69 / -0.32); Seed 900 ist der Ausreißer (base Test-MSE 0.0122 gegen ca. 0.0006–0.0024 bei den übrigen).
- ridge_mix d64: 4/5 (Median -0.32 / -0.46); Seed 901 ist der Ausreißer.

## Variabilität
Variationskoeffizient der Test-MSE von base: radial d64 = 1.44, ridge d64 = 1.03, radial d16 = 0.47. Die Streuung steigt mit Tiefe stark.

## Verläufe (Median log10 Variante/base)
- radial d64: c075 und s075 laufen ab Schritt 300 stetig schlechter weg (bis +0.57 / +0.49). c125 und s225 sind bei Schritt 800 besser (-0.21 / -0.23), geben den Vorteil bis Schritt 1200 aber wieder ab (+0.12 / +0.15).
- radial d16: s225 ist ab Schritt 500 klar besser (-0.85, am Ende -0.28); s075 deutlich schlechter.
- ridge d64: kein stabiles Muster, Vorzeichen wechselt.
- base hat in radial d64 bei Seed 901 einen Spike (Faktor 4.57 zwischen Checkpoints), erholt sich aber auf 0.0006.
- Der Vorteil der Basis gegen SiLU/GELU entsteht in d16 erst ab Schritt 500–800, nicht früh.

## Stabilität
Kleine Peaks sind nicht gleich gute Genauigkeit: s075 hat die kleinsten hmax_peak (radial d64: 1.58 gegen 2.40 bei base), aber die schlechteste MSE. c075 hat den höchsten Cap-Anteil (0.022–0.039) und ist ebenfalls schlechter.

## Schlussfolgerung
1. Stärkere Begrenzung (c075, s075): abgelehnt.
2. Schwächere Begrenzung (c125, s225): nur schwacher, aufgabenabhängiger Hinweis; s225 evtl. relevant für mittlere Tiefe.
3. Variabilität (einzelne Ausreißer-Seeds) dominiert die Tiefe-64-Ergebnisse.
4. Basis c=1, s=1.5 bleibt Standard.

Grenzen: 5 Seeds; 5/5 entspricht einseitig p = 0.031, 4/5 p = 0.19. Nur Hinweise.
