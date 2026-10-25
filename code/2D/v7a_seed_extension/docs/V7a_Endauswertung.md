# v7a — Endauswertung der unabhängigen Seed-Erweiterung

**Datum:** 21.10.2026  
**Status:** vollständig und gültig: 300/300 Runs, 1800/1800 Checkpoint-Zeilen, 1800 eindeutige Schlüssel, keine NaN, Formel-Selfcheck = 0.

## Design
Unabhängige Seeds 1000–1019; Zellen radial_ripple d64 und ridge_mix d64 (Mechanismus) sowie radial_ripple d16 (Anker). Verglichen wurden PreCap-base (c=1, s=1.5), c125, s225, SiLU und GELU. Alle übrigen Einstellungen entsprechen V6.

Metrik: d = log10(MSE_Variante/MSE_base); d<0 begünstigt die Variante gegenüber base.

## Eingefrorene Regel
Tiefe-64-Zelle bestanden, wenn: Median d<0, Wilcoxon einseitig p<0.025 und mindestens 13/20 Seeds gewinnen. Anker bestanden, wenn Median d<=+0.05.

## Ergebnis: Varianten gegen base
| Zelle | c125: Median d, Siege, p | s225: Median d, Siege, p | Urteil |
|---|---|---|---|
| radial_ripple d64 | -0.055; 11/20; 0.139 | -0.202; 13/20; 0.0448 | beide nicht bestanden (s225 verfehlt korrigiertes p-Kriterium) |
| ridge_mix d64 | +0.083; 6/20; 0.962 | -0.064; 12/20; 0.0768 | beide nicht bestanden |
| radial_ripple d16 (Anker) | +0.027; 9/20; 0.785 | -0.256; 16/20; 0.0002 | c125 nicht schlechter; s225 klar besser |

**Primäre Entscheidung:** c125 = KEIN BELASTBARER EFFEKT. s225 = KEIN BELASTBARER EFFEKT. Keiner der beiden festen Werte ist über beide Tiefe-64-Zellen robust.

## Basis gegen Referenzen
| Zelle | base vs SiLU | base vs GELU |
|---|---|---|
| radial_ripple d16 | -0.881; 20/20, PASS | -0.632; 20/20, PASS |
| radial_ripple d64 | -0.028; 11/20, kein PASS | -0.202; 13/20, PASS |
| ridge_mix d64 | -0.253; 18/20, PASS | -0.108; 14/20, PASS |

## Verläufe
- s225 ist auf radial d16 ab Schritt 500 stabil besser: d=-0.265 (500), -0.148 (800), -0.256 (1200).
- radial d64: s225 wird ab Schritt 300 besser und bleibt im Median bis zum Ende besser (-0.202), streut aber zu stark für den korrigierten Signifikanztest.
- c125 ist radial d64 zunächst neutral, am Schritt 800 besser (-0.170), aber am Ende nur -0.055; auf ridge d64 ab Schritt 500 schlechter (+0.079 bis +0.083).
- Die Referenzvorteile der Basis auf d16 entstehen nach Schritt 300; auf d64 ist radial gegenüber SiLU nicht stabil.

## Variabilität
Die Basis hat hohe, aber keine extreme Ausreißerrate nach der vorab gewählten >5x-Median-Regel (0 in allen v7a-Zellen). CV der finalen MSE: base 0.63 (radial d16), 0.87 (radial d64), 0.73 (ridge d64). Die hohe Tiefe-64-Streuung ist folglich breit über Seeds verteilt, nicht nur ein einzelner extremer Ausreißer.

## Explorative Pooling-Prüfung (V6+v7a, n=25; nicht primär)
Pooling ändert die Entscheidung nicht: c125 ist in radial d64 bei -0.038 / 13 von 25 / p=0.237 und in ridge d64 bei +0.033 / 11 von 25 / p=0.837. s225 ist bei -0.111 / 14 von 25 / p=0.091 (radial) und -0.038 / 16 von 25 / p=0.091 (ridge). Der s225-Hinweis bleibt daher explorativ, nicht bestätigt.

## Schlussfolgerung
1. Unterbegrenzung ist ausgeschlossen (V6: c075/s075 schlechter).
2. Eine global schwächere Begrenzung ist nicht belegt; s225 ist jedoch ein robuster positiver Befund für radial_ripple d16.
3. Die Tiefe-64-Grenze wird nicht durch einen einzelnen festen c- oder s-Wert gelöst.
4. Die Richtung ist aufgaben- und tiefenabhängig: s225 ist bei d16 stark, bei d64 nur ein instabiler Hinweis.
5. Die Basisformel c=1, s=1.5 bleibt Standard.
