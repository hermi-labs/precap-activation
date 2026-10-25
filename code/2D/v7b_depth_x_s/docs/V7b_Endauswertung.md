# v7b — Endauswertung Tiefe x Sättigungsbreite (21.10.2026)

Status: gültig, 640/640 Runs, 3840/3840 Zeilen, keine NaN/inf, Hashes unverändert.

## Primär (eingefroren): s225 vs base, d=log10(MSE_s225/MSE_base)
| Aufgabe | Tiefe | Median d | Siege | p | Zelle bestätigt (p<0.0125, >=13/20) |
|---|---|---|---|---|---|
| radial | 8 | -0.142 | 15/20 | 0.0021 | ja |
| radial | 16 | -0.213 | 14/20 | 0.0664 | nein |
| radial | 32 | -0.309 | 16/20 | 0.0007 | ja |
| radial | 64 | -0.127 | 13/20 | 0.0348 | nein |
| ridge | 8 | -0.101 | 12/20 | 0.351 | nein |
| ridge | 16 | +0.064 | 7/20 | 0.853 | nein |
| ridge | 32 | -0.015 | 11/20 | 0.478 | nein |
| ridge | 64 | +0.006 | 10/20 | 0.435 | nein |

**Formales Primärurteil:** Tiefe x s ist NICHT bestätigt. Bestätigt sind Tiefe 8 und 32, aber nicht benachbart; der d16/d64-Kontrast ist nicht anwendbar, weil d16 hier nicht bestätigt wurde.

## Befund trotz formalem Nein
Alle vier radialen Tiefen haben einen negativen Median (-0.13 bis -0.31); kein Tiefenmuster, kein Wirkungsverlust zu d64 in der Richtung. Exploratorisch (nicht präregistriert): radial gepoolt Median -0.187, 58/80, p=3.7e-6; seed-gemittelt 18/20, p=4.1e-5. ridge_mix: Median 0.001, 40/80, p=0.54, also keine Replikation.

## Interpretation
s225 ist ein aufgabenspezifischer Vorteil auf radial_ripple (ca. 25-50 % kleinerer Median-MSE, d=-0.13 bis -0.31), kein tiefenabhängiger und kein allgemeiner Effekt. Das v7a-Ergebnis (d16: -0.256) ist in Richtung und Größe repliziert (-0.213), aber knapp unter der strengen Zellschwelle.

## Referenzen (explorativ, radial, Median log10-Verhältnis je Tiefe 8/16/32/64)
- base vs SiLU: -1.47/-0.97/-0.39/-0.30 (70/80 Siege); base vs GELU: -0.79/-0.61/-0.29/-0.31 (71/80).
- s225 vs SiLU: -1.52/-0.99/-0.67/-0.53 (78/80); s225 vs GELU: -0.89/-0.72/-0.67/-0.59 (76/80).
- Der Vorsprung gegenüber SiLU/GELU schrumpft mit der Tiefe, ist aber bei s225 auch bei d64 noch deutlich.

## Konsequenz
Keine adaptive c/s-Regel. Base bleibt Standard; s225 bleibt aufgabenspezifische Option für radial-artige Probleme. Nächster Schritt: Aufgabenabhängigkeit untersuchen (weitere Zielfunktionen), nicht Tiefe.
