# V9 Endauswertung — Geometrie-Kontinuum (21.10.2026)

Status: gültig abgeschlossen. 400/400 Runs, 2400/2400 Zeilen, eindeutige Schlüssel, 0 NaN/inf, gefrorene Hashes unverändert.

## Eingefrorene Primärentscheidung
Geometrie-Skalierung sollte gelten, wenn (1) Spearman rho zwischen Anisotropie und Zellmedian d >=0.6, (2) seed-gemittelter Endpunktkontrast a=0 < a=1 mit einseitigem Wilcoxon p<0.05 und (3) a=0 Median d<0.

Ergebnis: rho=0.739, p=0.0147 erfüllt Kriterium 1. Der Endpunktkontrast ist aber nur -0.062, 11/20, p=0.194 und verfehlt Kriterium 2. a=0 ist negativ (-0.129 seed-gemittelt), erfüllt Kriterium 3. Gesamturteil: **Geometrie-Skalierung NICHT bestätigt.**

## Zellbild
| a / Anisotropie | d16 Median d | d32 Median d |
|---|---:|---:|
| 0.00 / 0.013 | -0.150 | -0.132 |
| 0.25 / 0.222 | -0.438 | -0.378 |
| 0.50 / 0.449 | -0.112 | -0.154 |
| 0.75 / 0.786 | -0.082 | +0.201 |
| 1.00 / 0.845 | -0.019 | -0.026 |

Der stärkste s225-Vorteil tritt unerwartet bei mittlerer Geometrie a=0.25 auf, nicht beim reinen radialen Endpunkt. Daher ist die Beziehung nicht monoton genug für die ursprüngliche Ursache-Hypothese.

## Explorativ
Je Tiefe ist die Beziehung bei d16 positiv (rho=0.9, p=0.037), bei d32 aber nicht belastbar (rho=0.6, p=0.285). Über Tiefen gepoolt ist a=.25 klar besser (-0.411, 33/40, p=3.9e-7); a=.75 ist schlechter (+0.144, 15/40, p=0.965). Die Kontraste .25 vs .5 und .5 vs .75 sind signifikant, der Endpunktkontrast 0 vs 1 nicht. Dies unterstützt ein nichtmonotones, resonanzartiges Aufgabenphänomen statt einer einfachen Radialitätsregel.

## Schluss
V9 widerlegt keine kleine geometrische Rolle, aber sie widerlegt die einfache Regel "je radialer, desto besser s225". s225 bleibt kein universeller Ersatz für base und auch keine durch reine Anisotropie vorhersagbare Spezialoption.
