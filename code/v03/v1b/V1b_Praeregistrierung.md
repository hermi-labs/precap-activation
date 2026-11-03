# V1b – Präregistrierung: Wo liegt die LR-Grenze mit Anlauf? (beschreibend, CIFAR-10 d100, privat)

Stand: 03.11.2026, nachts, vor dem Start. Die Freeze-Zeit steht in `meta/frozen_at.txt`. PreCap und BPreCap sind Aarons Ideen. Das Design ist mit Claude entworfen und wird in Aarons Abwesenheit gestartet (Leitlinie für das Alleinarbeiten). Kleine Folgestudie zu V1, Teil V1 des Fahrplans v0.4.

## Ausgangslage (V1, präregistriert, Erfolg)

- Mit 5 % Anlauf bricht BPreCap bei 2e-4 nicht mehr ein, bei 5e-4 schon. TAT verliert bei 5e-4 ebenfalls stark.
- Das Tuning wählt für BPreCap 1e-4, das ist der **untere Rand** des V1-Rasters.
- **Offen:**
  - Wo zwischen 2e-4 und 5e-4 liegt die Grenze?
  - Schiebt ein längerer Anlauf sie nach oben?
  - Wäre 5e-5 mit Anlauf besser als 1e-4?

## Fragen (beschreibend, keine Signifikanztests, 2 Seeds)

| Nr. | Frage | Kriterium |
|---|---|---|
| G1 | Höchste LR ohne Einbruch mit 5 % bzw. 10 % Anlauf, je Form | „Einbruch“: Val-Fehler am Ende über 80 %, Ausfall zählt als Einbruch. Die Grenze ist die höchste LR aus {2e-4, 3e-4, 5e-4}, bis zu der in beiden Seeds kein Einbruch auftritt, von unten gezählt |
| G2 | Verschiebt 10 % Anlauf die Grenze? | ja, wenn es eine LR gibt, bei der mit 344 Schritten ein Einbruch auftritt und mit 688 in keinem Seed |
| G3 | Unterer Rand: 5e-5 mit 5 % Anlauf gegen 1e-4 (V1-Tuning, gleicher WD) | Median-Differenz Val-Fehler. Unter −0,5 pp: „5e-5 besser“, über +0,5 pp: „5e-5 schlechter“, sonst „gleich“ |
| G4 | 10 % gegen 5 % bei 2e-4 | Median-Differenz Val-Fehler, nur beschreibend |

Daneben wird der 4-Bit-Fehler (w8a4a_best am Ende) je Zelle berichtet, nur beschreibend.

## Änderungen gegenüber V1

- **Nur die Tuning-Stufe:** keine Wahl, keine Bestätigung.
- **Formen:** kL2B_tr mit WD 0,3 und tat mit WD 0. Das ist der WD ihrer V1-Wahl. LB_tr entfällt.
- **Raster:** LR-Spitze {5e-5, 2e-4, 3e-4, 5e-4} × Anlauf {344 (5 %), 688 (10 %)}, danach Kosinus auf 0.
- **Seeds:** 42000, 42001 wie das V1-Tuning.
- **Alles andere wie V1/B10:** Rezept, Init, Netz (d100, Breite 256), Batch 128, 6880 Schritte, Messpunkte, Daten, Kalibrierung, Quantisierung.

## Läufe

- 2 Formen × 4 LR × 2 Anläufe × 2 Seeds = **32 Läufe**, geschätzt 50–60 min auf 2 Workern (M1 Pro), nur am Netzteil.
- **Kontrolle eingebettet:** Die Zellen mit LR 2e-4 und Anlauf 344 (4 Läufe) müssen in allen Kennzahlen bitgleich zum V1-Tuning sein.

## Deutungsregeln (vorab)

| Ergebnis | Deutung |
|---|---|
| G1 BPreCap mit 5 % ≥ 3e-4 | Die v0.4-Formulierung lautet „trainiert mit Anlauf bis 3·10⁻⁴“, statt nur „bei 2·10⁻⁴“ |
| G1 BPreCap mit 5 % = 2e-4 | Die Grenze liegt zwischen 2e-4 und 3e-4. Die Formulierung aus V1 bleibt |
| G2 ja | Die Grenze hängt an der Anlauflänge. In v0.4 wird ein längerer Anlauf empfohlen |
| G2 nein | Ein längerer Anlauf hilft hier nicht. Die Grenze ist eher eine Eigenschaft von Form und Rezept |
| G3 „5e-5 besser“ | Die V1-Wahl am unteren Rand war nicht optimal. Die V1-Ergebnisse gelten wie präregistriert, das wird aber offen vermerkt. Dann sollte eine V1-Bestätigung mit 5e-5 erwogen werden |
| G3 „gleich“ oder „schlechter“ | Die V1-Wahl 1e-4 ist am unteren Rand ausreichend abgesichert |
| Kontrolle nicht bitgleich | Abweichung wird gesucht und berichtet |

V1b ändert den V1-Primärtest nicht.

## Vorab-Erwartung

| Größe | Erwartung |
|---|---|
| G1 BPreCap mit 5 % hält bis 3e-4 | 0,5 |
| G1 TAT mit 5 % hält bis 3e-4 | 0,6 |
| G2 10 % verschiebt die Grenze bei BPreCap | 0,4 |
| G3 BPreCap „5e-5 besser“ | 0,15 (in B10 ohne Anlauf war 5e-5 schlechter als 1e-4) |
| Kontrolle bitgleich | 0,95 |

## Grenzen

- 2 Seeds, nur beschreibend. Die Lage der Grenze ist damit grob.
- Nur zwei Anlauflängen, WD fest auf die V1-Wahl.
- Nur CIFAR-10, d100, TAT-Rezept, MLPs.

## Freeze

**Eingefroren werden:**
- `v1b_run.py`, `v1b_analyze.py`;
- `test_v1b_mechanik.py`, `test_v1b_analyse.py`;
- diese Datei;
- `src/precap/*.py`;
- die CIFAR-10-Dateien.

**Vorab gemachte Tests:**
- **Selbsttest:** n_tune = 32.
- **Mechanik:** MECHANIK OK.
  - Formen, Init, Daten und Konstanten gleich V1.
  - 32 eindeutige run_ids.
  - Plan linear bis 688, dann Kosinus.
  - `wcos344` bitgleich zum V1-Pfad in einem Kurzlauf über 800 Schritte, `wcos688` wirkt.
- **Auswertung an künstlichen Daten:** AUSWERTUNG OK. Szenarien:
  - Grenze bei 2e-4, 3e-4 und 5e-4;
  - 10 % verschiebt oder verschiebt nicht;
  - unterer Rand besser oder gleich;
  - Ausfall zählt als Einbruch;
  - falsche Kontrolle wird erkannt.
