# V1 – Präregistrierung: LR-Anlauf statt kleiner LR (CIFAR-10, d100, TAT-Rezept, privat)

Stand: 02.11.2026, nachts, vor dem Start. Die Freeze-Zeit steht in `meta/frozen_at.txt`. PreCap, die Tiefenfamilie und Lagma sind Aarons Ideen. Das Design ist mit Claude entworfen und wird in Aarons Abwesenheit gestartet (Leitlinie für das Alleinarbeiten). Teil V1 des Fahrplans v0.4.

## Ausgangslage

- **v0.3 nennt als Grenze:** „BPreCap trainiert nur mit kleinen LR, bei einer LR-Spitze von 2·10⁻⁴ divergiert es.“ Im B10-Tuning brechen BPreCap und LB bei 2e-4 ein (Val-Fehler 82–86 %), TAT nicht.
- **D2 (präregistriert, beschreibend):**
  - Die Hülle ist nicht die Ursache. LB bricht ganz ohne Sättigung ein, ein harter Clip genauso.
  - Die Gradienten verschwinden nach vorne, die Gewichte wachsen nicht.
- **D2b (explorativ, ein Seed):** Ein linearer Anlauf über 5 % der Schritte verhindert den Einbruch. BPreCap kommt bei 2e-4 auf 45,9 % (B10-Wahl 1e-4: 45,8 %), LB auf 48,7 %. Ohne Anlauf springt früh vor allem die Eingangsschicht.
- **B10 lief ohne Anlauf, auch für TAT.** Ein fairer Test braucht den Anlauf für alle Formen.

## Frage

Hält der 4-Bit-Vorteil von BPreCap gegen TAT (B10: −4,48 pp gegen TAT mit WD in B10b, −11,67 pp gegen TAT in der B10-Wahl), wenn **alle** Formen mit LR-Anlauf trainieren und die LR bis 5e-4 getunt wird? Und löst der Anlauf die Grenze „nur kleine LR“?

## Eine Änderung gegenüber B10: linearer LR-Anlauf

- **LR-Plan:** linear von 0 auf die LR-Spitze über 344 Schritte (5 % von 6880), danach Kosinus auf 0 über die restlichen 6536 Schritte. Das gilt für alle Formen (kL2B_tr, LB_tr, tat).
- **Tuning wie B10,** nur mit verschobenem LR-Raster: LR-Spitze {1e-4, 2e-4, 5e-4} × WD {0; 0,3; 3,0}, 2 Tuning-Seeds (42000, 42001), Wahlregel wie B10 (höchste mediane Val-Genauigkeit am Ende, Gleichstand: kleinere LR, dann kleinerer WD, Ausfall = 10 %).
  - Das Raster ist nach oben verschoben, weil mit Anlauf größere LR möglich sein sollten.
  - 5e-5 fällt weg. In B10 lag keine Wahl dort.
- **Konstanten wie B10:** Formen, Rezept, Init, Netz (d100, Breite 256), Batch 128, 6880 Schritte, Messpunkte alle 860 Schritte, frühes Stoppen, Daten, Kalibrierung, Quantisierung, Cache-Grenze.
- **Seeds der Bestätigung:** **dieselben wie B10** (42168–42179). So sind Init und Batch-Reihenfolge gleich, und V1 lässt sich je Seed mit B10 paaren.

## Läufe

- **Tuning:** 3 Formen × 9 × 2 = 54 Läufe.
- **Bestätigung:** 3 Formen × 12 Seeds = 36 Läufe.
- **Kontrolle:** kL2B_tr (1e-4/0,3) und tat (2e-4/0), Seed 42168, Kosinus **ohne** Anlauf. Alle 7 Kennzahlen an allen 8 Messpunkten müssen bitgleich zur B10-Bestätigung sein.
- **Zusammen:** 92 Läufe, geschätzt 3–4 h auf 2 Workern (M1 Pro), nur am Netzteil.

## Primärtest

**kL2B_tr − TAT bei w8a4a_best am Punkt des frühen Stoppens**, beide mit Anlauf. Wilcoxon zweiseitig exakt, 12 Paare, α = 0,05.

**Erfolg:** p < 0,05 und Median < 0 („besser“).

## Sekundär (Holm über 8)

| Nr. | Vergleich | Frage |
|---|---|---|
| S1 | kL2B_tr − TAT, fp32, ES | fp32-Gleichstand mit Anlauf? |
| S2 | kL2B_tr V1 − kL2B_tr B10, fp32, ES (gepaart über Seed) | Hilft der Anlauf BPreCap in fp32? |
| S3 | kL2B_tr V1 − kL2B_tr B10, w8a4a, ES | … und bei 4 Bit? |
| S4 | TAT V1 − TAT B10, fp32, ES | Hilft der Anlauf TAT? |
| S5 | TAT V1 − TAT B10, w8a4a, ES | … und bei 4 Bit? |
| S6 | LB_tr − TAT, w8a4a, ES | wie B10 S3 |
| S7 | kL2B_tr − LB_tr, fp32, ES | Kern gegen linear wie B10 S4 |
| S8 | kL2B_tr − TAT, w8a4a, Ende (6880) | ohne frühes Stoppen |

**Beschreibend:**
- Tuning-Tabelle. Bricht BPreCap mit Anlauf bei 2e-4 oder 5e-4 noch ein (Val-Fehler über 80 %)?
- Gewählte LR und WD, Lage am Rand.
- Stopp-Punkte, 4-Bit-Preis, Bereich.

## Deutungsregeln (vorab)

| Ergebnis | Deutung |
|---|---|
| Erfolg und Wahl für BPreCap ≥ 2e-4 ohne Einbruch | Die v0.3-Grenze „nur kleine LR“ lag am fehlenden Anlauf. Der 4-Bit-Vorteil gegen TAT hält auch im faireren Regime |
| Erfolg, Wahl für BPreCap bleibt 1e-4 | Der Vorteil hält. Der Anlauf verhindert den Einbruch, eine größere LR bringt BPreCap aber nichts |
| Primär „unklar“, S5 „besser“ | Mit Anlauf holt TAT bei 4 Bit auf. Der v0.3-Vorteil hing teilweise am Fehlen des Anlaufs. Das muss in v0.4 offen stehen |
| Primär „schlechter“ | Mit Anlauf ist TAT bei 4 Bit vorn. Die v0.3-Aussage gilt nur ohne Anlauf |
| Kontrolle nicht bitgleich | Abweichung wird gesucht und berichtet, die Tests gelten |

## Vorab-Erwartung

| Größe | Erwartung |
|---|---|
| BPreCap bricht mit Anlauf bei 2e-4 nicht mehr ein (Tuning, beide Seeds) | 0,8 |
| BPreCap bricht mit Anlauf bei 5e-4 ein | 0,5 |
| Primär „besser“ (Erfolg) | 0,65 |
| S2: Anlauf verbessert BPreCap in fp32 | 0,3 |
| S4: Anlauf verbessert TAT in fp32 | 0,4 |
| S5: Anlauf senkt TATs 4-Bit-Fehler | 0,4 |
| Kontrolle bitgleich | 0,95 |

## Grenzen

- Eine Anlauflänge (5 %), nicht gesucht.
- Nur CIFAR-10, d100, TAT-Rezept, nur MLPs.
- Raster ohne 5e-5. Falls TATs beste LR mit Anlauf darunter läge, fehlt sie.
- Gepaart mit B10 über den Seed, aber mit anderem LR-Plan und eventuell anderer LR. S2–S5 messen die Wirkung des ganzen Pakets „Anlauf plus neues Tuning“, nicht des Anlaufs allein.

## Freeze

Eingefroren werden:
- `v1_run.py`, `v1_analyze.py`;
- `test_v1_mechanik.py`, `test_v1_analyse.py`;
- diese Datei;
- `src/precap/*.py`;
- die CIFAR-10-Dateien.

**Vorab gemachte Tests:**
- **Selbsttest:** n_confirm = 38, n_tune = 54, Raster 1e-4/2e-4/5e-4.
- **Mechanik:** MECHANIK OK.
  - Formen, Init und Daten gleich B10.
  - Seeds wie B10, run_ids eindeutig.
  - Plan linear bis 344, dann Kosinus.
  - Pfad „cos“ bitgleich zu B10 in einem Kurzlauf mit Zwischenmessung, „wcos“ wirkt unter compile.
- **Auswertung an künstlichen Daten:** AUSWERTUNG OK, mit den Szenarien vorn, hinten, Null (20× ohne Fehlalarm), falsche Kontrolle am Ende bzw. am Messpunkt, Ausfall.
