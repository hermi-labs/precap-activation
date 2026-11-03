# V1c – Präregistrierung: LR-Anlauf bei Tiefe 200, BPreCap gegen SELU (CIFAR-10, d200, TAT-Rezept, privat)

Stand: 03.11.2026, morgens, vor dem Start. Die Freeze-Zeit steht in `meta/frozen_at.txt`. PreCap, BPreCap und die Tiefenfamilie sind Aarons Ideen. Das Design ist mit Claude entworfen und wird in Aarons Abwesenheit gestartet (Leitlinie fürs Alleinarbeiten, Freigabe vom 03.11. bis 18:00). Kandidat für v0.3.1. Aufbau wie V1, übertragen von B10 (d100, gegen TAT) auf B15 (d200, gegen SELU).

## Ausgangslage

- **V1 (d100, präregistriert, Erfolg):** Mit 5 % Anlauf für alle Formen bricht BPreCap bei 2e-4 nicht mehr ein, und der 4-Bit-Vorteil gegen TAT hält (−8,82 pp, 12/0).
- **V1b:** Mit Anlauf liegt die Grenze bei 3e-4, die beste LR bleibt 1e-4.
- **Bei d200 ist die Grenze enger (B15-Tuning, ohne Anlauf):** BPreCap bricht schon bei 1e-4 in allen WD ein (Val-Fehler 85–86 %). Gewählt wurde 5e-5 / 0,3.
- **SELU bei d200 (B15):** gewählt 2e-4 / 0,3, am oberen Rand des Rasters. 5 von 12 Läufen trainieren kaum.
- **v0.3 sagt:** „BPreCap is better by 11.97 pp in fp32 and 8.95 pp at 4 bits (12/0)“ und „at depth 200 it diverges already at peak LR 10⁻⁴“. Beides lief ohne Anlauf, für beide Formen.

## Frage

Hält der Vorteil von BPreCap gegen SELU bei d200, wenn **beide** mit Anlauf trainieren und die LR bis 3e-4 getunt wird? Hebt der Anlauf auch bei d200 die LR-Grenze von BPreCap?

## Eine Änderung gegenüber B15: linearer LR-Anlauf

- **LR-Plan:** linear von 0 auf die LR-Spitze über 344 Schritte (5 % von 6880), danach Kosinus auf 0. Das gilt für beide Formen (kL2B_tr, selu_tr). Der Code ist derselbe wie in V1.
- **Tuning wie B15,** mit verschobenem Raster: LR-Spitze {5e-5, 1e-4, 2e-4, 3e-4} × WD {0; 0,3; 3,0}, Tuning-Seeds 42000 und 42001.
  - Wahlregel wie B15: höchste mediane Val-Genauigkeit am Ende; bei Gleichstand die kleinere LR, dann der kleinere WD; ein Ausfall zählt als 10 %.
  - 2e-5 fällt weg, weil in B15 keine Wahl dort lag. 3e-4 kommt hinzu, weil SELU in B15 am oberen Rand lag.
- **Konstanten wie B15:**
  - Formen mit d200-Konstanten, Rezept, Init, Netz (d200, Breite 256);
  - Batch 128, 6880 Schritte, Messpunkte alle 860 Schritte, frühes Stoppen;
  - Daten, Kalibrierung, Quantisierung, Cache-Grenze.
- **Seeds der Bestätigung:** dieselben wie B15 (42192–42203), damit gepaart mit B15.

## Läufe

- **Tuning:** 2 Formen × 12 × 2 = 48.
- **Bestätigung:** 2 Formen × 12 Seeds = 24.
- **Kontrolle:** kL2B_tr (5e-5/0,3) und selu_tr (2e-4/0,3), Seed 42192, Kosinus **ohne** Anlauf. Alle 7 Kennzahlen an allen 8 Messpunkten müssen bitgleich zur B15-Bestätigung sein.
- **Zusammen:** 74 Läufe, geschätzt 3–3,5 h auf 2 Workern (M1 Pro), nur am Netzteil.

## Primärtest

**kL2B_tr − selu_tr bei w8a4a_best am Punkt des frühen Stoppens**, beide mit Anlauf. Wilcoxon zweiseitig exakt, 12 Paare, α = 0,05. **Erfolg:** p < 0,05 und Median < 0.

## Sekundär (Holm über 6)

| Nr. | Vergleich | Frage |
|---|---|---|
| S1 | kL2B_tr − selu_tr, fp32, ES | Hält der fp32-Vorteil? |
| S2 | kL2B_tr V1c − kL2B_tr B15, fp32, ES (gepaart über Seed) | Hilft der Anlauf BPreCap in fp32? |
| S3 | kL2B_tr V1c − kL2B_tr B15, w8a4a, ES | … und bei 4 Bit? |
| S4 | selu_tr V1c − selu_tr B15, fp32, ES | Hilft der Anlauf SELU? |
| S5 | selu_tr V1c − selu_tr B15, w8a4a, ES | … und bei 4 Bit? |
| S6 | kL2B_tr − selu_tr, w8a4a, Ende (6880) | ohne frühes Stoppen |

**Beschreibend:**
- **Tuning-Tabelle:** Bricht BPreCap mit Anlauf bei 1e-4, 2e-4 oder 3e-4 ein (Val-Fehler in beiden Seeds über 80 %)?
- **Wahl:** gewählte LR und WD, Lage am Rand.
- **Wie oft SELU kaum trainiert:** Trainingsverlust am Ende über 2,0, im Vergleich zu B15 (5 von 12).
- **4-Bit-Preis von BPreCap bei d200:** B15 +7,3 pp.
- **Außerdem:** Stopp-Punkte und Bereich.

## Deutungsregeln (vorab)

| Ergebnis | Deutung |
|---|---|
| Erfolg und Wahl für BPreCap ≥ 1e-4 ohne Einbruch | Der Anlauf hebt die LR-Grenze auch bei d200. Der Vorteil gegen SELU hält im faireren Regime |
| Erfolg, Wahl für BPreCap bleibt 5e-5 | Der Vorteil hält. Der Anlauf bringt BPreCap bei d200 keine größere LR. Aus dem Tuning folgt, ob er den Einbruch bei 1e-4 verhindert |
| Primär „unklar“, S4 oder S5 „besser“ | Mit Anlauf holt SELU bei d200 auf. Der Vorteil aus v0.3 hing teilweise am fehlenden Anlauf. Das muss in v0.3.1 offen stehen |
| Primär „schlechter“ | Mit Anlauf ist SELU bei d200 bei 4 Bit vorn. Die Aussage aus v0.3 gilt nur ohne Anlauf und wird korrigiert |
| Kontrolle nicht bitgleich | Die Abweichung wird gesucht und berichtet. Die Tests gelten trotzdem |

## Vorab-Erwartung

| Größe | Erwartung |
|---|---|
| BPreCap bricht mit Anlauf bei 1e-4 nicht mehr ein (Tuning, beide Seeds, mind. ein WD) | 0,7 |
| Wahl für BPreCap ≥ 1e-4 | 0,45 |
| Primär „besser“ (Erfolg) | 0,7 |
| S4: Anlauf verbessert SELU in fp32 | 0,5 |
| SELU trainiert mit Anlauf in allen 12 Läufen (Verlust ≤ 2,0) | 0,35 |
| Kontrolle bitgleich | 0,95 |

## Grenzen

- Nur eine Anlauflänge (5 %), wie in V1.
- Nur CIFAR-10, d200, TAT-Rezept, nur MLPs.
- S2–S5 messen das Paket „Anlauf plus neues Tuning“, nicht den Anlauf allein.
- SELU wird mit Gauß-artiger Init im TAT-Rezept trainiert wie in B12/B15, also nicht mit der LeCun-Init aus dem SELU-Paper.

## Freeze

**Eingefroren werden:**
- `v1c_run.py`, `v1c_analyze.py`;
- `test_v1c_mechanik.py`, `test_v1c_analyse.py`;
- diese Datei;
- `src/precap/*.py`;
- die CIFAR-10-Dateien.

**Vorab gemachte Tests:**
- **Selbsttest:** n_confirm = 26, n_tune = 48.
- **Mechanik:** MECHANIK OK.
  - Formen, Init, Daten und Konstanten gleich B15.
  - Seeds wie B15, run_ids eindeutig.
  - Plan linear bis 344, dann Kosinus.
  - Pfad „cos“ bitgleich zu B15 in einem Kurzlauf mit Zwischenmessung, „wcos“ wirkt.
- **Auswertung an künstlichen Daten:** AUSWERTUNG OK. Szenarien:
  - BPreCap vorn, SELU vorn;
  - Null (20 Wiederholungen, 0 Fehlalarme);
  - falsche Kontrolle am Ende bzw. am Messpunkt;
  - Ausfall.
- **Rauchtest:** im Startskript.
