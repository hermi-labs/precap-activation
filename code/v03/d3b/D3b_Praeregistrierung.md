# D3b – Präregistrierung (beschreibend): Warum erhöht der LR-Anlauf BPreCaps 4-Bit-Preis bei Tiefe 200? (privat)

Stand: 03.11.2026, nachmittags, vor dem Start. Die Freeze-Zeit steht in `meta/frozen_at.txt`. PreCap und BPreCap sind Aarons Ideen. Das Design ist mit Claude entworfen und wird in Aarons Abwesenheit gestartet (Leitlinie fürs Alleinarbeiten, Freigabe vom 03.11. bis 18:00).

## Anlass

- **V1c S3:** Mit 5 % Anlauf ist BPreCap bei d200 in fp32 0,59 pp besser, bei 4 Bit aber 1,64 pp schlechter als ohne Anlauf (0/12). Der 4-Bit-Preis steigt von +7,3 auf +9,5 pp.
- **V1 bei d100:** dieselbe Richtung, dort nicht signifikant (+0,31 pp).
- **D3:** Die Messung je Schicht an B15-Netzen (ohne Anlauf, Seeds 42192/42193) liegt bereits vor.

## Frage

Welche Größe aus der D3-Messung unterscheidet die Netze mit Anlauf (V1c) von denen ohne (B15)? Ist der lokale Rundungsfehler größer, die Dämpfung schwächer, der Anteil im weichen Bereich kleiner, oder häuft sich der Fehler stärker an?

## Läufe

- **Netze:** 4 Netze aus V1c mit gleichen Seeds wie D3, bitgleich nachtrainiert:
  - BPreCap kL2B_tr, LR 5e-5, WD 0,3;
  - SELU, LR 3e-4, WD 0,3, nur zur Einordnung;
  - je Seed 42192 und 42193.
- **Training:** wie die V1c-Bestätigung (Anlauf 344 Schritte, dann Kosinus, 6880 Schritte).
- **Pflichtprüfung:** val, test, w8a4a_best und gewählte Kalibrierung müssen bitgleich zur V1c-Zeile sein. Sonst ist der Lauf ungültig.
- **Messung:** `d3/d3_diag.py` unverändert, `measure()` und `verify()`, am Ende des Trainings (6880). Gleiche Kennzahlen wie in D3.
- **Vergleich:** mit den D3-Ergebnissen `d3/results/d3_{bp200,se200}_s{42192,42193}.json`.

## Lesarten (vorab, BPreCap, je Seed; ein Befund gilt nur, wenn beide Seeds gleich lauten)

Verhältnis q = V1c / B15:

| Nr. | Größe | Lesart |
|---|---|---|
| L1 | ε4, Median über die Schichten | q ≥ 1,1 „lokal größer“, q ≤ 1/1,1 „lokal kleiner“, sonst „gleich“ |
| L2 | Ausbreitungsgewinn G4, Median | q ≥ 1,1 „Dämpfung schwächer“, q ≤ 1/1,1 „Dämpfung stärker“, sonst „gleich“ |
| L3 | Anteil im weichen Bereich (\|a\| > 2,4), Median | q ≥ 1,25 „mehr“, q ≤ 1/1,25 „weniger“, sonst „gleich“ |
| L4 | relativer Logit-Fehler, alle Schichten 4 Bit (r_all4) | q ≥ 1,1 „größer“, q ≤ 1/1,1 „kleiner“, sonst „gleich“ |

**Beschreibend:**
- 4-Bit-Preis auf 2000 Testbildern;
- Anhäufung r4 je Schichtgruppe;
- Kurtosis, Max/p99,99;
- dieselben Lesarten für SELU.

## Deutung (vorab)

| Befund | Deutung |
|---|---|
| L4 „gleich“ | Der höhere 4-Bit-Preis kommt nicht aus einem größeren Logit-Fehler, sondern daraus, dass das Netz mit Anlauf empfindlicher auf denselben Fehler reagiert (z. B. knappere Entscheidungsränder). Das wird nur so benannt, nicht weiter getestet |
| L4 „größer“ und L1 „lokal größer“ | Der Anlauf führt zu Schichteingängen, die schlechter runden. Mit L3 „weniger“ passt das zur Vermutung, dass weniger Werte im weichen Bereich liegen (D3: F6, Korrelation −0,72) |
| L4 „größer“ und L2 „Dämpfung schwächer“, L1 „gleich“ | Der lokale Fehler ist gleich, aber er wird weniger gedämpft |
| alles „uneinheitlich“ | Mit 2 Seeds nicht entscheidbar |

## Vorab-Erwartung

| Größe | Erwartung |
|---|---|
| alle 4 Netze bitgleich | 0,9 |
| L4 „größer“ | 0,55 |
| L1 „lokal größer“ | 0,35 |
| L2 „Dämpfung schwächer“ | 0,35 |
| L3 „weniger“ | 0,35 |

## Grenzen

- Nur 2 Seeds, nur beschreibend.
- Gemessen am Ende (6880), nicht am Punkt des frühen Stoppens. Für die Seeds 42192 und 42193 stoppt V1c laut Bericht meist bei 6880, der Primärtest nutzt aber das frühe Stoppen.
- Gemessen wird nur an 2000 Testbildern.
- Die Schwellen 1,1 und 1,25 sind gesetzt, nicht hergeleitet.

## Freeze und Tests

- **Eingefroren:**
  - `d3b_diag.py`, `d3b_auswerten.py`, diese Datei, die Tests;
  - `../d3/d3_diag.py` und `../v1c/v1c_run.py` (gegen ihren Freeze geprüft);
  - die vier D3-Ergebnisdateien.
- **Selbsttest:** ok. 4 Läufe; der Plan liefert 0 am Start, die Spitze bei 344 und 0 am Ende; die Vergleichszeilen und die D3-Ergebnisse werden gefunden; die Konstanten sind gleich B15.
- **Rauchtest:** ok, 60 Schritte mit Anlauf 20, volle Messung, alle Kennzahlen endlich.
- **Auswertung an künstlichen Daten:** AUSWERTUNG OK. Geprüft wurden:
  - Null;
  - Hypothese;
  - uneinige Seeds;
  - ungültiger Lauf;
  - Schwelle.
