# D3 – Präregistrierung: Ausreißer und Fehlerfortpflanzung je Schicht bei Tiefe 200, int4 und int8 (beschreibend, CIFAR-10, privat)

Stand: 03.11.2026, morgens, vor dem Start. Die Freeze-Zeit steht in `meta/frozen_at.txt`. PreCap und BPreCap sind Aarons Ideen. Das Design ist mit Claude entworfen und wird in Aarons Abwesenheit gestartet (Leitlinie fürs Alleinarbeiten, Freigabe vom 03.11. bis 18:00). Kandidat für v0.3.1.

## Anlass

- **Reddit-Kommentar zu v0.3:** Halten sich die int8-Ausreißer je Schicht bei Tiefe 200 in Grenzen? Ist es die weiche Sättigung nahe 3, die verhindert, dass sich der Clipping-Fehler aufsummiert?
- **D1 (v0.3, Tabelle 4)** hat das nur bei Tiefe 100 und nur bei 4 Bit gemessen. Ergebnis für BPreCap: leichte Ränder (Kurtosis 2,75) und eine Dämpfung früher Fehler (Verstärkung 0,30).
- **Laut v0.3 (B15)** steigen die 4-Bit-Kosten von BPreCap bei Tiefe 200 auf +7,3 pp, bei Tiefe 100 sind es +3,0 pp. Der Grund ist nicht gemessen.

## Fragen (beschreibend, keine Signifikanztests, 2 Seeds je Netz)

| Nr. | Frage | Kennzahl | Lesart (vorab) |
|---|---|---|---|
| F1 | Hält die Dämpfung bei Tiefe 200? | `gain4_median` von bp200 (Median über k ∈ {2, 10, 25, 50, 100, 150, 199}) | ≤ 0,5 „hält“; 0,5–1,0 „schwächer“; > 1,0 „verstärkt“ |
| F2 | Wächst der lokale Fehler nach hinten? | `eps4_q4 / eps4_q1` von bp200 (Median der Schichten im letzten bzw. ersten Viertel) | ≥ 1,5 „wächst hinten“; ≤ 1/1,5 „sinkt hinten“; sonst „gleich“ |
| F3 | Entstehen Ausreißer hinten? | `kurt_q4` und `max_over_p9999_q4` gegen `_q1` von bp200 | Kurtosis q4 ≥ 2 · q1 oder > 5: „Ausreißer hinten“; sonst „keine“ |
| F4 | Ist int8 praktisch verlustfrei? | `err_all8_2000 − err_fp32_2000` (2000 Testbilder, alle Schichten gerundet, Gewichte int8) | ≤ 0,5 pp „verlustfrei“; sonst Größe berichten |
| F5 | Tiefe 200 gegen 100 | `gain4_median`, `eps4_median`, `r_all4` von bp200 gegen bp100 | Verhältnis ≥ 1,5 „bei 200 größer“; sonst „ähnlich“ |
| F6 | Rolle des weichen Bereichs | `frac_soft` (Anteil der Schichteingänge mit |a| > 2,4 = ρM) je Schicht, gegen `eps4` je Schicht | nur Beschreibung: Spearman-Korrelation über die Schichten, Median der beiden Seeds |

Eine Lesart gilt nur, wenn beide Seeds dieselbe ergeben. Sonst ist sie „uneinheitlich“. SELU bei Tiefe 200 (se200) wird mit denselben Kennzahlen beschrieben. Lesarten gibt es nur für bp200. Nach B15 trainiert SELU bei Tiefe 200 zweigeteilt. Die Seeds 42192 und 42193 werden ohne Auswahl genommen, es sind die ersten beiden der B15-Bestätigung. In B15 trainiert se200 bei 42192 nicht (Val-Fehler 82,9 %) und bei 42193 schon (58,2 %).

## Netze (keine Änderung am Training, nur Nachtraining und Messung)

| Name | Form | Tiefe | LR / WD | Seeds | Vergleichszeile |
|---|---|---|---|---|---|
| bp200 | kL2B_tr | 200 | 5e-5 / 0,3 | 42192, 42193 | B15 |
| se200 | selu_tr | 200 | 2e-4 / 0,3 | 42192, 42193 | B15 |
| bp100 | kL2B_tr | 100 | 1e-4 / 0,3 | 42168, 42169 | B10 (gleiche Seeds wie D1) |

- **Code:** `studien/b15/b15_run.py` bzw. `studien/b10/b10_run.py`, unverändert als Modul geladen. Der Hash wird gegen den jeweiligen Freeze geprüft. Die Trainingsschleife ist dieselbe wie in D1.
- **Pflichtprüfung:** `val_err_fp32`, `test_err_fp32`, `test_err_w8a4a_best` und die gewählte Kalibrierung müssen bitgleich zur Vergleichszeile sein. Sonst ist der Lauf ungültig.

## Kennzahlen

Gemessen wird am Ende (Schritt 6880). Es sind dieselben Definitionen wie in D1, dazu:

- **8 Bit:** asymmetrisch, Kalibrierung Min/Max. **4 Bit:** asymmetrisch mit der Kalibrierung, die die Wahlregel der Studie bestimmt.
- **Je Schicht:**
  - Kurtosis, Max / 99,99-%-Quantil, Anteil nahe am Maximum, `frac_soft`, RMS, Betragsmaximum;
  - ε4 und ε8 (relatives MSE, nur diese Schicht gerundet).
- **Fortpflanzung** je k:
  - r und Verstärkung G = r / √ε für 4 und 8 Bit;
  - Anteil geänderter Vorhersagen;
  - Rausch-Verstärkung (σ = 0,01 · RMS).
- **Anhäufung:** Die Eingänge der Schichten 1..k werden gerundet (k aus der Liste oben, dazu alle Schichten), die Gewichte bleiben fp32. Gemessen werden r und Testfehler auf 2000 Bildern, für 4 und 8 Bit.
- **Alle Schichten gerundet, Gewichte int8:** r_all und Testfehler auf 2000 Bildern.

## Vorab-Erwartung

| Größe | Erwartung |
|---|---|
| Nachbildung bitgleich (6 von 6) | 0,9 |
| F1 bp200 „hält“ | 0,55 |
| F2 „wächst hinten“ | 0,35 |
| F3 „Ausreißer hinten“ | 0,15 |
| F4 int8 „verlustfrei“ bei bp200 | 0,8 |
| F5 G bei 200 größer | 0,5 |

## Grenzen

- Beschreibend, 2 Seeds je Netz, gemessen am Ende statt am Stopp-Punkt.
- Nur MLPs, CIFAR-10, TAT-Rezept, AdamW-Regime.
- int8 nur mit Min/Max-Kalibrierung.
- G mischt Fortpflanzung und Empfindlichkeit der Logits. Die Rausch-Verstärkung ist die Gegenprobe.

## Ablauf und Tests

- **Selbsttest** `d3_diag.py --selfcheck`: ok.
  - Rundungsfehler der Gleichverteilung: 4 Bit 0,004441 gegen theoretisch 0,004444; 8 Bit 1,5380e-5 gegen 1,5379e-5.
  - `fwd_mod` mit Identität ist bitgleich.
  - 201 Schichteingänge, 6 Läufe, alle Vergleichszeilen gefunden.
- **Rauchtest der Messung** `test/rauch_messung.py`: untrainierte Netze, alle Kennzahlen endlich, 10–19 s je Netz.
- **Auswertung** `d3_auswerten.py`, an künstlichen Daten getestet (`test/test_auswertung.py`).
- **Freeze:**
  - `d3_diag.py`, `d3_auswerten.py`;
  - die Tests;
  - diese Datei;
  - `b10_run.py`, `b15_run.py`.
- **Start:** `launch_d3.sh`, 2 Worker in `screen` (d3w0, d3w1) mit `caffeinate -i`, nur am Netzteil. Geschätzt 25–35 min.
