# D1 – Auswertung: Warum zahlt TAT bei 4 Bit mehr als BPreCap? (CIFAR-10, d100, beschreibend)

Stand: 01.11.2026, 06:20. Die Auswertung folgt `D1_Praeregistrierung.md`. Die zwei Erweiterungen vor dem Freeze (SELU, harter linearer Clip) sind dort offen vermerkt und **nur beschreibend**. PreCap, die Tiefenfamilie und Lagma sind Aarons Ideen. Lauf und Auswertung habe ich mit Claude gemacht.

## Kurz

- **Nachbildung exakt:** Alle 12 nachtrainierten Netze sind **bitgleich** zu den Originalläufen aus B10, B10b, B12 und B13 (Val- und Testfehler fp32 sowie w8a4a_best mit gleicher Kalibrierwahl).
- **Vorab-Lesart (TAT a gegen BPreCap und LB): „uneinheitlich“.**
  - Der lokale Rundungsfehler ist bei TAT in beiden Seeds 4,7- bis 6,0-mal größer.
  - Der Verstärkungsfaktor liegt bei 1,28–1,34 in Seed 42168 und bei 1,59–1,60 in Seed 42169. Die Schwelle 1,5 wird also einmal unter- und einmal überschritten.
  - Nach der strengen Regel ist das „uneinheitlich“.
- **In der Sache ist das Bild klar:**
  - **TAT (Leaky ReLU):** Die Verteilung hat schwere Ausläufer. Die Kurtosis je Schicht liegt im Median bei 12,5 mit WD und 29 ohne WD, bei BPreCap bei 2,7 und bei LB bei 2,3. Deshalb rundet 4 Bit jede Schicht etwa 5-mal ungenauer (**E1**). Die Fortpflanzung isotropen Rauschens über die Tiefe ist gleich (Faktor 0,91–0,99). Die Fortpflanzung des Rundungsfehlers ist etwas größer (1,3–1,6).
  - **Harter linearer Clip gegen weiche Hülle** (gleicher Kern, beschreibend): Der lokale Rundungsfehler ist gleich (1,17). Der Fehler wird aber über die Tiefe **gut doppelt so stark weitergereicht** (2,1–2,3, beide Messwege, beide Seeds) (**E2**). Die weiche Hülle dämpft, der harte Clip nicht.
  - **SELU** (beschreibend): leichte Ausläufer (Kurtosis 2,5), lokaler Fehler nur 1,3-mal der von BPreCap. Es dämpft am stärksten (Verstärkung 0,20 gegen 0,30 bei BPreCap).
- **Folge für v0.3:** Der kleine 4-Bit-Preis glatt sättigender Formen hat zwei messbare Gründe:
  - **leichte Ausläufer der Verteilung:** kleiner lokaler Rundungsfehler;
  - **Dämpfung über die Tiefe:** kleinere Fortpflanzung.
  - TAT verliert vor allem durch das Erste, der harte lineare Clip durch das Zweite. Der absolute Bereich spielt keine Rolle (B10b).

## Integrität

| Prüfung | Ergebnis |
|---|---|
| Läufe | 12 von 12 ok, alle `valid` (Nachbildung bitgleich) |
| Hash-Prüfung | 7 von 7 OK (`d1_diag.py`, `d1_auswerten.py`, Präregistrierung, Rauchtest, `b10_run.py`, `b12_run.py`, `b13_run.py`) |
| Startprüfungen | Hashes von `b10_run.py`, `b12_run.py`, `b13_run.py` gegen die jeweiligen Studien-Freezes, Selbsttest, Rauchtest der Messung: ok |
| Laufzeit | 05:57–06:11 |
| Sicherung | `sicherung/d1_2026-11-01/`, 20 von 20 Dateien OK |

Befehl: `python3 -I d1_auswerten.py`, Ausgabe in `logs/auswertung.json`.

## Kennzahlen (Mediane über die Schichten, je Seed)

| Netz | Seed | Kurtosis | Max/99,99 % | lokaler Fehler ε | Verstärkung G (Rundung) | Verstärkung (Rauschen) | Fehler fp32 / alle Schichten 4 Bit (2000 Bilder) |
|---|---|---|---|---|---|---|---|
| BPreCap | 42168 | 2,75 | 1,00 | 0,0080 | 0,30 | 0,29 | 46,1 / 49,1 % |
| BPreCap | 42169 | 2,74 | 1,00 | 0,0080 | 0,30 | 0,29 | 46,2 / 48,6 % |
| LB (weich) | 42168 | 2,34 | 1,00 | 0,0063 | 0,31 | 0,30 | 47,3 / 50,6 % |
| LB (weich) | 42169 | 2,35 | 1,00 | 0,0063 | 0,30 | 0,30 | 48,8 / 50,4 % |
| LBh (hart) | 42168 | 2,55 | 1,00 | 0,0074 | 0,67 | 0,68 | 47,1 / 53,2 % |
| LBh (hart) | 42169 | 2,56 | 1,00 | 0,0074 | 0,70 | 0,70 | 48,6 / 58,4 % |
| SELU | 42168 | 2,48 | 1,13 | 0,0103 | 0,20 | 0,20 | 46,6 / 47,9 % |
| SELU | 42169 | 2,48 | 1,15 | 0,0102 | 0,20 | 0,21 | 46,4 / 50,5 % |
| TAT a (WD 0,3) | 42168 | 12,5 | 1,45 | 0,0377 | 0,40 | 0,28 | 46,7 / 55,5 % |
| TAT a (WD 0,3) | 42169 | 12,6 | 1,38 | 0,0381 | 0,49 | 0,29 | 47,4 / 57,2 % |
| TAT (B10-Wahl, WD 0) | 42168 | 29,0 | 1,44 | 0,0575 | 0,69 | 0,39 | 46,0 / 62,7 % |
| TAT (B10-Wahl, WD 0) | 42169 | 29,7 | 1,58 | 0,0628 | 0,63 | 0,35 | 46,5 / 60,6 % |

- ε = relatives MSE der 4-Bit-Rundung je Schichteingang (asymmetrisch, je Lauf gewählte Kalibrierung).
- G = relativer Logit-Fehler, wenn nur eine Schicht gerundet wird, geteilt durch √ε. Median über die Schichten 2, 10, 25, 50, 75, 99.
- **Verlauf über die Tiefe:**
  - Bei BPreCap, LB und SELU fällt G von etwa 0,7–0,9 (frühe Schichten) auf 0,1–0,2 (Schichten 50–75). Frühe Fehler werden auf dem Weg gedämpft.
  - Beim harten Clip bleibt G bis Schicht 50 bei 0,57–0,85.
  - Bei TAT ohne WD steigt G in den Schichten 50–75 sogar wieder auf 0,7–0,9.

## Lesart nach der Präregistrierung

| Vergleich | Seed 42168 | Seed 42169 | Befund |
|---|---|---|---|
| TAT a gegen BPreCap (vorab) | f_ε 4,70, f_G 1,34 → E1 | f_ε 4,78, f_G 1,59 → zwischen | **uneinheitlich** |
| TAT a gegen LB (vorab) | f_ε 5,99, f_G 1,28 → E1 | f_ε 6,04, f_G 1,60 → zwischen | **uneinheitlich** |
| TAT a gegen SELU (beschreibend) | f_ε 3,66, f_G 2,03 → beide | f_ε 3,73, f_G 2,44 → beide | beide |
| LBh gegen LB (beschreibend) | f_ε 1,18, f_G 2,15 → E2 | f_ε 1,17, f_G 2,32 → E2 | **E2** |

- Nach der strengen Regel ist der vorab festgelegte Befund „uneinheitlich“. Der Grund ist allein f_G, das in Seed 42169 knapp über 1,5 liegt.
- f_ε ist in beiden Seeds weit über der Schwelle 2. **Dass E1 bei TAT trägt, ist eindeutig.** Offen ist nur, ob E2 zusätzlich einen kleinen Anteil hat.
- **Gegenprobe Rausch-Verstärkung:**
  - TAT gegen BPreCap und LB: 0,91–0,99. Isotropes Rauschen pflanzt sich bei TAT nicht stärker fort.
  - Der leicht höhere G-Wert von TAT kommt also aus der Struktur des Rundungsfehlers. Wahrscheinlich treffen die Ausreißer-Kanäle die Logits stärker.
  - Beim harten Clip stimmen Rundungs- und Rausch-Verstärkung überein (2,2–2,4). Dort ist die Dämpfung echt geringer.

## Einordnung

- **Passt zu He et al. 2024** (Ausreißer-Merkmale, große Kurtosis, LR-abhängig): TAT ohne WD hat Kurtosis 29, mit WD 12,5. Der 4-Bit-Preis sinkt entsprechend (B10b).
- **Warum der harte Clip nicht dämpft:** Innerhalb der Grenze ist er linear mit Steigung β ≈ 1,02. Ein Rundungsfehler läuft unverändert weiter, bis er an eine gesättigte Einheit kommt. Die weiche Hülle hat schon ab ρM = 0,8·M Steigung unter 1. Ein Teil jedes Fehlers wird in jeder Schicht gestaucht.
- **Für die Kernaussage von v0.3:** „Glatt sättigende Formen (BPreCap, LB mit weicher Hülle, SELU) haben im TAT-Rezept leichte Ausläufer und dämpfen Rundungsfehler über die Tiefe. Leaky ReLU (TAT) entwickelt schwere Ausläufer (Kurtosis 12–30), ein harter Clip dämpft nicht. Beides erhöht den 4-Bit-Preis.“

## Grenzen

- Beschreibend, n = 2 Seeds je Netz, keine Statistik.
- Gemessen am Ende (Schritt 6880), nicht am Stopp-Punkt der Studien.
- Nur CIFAR-10, d100, asymmetrisch 4 Bit, Messung auf 2000 Testbildern bzw. 1000 Kalibrierbildern.
- Die Vergleiche mit SELU und LBh sind nach den Ergebnissen von B12 und B13 hinzugekommen und deshalb nur beschreibend.
- G mischt die Fortpflanzung mit der Empfindlichkeit der Logits.

**Vertrauen:** 0,85, dass bei TAT die schweren Ausläufer (E1) den Hauptteil des 4-Bit-Preises ausmachen. 0,75, dass die weiche Hülle gegenüber dem harten Clip über Dämpfung (E2) wirkt. 0,9, dass die Messung technisch korrekt ist: Nachbildung bitgleich, Selbsttest gegen die Theorie, Rundungs- und Rausch-Verstärkung stimmen bei den glatten Formen überein.
