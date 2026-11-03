# D3 – Auswertung: Ausreißer und Fehlerfortpflanzung je Schicht bei Tiefe 200, int4 und int8 (beschreibend)

Stand: 03.11.2026, morgens. Auswertung strikt nach `D3_Praeregistrierung.md` (eingefroren 06:44:58). Lauf und Auswertung mit Claude, in Aarons Abwesenheit. Anlass war ein Reddit-Kommentar zu v0.3.

## Kurz

- **Bei Tiefe 200 gibt es keine Ausreißer, weder vorne noch hinten.** Die Kurtosis der Schichteingänge liegt bei 2,8. Das Maximum über dem 99,99-%-Quantil liegt im Median bei 1,00006–1,00007 und in keiner Schicht über 1,005. Jede Schicht von BPreCap reicht genau bis zur Grenze 3,0.
- **int8 ist verlustfrei.** Sind alle Schichten mit 8 Bit gerundet (Gewichte int8), ändert sich der Fehler auf 2000 Testbildern um −0,25 bzw. −0,10 pp.
- **Die Dämpfung hält bei Tiefe 200 (F1), ist aber schwächer als bei 100:**
  - Verstärkung im Median 0,41 bzw. 0,43, bei Tiefe 100 sind es 0,30;
  - Verhältnis 1,4, also nach der Vorab-Schwelle von 1,5 noch „ähnlich“ (F5).
- **Der Fehler häuft sich nicht unbegrenzt an.** Wird bis zur Schicht k mit 4 Bit gerundet, wächst der relative Logit-Fehler r vor allem in den ersten 50 Schichten. Danach flacht er ab:

  | Tiefe | k = 10 | k = 50 | k = 100 | alle |
  |---|---|---|---|---|
  | 200 | 0,19 | 0,37–0,38 | 0,42–0,43 | 0,46–0,47 |
  | 100 | 0,20 | 0,31 | – | 0,33 |

  Bei Tiefe 200 reicht die Anhäufung tiefer, bevor sie aufhört. Das passt zu den höheren 4-Bit-Kosten bei Tiefe 200 (v0.3: +7,3 gegen +3,0 pp).
- **Der lokale Rundungsfehler je Schicht ist über die Tiefe gleich (F2).** ε4 liegt im ersten und im letzten Viertel bei 0,0085–0,0087.
- **Der weiche Bereich hängt mit kleinerem Fehler zusammen (F6, nur beschreibend).** Schichten mit mehr Werten im weichen Sättigungsbereich (|a| > 2,4) haben einen kleineren lokalen 4-Bit-Fehler. Die Spearman-Korrelation über die Schichten beträgt −0,72 bei Tiefe 200 und −0,94 bei Tiefe 100. Der Anteil selbst ist klein, 2,8 % bei Tiefe 200 und 3,4 % bei Tiefe 100.

## Integrität

| Prüfung | Ergebnis |
|---|---|
| Nachbildung | 6 von 6 Läufen bitgleich zu B15 bzw. B10 (val, test, w8a4a_best, gewählte Kalibrierung) |
| Hash-Prüfung | 7 von 7 OK |
| Vorab-Tests | Selbsttest, Rauchtest der Messung, Auswertungstest an künstlichen Daten |
| Laufzeit | 06:45–07:02, 2 Worker, am Netzteil |

Befehl: `python3 -I d3_auswerten.py`, Ausgabe in `logs/auswertung.txt`.

## Lesarten nach der Präregistrierung (bp200, beide Seeds gleich)

| Nr. | Frage | Befund | Zahlen (Seed 42192 / 42193) |
|---|---|---|---|
| F1 | Hält die Dämpfung bei d200? | **hält** | G4 0,41 / 0,43 |
| F2 | Wächst der lokale Fehler nach hinten? | **gleich** | ε4 erstes Viertel 0,0087 / 0,0086, letztes 0,0086 / 0,0085 |
| F3 | Ausreißer hinten? | **keine** | Kurtosis 2,83 → 2,80 bzw. 2,83 → 2,79; Max/99,99 % 1,0002 |
| F4 | Ist int8 verlustfrei? | **verlustfrei** | fp32 46,1 / 45,2 %, int8 45,9 / 45,1 % |
| F5 | d200 gegen d100 | **ähnlich** in allen drei Kennzahlen | G4 1,39 / 1,40-fach, ε4 1,06-fach, r_all4 1,35 / 1,48-fach |
| F6 | weicher Bereich gegen ε4 | Spearman −0,67 / −0,77 | d100: −0,95 / −0,93 |

F5 liegt knapp unter der Schwelle. r_all4 hat bei Seed 42193 1,48. „Ähnlich“ ist hier also ein Grenzfall. Die Anhäufung zeigt, dass Tiefe 200 tatsächlich mehr Fehler sammelt.

## Fortpflanzung einzelner Rundungen (G4 und Rausch-Verstärkung, Seed 42192)

| k | 2 | 10 | 25 | 50 | 100 | 150 | 199 |
|---|---|---|---|---|---|---|---|
| bp200 G4 | 0,69 | 0,60 | 0,58 | 0,41 | 0,19 | 0,16 | 0,29 |
| bp100 G4 (k = 2, 10, 25, 50, 75, 99) | 0,72 | 0,68 | 0,36 | 0,18 | 0,15 | 0,23 | |

Rundungs- und Rausch-Verstärkung stimmen auf 0,01–0,04 überein. Bei Tiefe 200 wird ein Fehler aus Schicht 25 weniger gedämpft (0,58) als bei Tiefe 100 (0,36).

## SELU bei Tiefe 200 (nur beschreibend)

| Kennzahl | Seed 42192 (trainiert nicht) | Seed 42193 (trainiert) |
|---|---|---|
| Testfehler fp32 | 81,8 % | 57,2 % |
| Kurtosis | 2,4 | 2,4 |
| Max/99,99 %, erstes Viertel | 1,13 | 1,19 |
| Betragsmaximum je Schicht | 1,6–5,7 | 1,6–5,7 |
| G4 | 1,03 | 0,30 |
| Rausch-Verstärkung | 0,0001 | 0,30 |
| int8 gegen fp32 | ±0 | +0,6 pp |

- **Seed 42193:** Auch SELU hat leichte Ränder. Ausreißer gibt es wenig, aber vorne etwas mehr als bei BPreCap: Das Maximum über dem 99,99-%-Quantil liegt bei 1,19, das Betragsmaximum je Schicht wechselt zwischen 1,6 und 5,7.
- **Seed 42192:** Das Netz ist praktisch tot, die Logits reagieren kaum auf Störungen.

## Deutung

- **Zur Frage des Kommentators, „Ist es die weiche Sättigung nahe 3?“:** Teilweise ja, das ist aber nicht der ganze Grund.
  - **Was für die Sättigung spricht:**
    - Jede Schicht ist durch 3 begrenzt, es gibt keine Ausreißer, und int8 ist verlustfrei.
    - Schichten mit mehr Werten im weichen Bereich runden genauer (F6).
  - **Was gegen sie als einzigen Grund spricht:**
    - Nur etwa 3 % der Werte liegen im weichen Bereich.
    - Mit dem PreCap-Kern war ein harter Clip an derselben Grenze gleich gut (B13).
    - SELU hat ohne obere Grenze ähnlich leichte Ränder.
  - **Ehrliche Kurzform:** Die feste Grenze verhindert Ausreißer. Dass der Fehler sich nicht anhäuft, liegt an der Dämpfung über die Tiefe. Beides zeigt sich auch bei Tiefe 200.
- **Warum die 4-Bit-Kosten bei Tiefe 200 steigen:** Der lokale Fehler wächst nicht. Er wird aber über mehr Schichten weitergereicht, bevor die Dämpfung greift (Anhäufung bis etwa Schicht 100 statt bis 25–50).
- **Für v0.3.1:** Diese Antwort ist ein kleiner, sauberer Nachtrag zur Diagnose D1, mit Tabelle für Tiefe 200 und int8.

## Abgleich mit den Vorab-Erwartungen

| Größe | Erwartung | Ergebnis |
|---|---|---|
| Nachbildung bitgleich | 0,9 | ja, 6 von 6 |
| F1 „hält“ | 0,55 | ja |
| F2 „wächst hinten“ | 0,35 | nein, „gleich“ |
| F3 „Ausreißer hinten“ | 0,15 | nein |
| F4 int8 verlustfrei | 0,8 | ja |
| F5 G bei 200 größer | 0,5 | nein, „ähnlich“ (1,4-fach, Grenzfall) |

## Grenzen

- Beschreibend, 2 Seeds je Netz, gemessen am Ende statt am Stopp-Punkt.
- Nur MLPs, CIFAR-10, TAT-Rezept, AdamW.
- int8 nur mit Min/Max-Kalibrierung.
- F6 ist eine Korrelation über die Schichten eines Netzes, keine Ursache.

**Vertrauen:**
- 0,9, dass BPreCap bei Tiefe 200 keine Ausreißer hat und int8 verlustfrei ist;
- 0,75, dass die höheren 4-Bit-Kosten bei Tiefe 200 von der tiefer reichenden Anhäufung kommen und nicht von einem größeren lokalen Fehler.
