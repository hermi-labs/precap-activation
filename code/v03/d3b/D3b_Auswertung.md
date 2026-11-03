# D3b – Auswertung (beschreibend): Warum erhöht der LR-Anlauf BPreCaps 4-Bit-Preis bei Tiefe 200?

Stand: 03.11.2026, 13:30, mit Claude. Ausgewertet nach `D3b_Praeregistrierung.md` (Freeze 13:16). Läufe 13:17–13:26. PreCap und BPreCap sind Aarons Ideen.

## Kurzfassung

- **Alle 4 Netze sind bitgleich zur V1c-Bestätigung.** Die Hashes sind 11 von 11 OK.
- **Einziger klarer Befund (L3):** Mit Anlauf liegen deutlich weniger Werte im weichen Bereich der Hülle (|a| > 2,4). Es sind 25–30 % weniger, in beiden Seeds (Verhältnis 0,75 und 0,71).
- **Logit-Fehler (L4) „gleich“** nach der Vorab-Schwelle. Er ist aber in beiden Seeds etwas größer (+7,7 % und +8,1 %, Schwelle 10 %).
- **Lokaler Fehler (L1) „uneinheitlich“:** +7 % in einem Seed, +42 % im anderen.
- **Dämpfung (L2) „gleich“:** +6 % und +2 %.
- **Lesart:** Das passt zur Vermutung aus V1c, ist mit 2 Seeds aber nur ein Hinweis. Mit Anlauf nutzt BPreCap den weichen Bereich weniger, die Schichteingänge runden etwas schlechter, und der Fehler am Ausgang wächst leicht. Ein starker Einzelmechanismus zeigt sich nicht.

## Integrität

| Prüfung | Ergebnis |
|---|---|
| Hashes (`shasum -a 256 -c meta/frozen_sha256.txt`) | 11 von 11 OK |
| Netze bitgleich zu V1c (val, test, w8a4a_best, Kalibrierung) | 4 von 4 |
| **Abweichung vom Plan** | Das eingefrorene Auswertungsskript brach bei SELU mit einer Division durch 0 ab, weil der Anteil im weichen Bereich dort 0 ist. Neue Version `d3b_auswerten_v2.py`: einzige Änderung, ein Nenner 0 ergibt „nicht definiert“. Ein eigener Test `test/test_auswertung_v2.py` ist eine Kopie des eingefrorenen Tests: AUSWERTUNG OK. Eingefroren in `meta/frozen_sha256_v2.txt`. BPreCap ist nicht betroffen |

Befehl: `python3 -I d3b_auswerten_v2.py > analysis/d3b_result.json`.

## Lesarten BPreCap (Verhältnis mit Anlauf / ohne Anlauf)

| Nr. | Größe | Seed 42192 | Seed 42193 | Befund |
|---|---|---|---|---|
| L1 | ε4 Median | 1,07 (gleich) | 1,42 (größer) | uneinheitlich |
| L2 | G4 Median | 1,06 | 1,02 | gleich |
| L3 | Anteil weich, Median | 0,75 | 0,71 | **weniger im weichen Bereich** |
| L4 | r_all4 | 1,08 | 1,08 | gleich (knapp unter 1,1) |

## Rohwerte BPreCap d200

| Netz | ε4 | G4 | Anteil weich | r_all4 | Fehler fp32 / 4 Bit (2000 Bilder) | 4-Bit-Preis |
|---|---|---|---|---|---|---|
| ohne Anlauf, 42192 | 0,0085 | 0,41 | 2,75 % | 0,452 | 46,1 / 53,7 | +7,6 |
| ohne Anlauf, 42193 | 0,0085 | 0,43 | 2,78 % | 0,469 | 45,2 / 55,0 | +9,8 |
| mit Anlauf, 42192 | 0,0092 | 0,43 | 2,06 % | 0,486 | 44,3 / 54,4 | +10,2 |
| mit Anlauf, 42193 | 0,0121 | 0,44 | 1,96 % | 0,508 | 46,4 / 55,8 | +9,4 |

- **Anhäufung r4:** Mit Anlauf wächst der Fehler hinter Schicht 100 etwas weiter. Von Schicht 100 bis zum Ausgang +0,047/+0,042 gegen +0,037/+0,038 ohne Anlauf. Bis Schicht 50 ist er gleich oder kleiner.
- **4-Bit-Preis auf 2000 Bildern:** Er ist zwischen den beiden Seeds nicht einheitlich (42193 ohne Anlauf schon +9,8). Der Unterschied aus V1c (+1,64 pp, 12 Seeds) ist mit 2 Netzen und 2000 Bildern nicht auflösbar. Deshalb gibt es hier keine Aussage über den Preis selbst.

## SELU (nur Einordnung)

- Mit Anlauf (LR 3e-4 statt 2e-4) ist der Logit-Fehler bei 4 Bit deutlich größer (Verhältnis 1,71 und 1,46), ebenso der lokale Fehler (1,32 und 1,12).
- Der Anteil im Bereich |a| > 2,4 ist bei SELU fast 0 und für SELU keine sinnvolle Größe.
- Dass SELU in V1c bei 4 Bit nichts gewinnt, obwohl es in fp32 3,5 pp besser wird, passt zum größeren Logit-Fehler.

## Vorab-Erwartungen gegen Ergebnis

| Größe | Erwartung | Ergebnis |
|---|---|---|
| alle 4 Netze bitgleich | 0,9 | ja |
| L4 „größer“ | 0,55 | nein („gleich“, +8 %) |
| L1 „lokal größer“ | 0,35 | uneinheitlich |
| L2 „Dämpfung schwächer“ | 0,35 | nein |
| L3 „weniger“ | 0,35 | ja |

## Deutung nach der Vorab-Regel

- **Formal:** L4 „gleich“. Nach der Regel kommt der höhere Preis dann nicht aus einem größeren Logit-Fehler, sondern aus einer größeren Empfindlichkeit des Netzes.
- **Ehrlich ergänzt:**
  - Der Logit-Fehler ist in beiden Seeds um 8 % größer, also nur knapp unter der gesetzten Schwelle.
  - Der Rückgang des Anteils im weichen Bereich ist deutlich und einheitlich.
  - Zusammen mit D3 F6 (mehr weich ↔ kleinerer lokaler Fehler, Spearman −0,72) ist die naheliegende Lesart: Der Anlauf lässt die Gewichte so wachsen, dass weniger Werte die Hülle erreichen. Die Hülle dämpft dann weniger, und der Fehler wird etwas größer.
- **Nicht gezeigt:** dass das die Ursache der +1,64 pp ist. Dafür bräuchte es einen Eingriff, etwa einen kürzeren Anlauf oder ein größeres M, und mehr Seeds.

## Grenzen

- 2 Seeds, nur beschreibend, gemessen am Ende (6880) auf 2000 Testbildern.
- Schwellen gesetzt, nicht hergeleitet.
- Die Auswertung lief mit v2, dokumentiert oben.

Sicherung: `../../sicherung/d3b_2026-11-03/`.
