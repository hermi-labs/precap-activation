# B5 – Auswertung: Hält der B3-Befund bei Tiefe 200 nach Nachtuning der Lernrate?

Stand: 31.10.2026, morgens. Die Auswertung folgt strikt `B5_Praeregistrierung.md`. PreCap, die Tiefenfamilie und Lagma sind Aarons Ideen. Lauf und Auswertung habe ich mit Claude gemacht.

## Kurz

- **Erfolg.** BPreCap ist bei d200 auch nach Nachtuning in fp32 besser als LB: −2,57 pp, 12/0, p = 0,0005.
- **Der B3-Befund war kein LR-Artefakt.** LB wählt jetzt 7e-6 und gewinnt dadurch etwa 1,5 pp gegenüber B3. Der Abstand schrumpft von −4,79 auf −2,57 pp, bleibt aber deutlich.
- **Neu:** Bei d200 ist BPreCap auch bei 4 Bit besser als LB, symmetrisch −2,39 pp und asymmetrisch −3,57 pp (jeweils 12/0). In B3 war das „kein Effekt“.
- **Gegen TAT:** BPreCap ist bei symmetrischen 4 Bit besser (−1,69 pp, 11/1). In fp32 (+9,4 pp) und bei asymmetrischen 4 Bit (+2,7 pp) ist es schlechter.

## Integrität

| Prüfung | Ergebnis |
|---|---|
| Läufe | Tuning 48 von 48 ok, Bestätigung 49 von 49 ok |
| Hash-Prüfung | 18 von 18 OK |
| Kontrolle gegen B1 | kL2B MNIST d100 Seed 42060: bitgleich |
| Kontrolle gegen B3 | Die 12 Tuning-Läufe mit 1e-5 sind bitgleich zum B3-Tuning (12/12) |
| Schlaf | Größte Lücke zwischen zwei Läufen: 108 s bzw. 114 s. Kein Schlaf |
| Laufzeit | 03:24–04:26 |
| Sicherung | `sicherung/b5_2026-10-31/`, 15 von 15 Dateien OK |

Befehl: `.venv/bin/python -I b5_analyze.py ~/…/studien/b5 ~/…/studien/b1 ~/…/studien/b3`, Ergebnis in `analysis/b5_result.json`.

## LR-Tuning (mediane Val-Genauigkeit nach 1720 Schritten)

| Form | 3e-6 | 5e-6 | 7e-6 | 1e-5 | Wahl |
|---|---|---|---|---|---|
| kL2B | 37,9 | 38,8 | 39,1 | **39,6** | 1e-5 |
| LB | 35,1 | 35,7 | **36,4** | 34,3 | 7e-6 |
| kL2 | 39,2 | 41,2 | 42,3 | **43,5** | 1e-5 |

- Keine Form liegt am unteren Rand (3e-6).
- Weil B3 über 1e-5 bei allen drei Formen nur schlechtere Werte hatte, ist die Wahl auch über beide Raster zusammen gültig. Das stand so in der Präregistrierung.
- TAT läuft mit 2e-5 aus B3.

## Primär und sekundär (d200, 12 Seeds)

| Test | Vergleich | Median | Siege/Niederlagen | p bzw. Holm | Urteil |
|---|---|---|---|---|---|
| **Primär** | kL2B − LB, fp32 | **−2,57 pp** | 12/0 | 0,0005 | **besser** |
| S1 | kL2B − LB, w4 | −2,39 | 12/0 | 0,004 | besser |
| S2 | kL2B − LB, w4a | −3,57 | 12/0 | 0,004 | besser |
| S3 | kL2 − LB, fp32 | −7,93 | 12/0 | 0,004 | besser |
| S4 | kL2B − kL2, fp32 | +4,96 | 0/12 | 0,004 | schlechter |
| S5 | kL2B − TAT, fp32 | +9,39 | 0/12 | 0,004 | schlechter |
| S6 | kL2B − TAT, w4a | +2,66 | 0/12 | 0,004 | schlechter |
| S7 | LB − TAT, w4a | +5,99 | 0/12 | 0,004 | schlechter |
| S8 | kL2B − TAT, w4 | **−1,69** | 11/1 | 0,004 | **besser** |

## Mediane (Testfehler in %, d200)

| Form | LR | fp32 | w8a4 | w8a4a | w8a3 | Bereich | Sättigung |
|---|---|---|---|---|---|---|---|
| kL2B | 1e-5 | 61,3 | 70,2 | 69,0 | 86,6 | 3,0 | 0,06 |
| LB | 7e-6 | 64,1 | 72,8 | 72,6 | 87,4 | 2,45 | 0,10 |
| kL2 | 1e-5 | 56,3 | 79,7 | 79,3 | 89,8 | 7,7 | – |
| TAT | 2e-5 | 52,3 | 72,4 | 66,6 | 89,6 | 5,1 | – |

- Die Ordnung aus p8 gilt: fp32 TAT < kL2 < kL2B < LB.
- Die Mediane decken sich mit B3 d200 (kL2B 61,1, kL2 56,3, TAT 52,0). B5 ist damit zugleich eine Wiederholung von B3 auf neuen Seeds.

## Abgleich mit der Vorab-Erwartung

| Größe | Erwartung | Ergebnis |
|---|---|---|
| Primär „besser“ | 0,75 | eingetreten |
| LB wählt eine LR unter 1e-5 | 0,6 | ja (7e-6) |
| kL2B wählt eine LR unter 1e-5 | 0,4 | nein |
| eine Form bei 3e-6 | 0,15 | nein |
| S3 „besser“ | 0,85 | ja |
| S6 „schlechter“ | 0,65 | ja |
| S8 „besser“ | 0,35 | ja |
| p8-Ordnung | 0,65 | ja |
| Kontrollen bitgleich | 0,95 | ja |

## Deutung nach den Vorab-Regeln

- **Erfolg:** Der fp32-Vorteil von BPreCap bei d200 ist kein LR-Artefakt. Der B3-Befund gilt in beiden Tiefen.
- **S8 „besser“:** Bei symmetrischen 4 Bit und d200 ist BPreCap besser als TAT. Bei asymmetrischen 4 Bit bleibt TAT vorn. Der Grund ist der schiefe, unbeschränkte Bereich von TAT: Symmetrisch kostet er viel, asymmetrisch wenig.
- **Größe des Effekts:** Mit 7e-6 holt LB etwa die Hälfte des Abstands aus B3 auf (−4,79 auf −2,57 pp). Der B3-Wert war also teils ein LR-Effekt, aber nicht ganz.

**Vertrauen: 0,85**, dass BPreCap bei d200 in diesem Aufbau in fp32 und bei 4 Bit besser ist als LB.

## Grenzen

- Wie B3: MLP ohne Faltungen, kurzes Budget (etwa 5 Epochen), ein Datensatz, TAT im eigenen Rezept (dazu B6).
- TAT wurde nicht nachgetunt (in B3 nicht am Rand).
