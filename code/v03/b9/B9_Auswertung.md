# B9 – Auswertung: BPreCap gegen TAT im gleichen Rezept bei langem Training mit frühem Stoppen (CIFAR-10, d100)

Stand: 31.10.2026, 12:25. Die Auswertung folgt strikt `B9_Praeregistrierung.md`. PreCap, die Tiefenfamilie und Lagma sind Aarons Ideen. Lauf und Auswertung habe ich mit Claude gemacht.

## Kurz

- **Erfolg.** Im fairen Vergleich (gleiches Rezept, 4-mal so langes Training, jeder Lauf an seinem besten Validierungspunkt) ist BPreCap bei fairen 4 Bit besser als TAT: w8a4a **−3,76 pp, 12/0, p = 0,0005**.
- **In fp32 gleichauf** (+0,20 pp, 4/8, unklar). Beide liegen am Stopp-Punkt bei 49,6 %.
- **Bei symmetrischen 4 Bit −6,58 pp** (12/0).
- **Der Grund ist sichtbar.** Bei TAT wächst der Bereich der Aktivierungen während des Trainings von 6,2 auf 8,2. Bei BPreCap bleibt er durch die Hülle fest bei 3,0. Der 4-Bit-Preis am Stopp-Punkt beträgt +8,3 pp bei TAT und +4,0 pp bei BPreCap.
- **Anders als bei d200 (B8)** ist bei d100 BPreCap auch bei 4 Bit besser als LB_tr (−1,34 pp, 10/2, knapp). In fp32 liegt es 1,55 pp vorn.

## Integrität

| Prüfung | Ergebnis |
|---|---|
| Läufe | 38 von 38 ok, keine Ausfälle |
| Hash-Prüfung | 19 von 19 OK |
| Kontrolle | kL2B_tr und TAT mit Seed 42120: Messpunkt 1720 in allen 7 Kennzahlen bitgleich zu den Endwerten von B6 |
| Schlaf | `pmset` meldet zwischen 10:48 und 12:12 keinen Schlaf. Die größte Lücke im Log (308 s) entspricht der Dauer eines Laufs (etwa 3,5 bis 5 min bei 6880 Schritten) |
| Laufzeit | 10:48–12:11 |
| Sicherung | `sicherung/b9_2026-10-31/`, 12 von 12 Dateien OK |

Befehl: `.venv/bin/python -I b9_analyze.py ~/…/studien/b9 ~/…/studien/b6`, Ergebnis in `analysis/b9_result.json`.

## Primär und sekundär (d100, 12 Seeds, ES = frühes Stoppen)

| Test | Vergleich | Median | Siege/Niederlagen | p bzw. Holm | Urteil |
|---|---|---|---|---|---|
| **Primär** | kL2B_tr − TAT, w4a, ES | **−3,76 pp** | 12/0 | 0,0005 | **besser** |
| S1 | kL2B_tr − TAT, fp32, ES | +0,20 | 4/8 | 0,85 | unklar |
| S2 | kL2B_tr − TAT, w4, ES | −6,58 | 12/0 | 0,004 | besser |
| S3 | LB_tr − TAT, w4a, ES | −3,29 | 12/0 | 0,004 | besser |
| S4 | kL2B_tr − LB_tr, fp32, ES | −1,55 | 12/0 | 0,004 | besser |
| S5 | kL2B_tr − LB_tr, w4a, ES | −1,34 | 10/2 | 0,048 | besser (knapp) |
| S6 | kL2B_tr − TAT, w4a, Ende 6880 | −4,75 | 12/0 | 0,004 | besser |
| S7 | kL2B_tr − TAT, fp32, Ende 6880 | +0,10 | 5/7 | 0,85 | unklar |
| S8 | LB_tr − TAT, fp32, ES | +1,40 | 0/12 | 0,004 | schlechter |

## Mediane am Stopp-Punkt (Testfehler in %)

| Form | Stopp-Punkte | fp32 | w8a4 | w8a4a | w8a3 | 4-Bit-Preis (w4a − fp32) | Bereich |
|---|---|---|---|---|---|---|---|
| kL2B_tr | 4300–6880, Median 5160 | **49,6** | 55,3 | **53,7** | 74,8 | +4,0 | 3,0 (fest) |
| LB_tr | 3440–6880, Median 5160 | 51,3 | **54,5** | 54,7 | **68,2** | **+3,5** | 2,22 (fest) |
| TAT | 3440–6880, Median 5160 | **49,6** | 62,0 | 57,4 | 85,0 | +8,3 | 6,2 → 8,2 |

## Verlauf über die Messpunkte (Mediane: fp32 / w4a / Bereich)

| Schritt | kL2B_tr | LB_tr | TAT |
|---|---|---|---|
| 860 | 54,3 / 59,1 / 3,0 | 56,4 / 59,9 / 2,22 | 54,1 / 59,6 / 6,22 |
| 1720 | 52,7 / 56,6 | 54,3 / 56,9 | 51,3 / 56,9 / 7,07 |
| 2580 | 51,1 / 55,2 | 52,3 / 55,6 | 50,3 / 56,8 / 7,19 |
| 3440 | 50,4 / 54,4 | 52,4 / 55,6 | 50,0 / 56,8 / 7,41 |
| 4300 | 50,1 / 54,3 | 51,9 / 55,0 | 49,9 / 57,8 / 7,58 |
| 5160 | 50,2 / 54,3 | 51,7 / 55,4 | 49,7 / 58,0 / 7,84 |
| 6020 | 49,8 / 53,7 | 51,5 / 55,3 | 50,2 / 59,2 / 8,14 |
| 6880 | 50,2 / 54,4 | 51,6 / 55,3 | 49,8 / 59,1 / 8,22 |

**So lese ich den Verlauf:**
- Bei TAT wird fp32 bis etwa 5160 Schritte noch leicht besser.
- Zugleich wird TAT bei 4 Bit schlechter (56,9 → 59,1), weil der Bereich wächst.
- Bei BPreCap und LB werden fp32 und 4 Bit gemeinsam besser, der Bereich bleibt fest.
- Der 4-Bit-Vorsprung von BPreCap auf TAT wächst mit dem Training: 0,3 pp bei 1720 Schritten, 3,8 pp am Stopp-Punkt, 4,7 pp am Ende.

## Zusatz, nicht präregistriert (nur beschreibend)

Einwand: Das frühe Stoppen wählt nach fp32-Validierung. TAT ist bei 4 Bit früher am besten.

Härteste Gegenprobe: Ich habe je Lauf den besten **Test**-Wert bei w4a über alle 8 Messpunkte genommen („Orakel“). Das bevorzugt TAT, weil seine 4-Bit-Kurve am stärksten schwankt, und ist für eine echte Auswahl unzulässig.

| Form | Median Orakel-w4a |
|---|---|
| kL2B_tr | 53,5 |
| LB_tr | 54,4 |
| TAT | 56,5 |

Auch so liegt BPreCap 3,06 pp vorn, bei 12/0. Der Vorteil hängt also nicht an der Regel für das Stoppen.

## Deutung nach den Vorab-Regeln

- **Erfolg:** Im fairen Vergleich ist BPreCap bei 4 Bit besser als TAT. Nach Vorab-Regel ist das die Kernaussage für eine v0.3.
- **S6 „besser“ und Primär „besser“:** Der Vorteil aus B7 kam nicht nur von der Überanpassung am Ende. Er besteht auch am besten Punkt.
- **S2 „besser“:** Bei symmetrischen 4 Bit ist BPreCap besser.
- **S4 „besser“:** Der Kern-Befund (PreCap-Kern schlägt linearen Kern in fp32) gilt auch im Rezept bei langem Training.
- **Zusatz:**
  - Bei d100 ist BPreCap die beste der drei Formen bei w4a und in fp32 gleichauf mit TAT.
  - LB_tr ist bei w4 und w8a3 am besten (engere Grenze), in fp32 aber klar schlechter.
  - Bei d200 (B8) lag LB_tr auch bei w4a vorn. Die Front zwischen beiden Kernen verschiebt sich mit der Tiefe.

## Abgleich mit der Vorab-Erwartung

| Größe | Erwartung | Ergebnis |
|---|---|---|
| Primär „besser“ | 0,45 | **eingetreten, deutlicher als erwartet** |
| Primär „unklar“ | 0,4 | nein |
| Primär „schlechter“ | 0,15 | nein |
| S1 „schlechter“ (TAT in fp32 vorn) | 0,75 | **nein, unklar (gleichauf)** |
| S2 „besser“ | 0,85 | ja |
| S4 „besser“ | 0,75 | ja |
| S6 „besser“ | 0,75 | ja |
| Kontrolle bitgleich | 0,95 | ja |

Zum zweiten Mal (nach B8) lag ich bei S1 falsch. Bei längerem Training holt BPreCap in fp32 zu TAT auf.

## Antwort auf die Frage

Ja. Im TAT-Rezept ist BPreCap bei d100 auf CIFAR-10 nach langem Training mit frühem Stoppen bei fairen 4 Bit klar besser als TAT (−3,8 pp, alle 12 Seeds) und in fp32 gleichauf.

**Vertrauen: 0,85** für diesen Aufbau. Für eine allgemeine Aussage über andere Datensätze und Architekturen gilt **0,5**.

## Grenzen

- LR für 6880 Schritte nicht neu getunt, konstante LR ohne Plan. Ein LR-Abfall am Ende könnte die Überanpassung von TAT mildern.
- Nur Quantisierung nach dem Training (PTQ), kein quantisierungsbewusstes Training. Mit QAT könnte TAT den Bereichsnachteil teilweise ausgleichen.
- Ein Datensatz, MLP ohne Faltungen, d100. TAT ist unsere Nachbildung nach dks.
- Absolutes Niveau schwach (etwa 50 % Fehler auf CIFAR-10 mit einem tiefen MLP). Es geht um den Vergleich, nicht um Bestwerte.
- Frühes Stoppen nur an 8 Punkten.
