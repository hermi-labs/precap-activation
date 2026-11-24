# C7 – Auswertung: Wiederholung von V2a2 mit PyTorch/CUDA (RTX 4070)

Stand: 24.11.2026, 04:59. Präregistrierung: `C7_Praeregistrierung.md`, Freeze 2026-11-23T18:51:56+01:00 (auf dem PC mit der RTX 4070). PreCap, BPreCap, die Hülle und das Kalibrierungsrezept sind Aarons Ideen. Das Design ist mit Claude entworfen, ausgewertet habe ich mit Claude.

- **Lauf:** 23.11. 18:59:47 bis 24.11. 04:52:38 auf einem Windows-PC mit RTX 4070. Es liefen 2 Worker mit PyTorch 2.14.1+cu126, Gerät `cuda` in allen 36 Zeilen.
- **Übertragen** vom PC auf den Mac am 24.11. um 04:56:23, vollständiger Ordner, 61 Dateien. Die Auswertung lief auf dem Mac.

Befehle:
```
(cd code/v04/c7 && tr -d '\r' < meta/frozen_sha256.txt | shasum -a 256 -c -)   # 9/9 Code-Dateien OK, vorher und nachher
(cd data && <DATA-Zeilen der Hash-Datei, Pfad ohne DATA/> | shasum -a 256 -c -)            # 6/6 CIFAR-Dateien OK (Mac-Kopie)
(cd code/v04/v2a2 && shasum -a 256 -c meta/frozen_sha256.txt)                                # 21/21 OK, vorher und nachher
python -I code/v04/c7/c7_analyze.py "$PWD/code/v04/c7" "$PWD/code/v04/v2a2"
```
- Ausgaben: `analysis/c7_result.json`, `analysis_ausgabe.txt`.
- Die Abstände je Seed stehen in `analysis/je_seed_cuda_minus_mlx.txt`. Die Präregistrierung sieht sie als Sekundärgröße vor, das eingefrorene Skript gibt sie nicht aus. Ich habe sie deshalb getrennt aus denselben CSV-Dateien berechnet, ohne Test.

## Urteil

- **P1 (BPreCap − SELU, w8a4a_best): „besser“.** Median −3,09 pp, 12 von 12 Seeds, Holm-p 0,00098.
- **P2 (BPreCap − TAT, w8a4a_best): „besser“.** Median −15,65 pp, 12 von 12 Seeds, Holm-p 0,00098.
- **Folge nach der Entscheidungsregel** (Zeile „P1 und P2 besser“): Der Hauptbefund hängt nicht an MLX oder Apple-Hardware. v0.4 berichtet C7 als Wiederholung, wie C6 in v0.3.1.

## Primärtests (Wilcoxon gepaart, zweiseitig, Holm über 2, Ausfall = 90 %)

| Test | Median Δ (pp) | Siege / Niederlagen | p | Holm-p | Urteil | V2a2 (MLX) zum Vergleich |
|---|---|---|---|---|---|---|
| P1 BPreCap − SELU | −3,09 | 12 / 0 | 0,00049 | 0,00098 | besser | −2,55 pp, 12/0 |
| P2 BPreCap − TAT | −15,65 | 12 / 0 | 0,00049 | 0,00098 | besser | −13,73 pp, 12/0 |

Δ je Seed, 42350–42361 (pp):
- P1: −5,44 / −1,06 / −3,89 / −4,44 / −2,93 / −5,30 / −3,18 / −0,67 / −2,12 / −2,12 / −3,59 / −3,01
- P2: −23,37 / −9,22 / −16,97 / −10,20 / −17,04 / −10,09 / −18,05 / −12,58 / −9,26 / −19,47 / −14,33 / −19,12

## Sekundär (beschreibend)

Mediane über 12 Seeds, Testfehler in %, CUDA / MLX (Differenz):

| Form | fp32 | w8a8 | w8a4a_max | w8a4a_best | 4-Bit-Preis (pp) |
|---|---|---|---|---|---|
| BPreCap | 43,11 / 42,88 (+0,24) | 43,83 / 43,71 (+0,12) | 49,33 / 49,08 (+0,25) | 48,07 / 48,07 (−0,01) | 4,60 / 4,64 |
| SELU | 41,62 / 41,40 (+0,22) | 43,19 / 43,02 (+0,17) | 56,43 / 55,87 (+0,56) | 51,18 / 50,33 (+0,85) | 9,33 / 8,83 |
| TAT | 39,97 / 40,27 (−0,31) | 41,90 / 41,36 (+0,54) | 89,89 / 89,89 (0,00) | 63,76 / 61,63 (+2,14) | 22,02 / 21,52 |

- **TAT nach CLE, w8a4a_best:** CUDA 64,97 %, MLX 63,91 %.
- **Abstand je Seed, CUDA − MLX:** Median, Spanne und Median des Betrags, in pp.

  | Form | fp32 | w8a4a_best |
  |---|---|---|
  | BPreCap | +0,06 [−0,40; +0,75], \|d\| 0,20 | −0,28 [−1,26; +2,15], \|d\| 0,38 |
  | SELU | +0,34 [−0,22; +0,90], \|d\| 0,34 | +0,70 [−1,06; +1,64], \|d\| 0,79 |
  | TAT | −0,52 [−3,28; +4,32], \|d\| 1,02 | −1,41 [−14,90; +7,67], \|d\| 5,16 |

- **Gewählte Kalibrierung:**
  - TAT: 12-mal mse4.
  - BPreCap: 10-mal mse4, je einmal p999 und p9999.
  - SELU: 7-mal mse4, 5-mal p999.
- **Laufzeit je Lauf** (Median, meist 2 Worker parallel): BPreCap 2 834 s, TAT 1 020 s, SELU 695 s. Allein auf der GPU waren es etwa 870, 358 und 329 s.

## Vorab-Wahrscheinlichkeiten gegen Ausgang

| Aussage | vorab | eingetreten |
|---|---|---|
| P2 „besser“ | 0,95 | ja |
| P1 „besser“ | 0,75 | ja |
| Mediane je Form innerhalb von 1 pp zu MLX | 0,70 | nein: 14 von 15 Medianen ja, TAT w8a4a_best +2,14 pp |
| alle 36 Läufe ok | 0,95 | ja |

## Integrität

| Prüfung | Ergebnis |
|---|---|
| Zeilen | 36/36 ok, run_id eindeutig, keine fehlenden Paare |
| nicht-endliche Werte | 0 |
| Tracebacks / Fehler im Log | 0 (w0.log, w1.log) |
| Hashes Code | 9/9 OK vor und nach der Analyse. Die Code-Dateien auf dem PC sind bytegleich mit dem Mac. |
| Hashes Daten | 6/6 CIFAR-Dateien der Hash-Datei stimmen mit der Mac-Kopie überein. Nach dem Lauf wurden sie nicht noch einmal auf dem PC geprüft, weil der PC aus war. |
| V2a2 (Referenz) | 21/21 OK vor und nach der Analyse |
| Vorab-Prüfung | Äquivalenz CUDA 25/25, Rauchtest 3/3, Auswertungstest 5/5 OK (`meta/`) |
| Gerät | alle Zeilen `cuda`, PyTorch 2.14.1+cu126 |

## Einordnung

- **Wiederholung:** Beide Hauptvergleiche aus V2a2 wiederholen sich auf anderer Hardware mit eigenständigem Code, mit derselben Richtung und denselben 12 von 12 Seeds. Die Abstände sind sogar etwas größer als auf MLX: 3,09 statt 2,55 pp und 15,65 statt 13,73 pp.
- **BPreCap und SELU sind sehr stabil über die Plattform.** Die Mediane liegen innerhalb von 0,25 bzw. 0,85 pp, je Seed meist unter 1 pp.
- **TAT ist bei 4 Bit plattformempfindlich.** Je Seed weicht w8a4a_best um bis zu 15 pp ab, der Median des Betrags liegt bei 5,2 pp. Bei TAT bricht w8a4a_max zusammen (89,9 %, praktisch Raten), und mse4 sitzt auf einer steilen Kante. Kleine Rundungsunterschiede im Training kippen dort viel. Der Median bleibt nahe (+2,1 pp), und der Abstand zu BPreCap ist auf jedem Seed groß (mindestens 9,2 pp). Für P2 spielt das deshalb keine Rolle.
- **Bitgleich ist das nicht zu erwarten** (siehe Präregistrierung). Die Abweichungen bei fp32 (Median des Betrags 0,2–1,0 pp) passen zu anderer Rundung in den Kernels.
- **Grenzen der Aussage:** C7 prüft die Plattform, nicht neue Stichproben. Es sind dieselben Seeds, Startgewichte und Hyperparameter wie in V2a2.
- **Bezug:** V2a2 ist der Hauptvergleich von v0.4.

## Grenzen

- Gleiche Seeds und Hyperparameter wie V2a2. Kein Tuning auf CUDA.
- Nur V2a2. Die tanh-Vergleiche (V2d, V2gb, V2h) und die Rauschmessungen (V2e, V2f, V2k, V2m) werden nicht wiederholt.
- Eine GPU, eine PyTorch-Version.
- Der Kopf der eingefrorenen Präregistrierung sagt noch „Entwurf, noch nicht eingefroren“. Der Inhalt ist der eingefrorene Stand (Hash OK), nur die Kopfzeile ist veraltet. Ändern darf ich sie nicht.

## Nächster Schritt

- v0.4 berichtet C7 in Abschnitt 4.6 und am Ende von 5.2.
