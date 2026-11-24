# V2c – Auswertung (privat, mit Claude, 19.11.2026, 02:50)

Stand: 19.11.2026. Präregistrierung: `V2c_Praeregistrierung.md`, Freeze 2026-11-18T17:47:09+0100, Hash-Prüfung vor und nach der Analyse: 19/19 OK. BPreCap und die Hülle sind Aarons Ideen, die Auswertung habe ich mit Claude gemacht. Lauf: 18.11. 18:30 bis 19.11. 02:47, ein Worker (`v2cw0`), am Netzteil.

Befehle (im Projektordner, 02:50):
```
(cd studien/v2c && shasum -a 256 -c meta/frozen_sha256.txt)      # 19/19 OK, vorher und nachher
python studien/v2c/v2c_analyze.py <absoluter Pfad>/studien/v2c
```
Ausgabe: `analysis/v2c_result.json`. tanh kommt aus V2c, BPreCap, SELU und TAT aus der V2a2-CSV (historische Arme, abgesichert durch die bitgleiche Kontrolle in `meta/kontrolle.txt`, 54/54 Spalten).

## Urteil

- **P1 (Test-Fehler w8a4a_best): „BPreCap besser“.** BPreCap − tanh: −20,06 pp, 12/0, Holm-p = 0,0010.
- **P2 (4-Bit-Preis, w8a4a_best − fp32): „tanh besser“.** BPreCap − tanh: +3,29 pp, 0/12, Holm-p = 0,0010. tanh verliert durch 4 Bit weniger als BPreCap.
- **Deutung nach der Vorab-Tabelle (Zeile 1):** „Begrenzung macht den Preis klein, die Tiefenkalibrierung macht das Netz gut.“ BPreCap gewinnt bei 4 Bit über die Trainierbarkeit, nicht über einen kleineren Preis als andere begrenzte Formen.

## Primärtests (Holm über 2)

| Test | Größe | Median BPreCap | Median tanh | Δ BPreCap − tanh (pp, Median je Seed) | Siege BPreCap | p (Holm) | Urteil |
|---|---|---|---|---|---|---|---|
| P1 | Test-Fehler w8a4a_best | 48,07 % | 68,11 % | −20,06 | 12/0 | 0,0010 | BPreCap besser |
| P2 | 4-Bit-Preis | 4,64 pp | 1,26 pp | +3,29 | 0/12 | 0,0010 | tanh besser |

## Sekundärtests (Holm über 3)

| Test | Größe | Δ (pp) | Siege | p (Holm) | Urteil | vorab |
|---|---|---|---|---|---|---|
| S1 | fp32, BPreCap − tanh | −23,36 | 12/0 | 0,0015 | besser | 0,45 |
| S2 | w8a4a_best, tanh − SELU | +17,94 | 0/12 | 0,0015 | schlechter | „besser“ 0,55 |
| S3 | w8a4a_best, tanh − TAT | +6,00 | 2/10 | 0,11 | unklar | „besser“ 0,85 |

## Beschreibend

Mediane über 12 Seeds, Test-Fehler am frühen Stopp (für alle Formen im Median Schritt 3440, also Ende):

| Form | LR / WD | fp32 | w8a8 | w8a4a_max | w8a4a_best | 4-Bit-Preis | Aktivierung unten / oben / |max| |
|---|---|---|---|---|---|---|---|
| BPreCap | 3e-5 / 0 | 42,88 % | 43,71 % | 49,08 % | 48,07 % | 4,64 pp | −2,24 / 3,00 / 3,00 |
| SELU | 3e-5 / 0 | 41,40 % | 43,02 % | 55,87 % | 50,33 % | 8,83 pp | −1,75 / 5,94 / 29,1 |
| TAT | 1e-4 / 0 | 40,27 % | 41,36 % | 89,89 % | 61,63 % | 21,52 pp | −3,70 / 9,10 / 158,2 |
| **tanh** | 3e-5 / 0 | 66,53 % | 67,01 % | 67,83 % | 68,11 % | 1,26 pp | −0,93 / 0,93 / 1,00 |

- **Tuning tanh:** 12/12 ok. Gewählt LR 3e-5, WD 0 (Median-Validierungsgenauigkeit 33,1 %), nicht am Rand des Rasters (vorab 0,35 für „am Rand“). WD 3,0 schadet auch hier.
- **Kalibrierung tanh:** sehr gemischt (p9999 5×, mse4 3×, p99999 2×, p999 1×, max 1×). Bei |max| ≤ 1 sind die Kalibrierungen fast gleichwertig.
- **Trainingsverlust (letzte 50 Schritte):** tanh 1,76, BPreCap 1,18, SELU 1,11, TAT 1,03. tanh lernt in 3440 Schritten deutlich weniger.
- **Am Ende statt am Stopp:** BPreCap − tanh bei w8a4a_best −19,94 pp, fp32 −23,52 pp, w8a4a_max −18,45 pp, jeweils 12/0. Gleiche Richtung.

## Explorativ (nicht präregistriert, nachträglich gerechnet)

Der absolute Preis ist bei einem schlecht trainierten Netz kleiner, weil es weniger zu verlieren hat: tanh liegt in fp32 nur 23,5 pp über dem Zufall (90 %). Deshalb habe ich zusätzlich den **relativen Preis** gerechnet, (w8a4a_best − fp32) / (90 % − fp32), also den Anteil des Vorsprungs vor dem Zufall, den 4 Bit kostet:

| Form | relativer Preis (Median) |
|---|---|
| tanh | 5,4 % |
| BPreCap | 9,9 % |
| SELU | 18,2 % |
| TAT | 44,7 % |

Gepaart ist der relative Preis von BPreCap um 4,9 Prozentpunkte größer als bei tanh, schlechter bei 11 von 12 Seeds. Auch bereinigt ist tanh also robuster gegen 4 Bit. Das passt zum Bereich: tanh bleibt in [−0,93; 0,93] und ist symmetrisch, BPreCap nutzt [−2,24; 3,00]. Die Rangfolge beim Preis folgt der Breite des Bereichs (tanh < BPreCap < SELU < TAT).

Grenze dieser Rechnung: Der Abstand zum Zufall ist nur eine grobe Normierung. Ein Netz mit 66 % Fehler kann andere, gröbere Merkmale nutzen, die von Natur aus weniger empfindlich sind.

## Integrität

| Prüfung | Ergebnis |
|---|---|
| Zeilen | Bestätigung 12/12 ok, Tuning 12/12 ok, 0 Ausfälle, 0 fehlend; historisch 36/36 ok |
| run_id | eindeutig |
| nicht-endliche Werte | 0 |
| Tracebacks/Fehler | 0 in `w0.log` und `smoke.log` |
| Hashes | 19/19 OK, vor und nach der Analyse (einschließlich V2a2-Code, V2a2-CSV, Pilot, Daten) |
| Mechanik vorab | volle Kontrolle: V2a2-Zeile `confirm_bprecap_lr3e-05_wd0_s42350` mit dem V2c-Skript bitgleich nachgerechnet, 54/54 Spalten |

## Vorab-Wahrscheinlichkeiten gegen Ausgang

| Aussage | vorab | eingetreten |
|---|---|---|
| alle 12 tanh-Läufe ok | 0,9 | ja |
| P1 „BPreCap besser“ (vor Pilot) | 0,4 | ja |
| P1 „tanh besser“ | 0,2 | nein |
| S1 „besser“ | 0,45 | ja |
| S2 „besser“ (tanh vor SELU) | 0,55 | nein (schlechter) |
| S3 „besser“ (tanh vor TAT) | 0,85 | nein (unklar) |
| Raster am Rand | 0,35 | nein |
| *nach Pilot:* P1 „BPreCap besser“ | 0,85 | ja |
| *nach Pilot:* P2 „BPreCap besser“ | 0,35 | nein |
| *nach Pilot:* P2 „tanh besser“ | 0,25 | ja |

Vor dem Pilot habe ich tanh deutlich überschätzt: Ich hatte erwartet, dass eine begrenzte Form ohne Kalibrierung bei 50 Schichten noch gut trainiert. Der Pilot hat das korrigiert, und die nach dem Pilot ergänzten, markierten Werte lagen bei P1 richtig. Bei P2 lag ich mit 0,35 zu 0,25 auf der falschen Seite.

## Einordnung

- **V2a2:** BPreCap war bei 4 Bit besser als SELU (−2,55 pp) und TAT (−13,73 pp), mit einem fp32-Preis von 1,6–3,2 pp. V2c zeigt, woher das kommt. Der kleine 4-Bit-Preis ist eine Eigenschaft der Begrenzung; eine noch engere Begrenzung (tanh) hat einen noch kleineren Preis. Dass BPreCap am Ende vorne liegt, braucht zusätzlich die Tiefenkalibrierung: Ohne sie (tanh, Verstärkung 1) trainiert das 50-Schichten-Netz schlecht (66,5 % fp32-Fehler gegen 42,9 %).
- **Literatur:** Xiao et al. 2018 (https://arxiv.org/abs/1806.05393) trainieren tanh-CNNs ohne Normierung bis 10 000 Schichten, aber mit Delta-Orthogonal **an der geordnet-chaotischen Grenze** (σ_w², σ_b so gewählt, dass χ1 = 1). Unsere tanh hat Verstärkung 1 und Bias 0. Das ist der Randpunkt dieser Linie mit q* = 0: Die Varianz fällt über die Tiefe langsam gegen 0, und bei endlichem q ist χ = E[tanh'(z)²] < 1. Nachgerechnet (Mean-Field, Gauß-Hermite, Start q = 1, 19.11. 02:49): nach 50 Schichten q ≈ 0,010, Produkt der χ ≈ 0,014. Signal und Gradient schrumpfen also stark. Das schlechte Training ist damit erwartbar und kein Widerspruch zu Xiao et al. Es heißt aber: **V2c vergleicht BPreCap mit einer unkalibrierten tanh, nicht mit der besten tanh.** Ein Gutachter würde die kritische tanh nach Xiao et al. als Vergleich fordern.

## Grenzen

- **Eine tanh-Variante, unkalibriert.** Eine tanh an der kritischen Linie (Xiao et al.) ist nicht geprüft. Sie könnte gut trainieren und zugleich den kleinen Preis behalten. Dann würde BPreCaps P1-Vorteil schrumpfen oder verschwinden. Das ist die wichtigste offene Lücke für v0.4.
- **Preis-Vergleich bei sehr verschiedener Güte:** tanh hat in fp32 23,6 pp mehr Fehler. Die explorative Normierung stützt P2, ersetzt aber keinen Vergleich bei gleicher fp32-Güte.
- **Historische Arme**, nicht zeitgleich gerechnet; abgesichert durch die bitgleiche Kontrolle.
- **Alle Grenzen von V2a2:** ein Datensatz (CIFAR-10), ein schlichtes CNN (L = 50, Breite 32), nur PTQ je Tensor, M1 Pro.

## Nächster Schritt

- **v0.4:** Die Kernaussage wird präziser, aber auch bescheidener: „Begrenzung macht den 4-Bit-Preis klein, die Tiefenkalibrierung macht das tiefe Netz trainierbar; BPreCap verbindet beides.“ Die Grenze „ohne tanh“ in Abschnitt 7 ist erledigt und wird ersetzt durch „nur unkalibrierte tanh“.
- **Vorschlag V2d (Aaron entscheidet):** tanh an der kritischen Linie nach Xiao et al. (σ_w² und σ_b so, dass χ1 = 1 bzw. 1 + 2/L wie bei BPreCap), sonst wie V2c. Etwa 9 h, ein Worker. Das ist der faire Gegner und schließt die Lücke vor einer Veröffentlichung.
- **Sicherung:** `sicherung/v2c_2026-11-19/`.
