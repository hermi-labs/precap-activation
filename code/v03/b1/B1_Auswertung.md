# B1 – Auswertung

Stand: 30.10.2026, 03:15. PreCap/SPreCap, die Tiefenfamilie und Lagma sind Aarons Ideen. Auswertung mit Claude, strikt nach `B1_Praeregistrierung.md` (eingefroren 30.10. 00:24:44). Lauf: 00:26 bis 03:05, 2 Worker, am Netzteil.

## Integrität

| Prüfung | Ergebnis |
|---|---|
| Läufe | 242/242 ok (121 + 121), 0 Ausfälle |
| Hash-Prüfung (Code, Präregistrierung, LR-Tabellen, Vorrechnung, Paket) | alle 13 OK, vor und nach der Auswertung |
| Kontrolle kL2 und kL2_LD100, MNIST d100, Seed 42045, gegen Lagma-5 | **bitgleich** (fp32, w8a4_best, Trainingsverlust) |
| Sicherung | `sicherung/b1_2026-10-30/` mit `SHA256.txt` |

Befehl: `.venv/bin/python -I studien/b1/b1_analyze.py ~/PyCharmMiscProject/studien/b1`. Das Ergebnis steht in `analysis/b1_result.json`.

**Aufruffehler, offen genannt:** Mein erster Aufruf aus dem Ordner selbst mit `.` als Pfad meldete „Repro ok: False“. Grund: `Path('.').parent` ist wieder `.`, deshalb fand das Skript den Lagma-5-Ordner nicht. Der direkte Vergleich der Zeilen zeigt Gleichheit bis auf die letzte Stelle. Mit absolutem Pfad gibt das unveränderte Skript „Repro ok: True“ aus, alle Tests sind gleich. Das eingefrorene Skript wurde nicht verändert. Für spätere Studien gilt: die Auswertung immer mit absolutem Pfad aufrufen.

## Primärtest (kL2B gegen kL2_LD100, w8a4_best, Holm über 4 Zellen)

| Zelle | Median kL2B − LD100 | G/V | Holm | Urteil |
|---|---|---|---|---|
| MNIST d100 | **−9,6 pp** | 12/0 | 0,002 | besser |
| Fashion d100 | **−10,5 pp** | 12/0 | 0,002 | besser |
| MNIST d200 | **−29,0 pp** | 12/0 | 0,002 | besser |
| Fashion d200 | **−21,2 pp** | 12/0 | 0,002 | besser |

**Gesamturteil: besser als LD100**, in allen 48 von 48 Seed-Paaren.

## H1 (fp32 kL2B − kL2 höchstens +0,5 pp in jeder Zelle)

| Zelle | Median | G/V |
|---|---|---|
| MNIST d100 | +0,35 pp | 0/11 |
| Fashion d100 | +0,25 pp | 3/9 |
| MNIST d200 | **+1,12 pp** | 0/12 |
| Fashion d200 | **+1,09 pp** | 3/9 |

**H1 verletzt** (bei d200). **Kein Erfolg** nach der Vorab-Regel.

## Sekundär (je Familie Holm über 4 Zellen)

| Nr. | Vergleich | MNIST d100 | Fashion d100 | MNIST d200 | Fashion d200 | Gesamt |
|---|---|---|---|---|---|---|
| S1 | kL2B − kL2, fp32 | +0,35 (s) | +0,25 (u) | +1,12 (s) | +1,09 (s) | schlechter |
| S2 | kL2B − kL2, w8a4_best | −14,9 | −15,1 | −62,9 | −55,2 | **besser** |
| S3 | kL2B − LB, fp32 | +0,01 (u) | +0,22 (u) | +0,26 (u) | −0,46 (u) | **kein Effekt** |
| S4 | kL2B − LB, w8a4_best | +4,6 | +4,1 | +3,0 | +1,8 | **schlechter als LB** |
| S5 | kL2B − LD100, w8a4a_best | −3,3 | −3,7 | −12,7 | −10,6 | besser |
| S6 | kL2B − cs, fp32 | −1,2 | −1,5 | −35,3 | −13,3 | besser |
| S7 | kL2B − cs, w8a4_best | +4,6 | +3,4 | −56,1 | −39,8 | gemischt |

Mediane Differenzen in pp. s = schlechter, u = unklar. Alle anderen Zellen sind nach Holm signifikant in Richtung des Vorzeichens.

## Beschreibend (Median über 12 Seeds, Testfehler in %)

| Zelle | Form | fp32 | w8a4_best | w8a4a_best | Bereich max | Sättigung |
|---|---|---|---|---|---|---|
| MNIST d100 | kL2 | 5,12 | 28,6 | 19,1 | 7,0 | – |
| | LD100 | 4,98 | 23,1 | 14,3 | 5,0 | – |
| | kL2B | 5,51 | 13,6 | 11,1 | 3,0 | 5,1 % |
| | **LB** | 5,49 | **9,4** | 9,5 | 2,2 | 15,1 % |
| | cs | 6,51 | 9,2 | 8,1 | 3,3 | – |
| Fashion d100 | kL2 | 15,48 | 38,1 | 28,9 | 7,0 | – |
| | LD100 | 15,59 | 32,7 | 24,6 | 5,0 | – |
| | kL2B | 15,70 | 23,1 | 21,7 | 3,0 | 5,4 % |
| | **LB** | 15,49 | **19,1** | 19,3 | 2,2 | 15,2 % |
| | cs | 17,19 | 19,6 | 18,7 | 3,3 | – |
| MNIST d200 | kL2 | 5,67 | 77,3 | 57,1 | 10,3 | – |
| | LD100 | 6,00 | 43,9 | 24,4 | 5,2 | – |
| | kL2B | 7,06 | 14,4 | 11,7 | 3,0 | 9,6 % |
| | **LB** | 7,03 | **11,7** | 11,3 | 2,5 | 15,3 % |
| | cs | 42,69 | 71,4 | 62,9 | 3,3 | – |
| Fashion d200 | kL2 | 16,60 | 79,7 | 66,8 | 10,2 | – |
| | LD100 | 19,11 | 46,2 | 33,0 | 5,5 | – |
| | kL2B | 17,79 | 24,1 | 22,8 | 3,0 | 9,4 % |
| | **LB** | 18,03 | **22,0** | 22,6 | 2,5 | 15,4 % |
| | cs | 31,68 | 64,2 | 53,9 | 3,3 | – |

Sättigung ist der Anteil der Aktivierungen im Sättigungsbereich der Grenze am Ende des Trainings. Im Mechanik-Test (synthetische Daten, kurz trainiert) lag er bei 2,9 % (kL2B) bzw. 12,6 % (LB). Bei kL2B steigt er mit der Tiefe (5 % bei d100, 9,5 % bei d200), bei LB bleibt er bei etwa 15 %. Die Gradienten sind bei kL2B, LB, kL2 und LD100 gleich gesund (log10 erster/letzter Schicht 0,4–0,5). cs liegt bei 1,6 (d100) und 3,2 (d200).

## Vorab-Erwartung gegen Ergebnis

| Größe | Erwartung | Eingetreten |
|---|---|---|
| Primär „besser als LD100“ | 0,7 | ja, deutlicher als erwartet |
| H1 hält | 0,3 | nein |
| Erfolg | 0,25 | nein |
| S2 „besser als kL2“ | 0,95 | ja |
| S1 Fashion d200 einzeln „schlechter“ | 0,55 | ja |
| S3 „besser als LB“ | 0,45 | nein (kein Effekt) |
| S4 „schlechter als LB“ | 0,5 | ja |
| S6 „besser als cs“ | 0,8 | ja |
| S7 „schlechter als cs“ | 0,5 | gemischt: d100 schlechter, d200 besser |
| Kontrolle bitgleich | 0,95 | ja |

## Deutung nach den Vorab-Regeln

1. **Primär besser, H1 verletzt:** Die feste Grenze kauft 4 Bit mit fp32-Kosten bei der Tiefe, wie die Bremsen vorher. Der Preis liegt bei d200 bei etwa 1 pp, so groß wie bei LD100 in Lagma-5. Der 4-Bit-Gewinn ist aber viel größer: Bei d200 sinkt der w8a4-Fehler von 44–46 % (LD100) auf 14–24 %. Nach der Regel liegt der Zielkonflikt im Training, nicht in der Init. Dazu passt, dass die Sättigung von kL2B mit der Tiefe steigt (5 % auf 9,5 %).
2. **S3 kein Effekt und S4 schlechter als LB:** Der PreCap-Kern bringt in fp32 nichts gegenüber „linear plus Grenze“, und LB ist bei 4 Bit sogar besser. **Der Beitrag ist also nicht die PreCap-Form, sondern die bekannte kritische Begrenzung** (Pennington, Schoenholz & Ganguli 2017, https://arxiv.org/abs/1711.04735). Nach der Vorab-Regel ist LB die ehrlichere Empfehlung.
3. **S6 und S7:** kL2B und noch mehr LB vereinen bei d200 die Stärken von cs (schmal) und kL2 (tief). Bei d100 bleibt cs bei 4 Bit vorn, kostet aber 1–1,5 pp fp32.

## Eigene Einordnung (mit Claude, ehrlich)

- **Was B1 zeigt:** Der Satz aus v0.2 „kein Mitglied schafft beides“ beschreibt die alte Parametrisierung, keine Grenze des Problems. Bei 200 Schichten ohne Residuen trainiert eine Form mit Bereich ±2,5 auf 7,0 % (MNIST) und behält bei 4 Bit 11,7 %. kL2 kommt hier auf 77 %, LD100 auf 44 %.
- **Was B1 nicht zeigt:** Dass die PreCap-Form dafür nötig wäre. Die einfachere Form LB, M · φ(βx/M) mit zwei Konstanten aus der Tiefe, ist gleich gut in fp32 und besser bei 4 Bit. Für Lagma heißt das: Die geregelte Bremse ist in tiefen MLPs schwächer als eine feste, schmale Grenze, die schon bei der Init kritisch eingestellt ist.
- **Was noch fehlt:** 4 Bit kostet weiterhin gegenüber fp32: bei LB etwa 3,5–4,7 pp (MNIST d200: 11,7 gegen 7,0), bei kL2B 6–8 pp. Die Lücke ist kleiner, aber nicht geschlossen.
- **Neuheit:** Die kritische Begrenzung ist bekannt (Pennington 2017, „hard-tanh“). Diese Kombination habe ich in der Kurzrecherche nicht gefunden: Grenze und Steigung analytisch aus χ1 = 1 + 2/L bestimmt, dann gemessen bei 4 Bit. Das ist aber eine Kurzprüfung, keine systematische Recherche. Bis dahin gilt: bekanntes Prinzip, eigene Messung.

## Nachtrag Literatur (30.10., nachts, nach der Auswertung)

Pennington 2017 im Volltext: LB ist im Kern deren kritische hard-tanh (feste Grenze, kritisch über σw/σb). Bei LB ist sie weich und mit χ1 = 1 + 2/L eingestellt. Quantisierung kommt dort nicht vor. Details und Idee für B2 (orthogonale Gewichte) in `docs/wissen/tiefe_und_grenze.md`.

## Grenzen

- Plain MLP ohne Residuen, MNIST und Fashion, zwei Tiefen, 12 Seeds.
- Die Lernraten für kL2B und LB sind nicht getunt, sondern von kL2 übernommen. Ein Teil des fp32-Preises bei d200 kann daran liegen. Er würde LB eher noch begünstigen, weil LB enger ist.
- M = 3 und ρ = 0,8 sind fest, nicht optimiert.
- Nur Post-Training-Quantisierung pro Tensor.

## Folgen (Vorschlag, nicht gestartet)

1. **Wichtigste Frage:** Ist LB auf neuen Seeds und mit getunter LR wirklich die bessere Form? Dazu käme eine Grenze, die mitwachsen darf (M als Vielfaches eines Quantils), oder M = 4 gegen den fp32-Preis bei d200. Als Kandidat für **B2** ist das präregistrierbar.
2. **Ehrlicher Abgleich mit Pennington 2017** (Volltext lesen, ist ihre Familie mit hard-tanh das Gleiche wie LB?) und eine Recherche zu „critical initialization bounded activation quantization“, bevor etwas davon öffentlich wird.
3. **Lagma** bleibt für Fälle, in denen keine kritische Init möglich ist (Transformer, vortrainierte Netze).
