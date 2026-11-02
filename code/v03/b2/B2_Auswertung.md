# B2 – Auswertung: weiche Lagma-Hülle gegen harte Grenze

Stand: 30.10.2026, 23:10. PreCap, die Tiefenfamilie und Lagma sind Aarons Ideen. BPreCap (= kL2B) ist ein Arbeitsname. Die Auswertung habe ich mit Claude gemacht, strikt nach `B2_Praeregistrierung.md` (eingefroren 30.10. 10:39:03). Lauf: 10:41 bis 22:40. Dazwischen hat der Mac geschlafen (11:50–17:00 und bis 21:55 zeitweise: Deckel zu, am Akku). Die Läufe haben nur pausiert.

## Integrität

| Prüfung | Ergebnis |
|---|---|
| Läufe | 194/194 ok (97 + 97), 0 Ausfälle |
| Hash-Prüfung (Code, Präregistrierung, LR-Tabellen, p7-Konstanten, Paket) | alle 12 OK nach dem Lauf |
| Kontrolle LB und kL2B, MNIST d100, Seed 42060, gegen B1 | **bitgleich** (fp32, w8a4_best, w8a4a_best, Trainingsverlust) |
| Läufe über die Schlafpause | 8 Läufe. **2 davon nachgerechnet** (mnist_d200_LB_s42082, fashion_d200_kL2Bh_s42072): **bitgleich** in allen 8 Kennzahlen (`nachpruefung_pause.py`) |
| Sicherung | `sicherung/b2_2026-10-30/` mit `SHA256.txt` |

Befehl: `.venv/bin/python -I studien/b2/b2_analyze.py ~/PyCharmMiscProject/studien/b2`. Das Ergebnis steht in `analysis/b2_result.json`.

## Primärtest (LB weich − LBh hart, w8a4_best, Holm über 4 Zellen)

| Zelle | Median | G/V | Holm | Urteil |
|---|---|---|---|---|
| MNIST d100 | **−1,31 pp** | 10/2 | 0,028 | besser |
| Fashion d100 | **−0,86 pp** | 11/1 | 0,006 | besser |
| MNIST d200 | **−1,36 pp** | 10/2 | 0,028 | besser |
| Fashion d200 | **−1,80 pp** | 9/3 | 0,028 | besser |

**Gesamturteil: besser als LBh** (4 von 4 Zellen).

**H1** (fp32 LB − LBh ≤ +0,5 pp je Zelle): **hält**. Die Mediane liegen bei −0,18 / −0,14 / −0,52 / −0,82 pp, die weiche Hülle ist in fp32 tendenziell sogar besser.

**Ergebnis: Erfolg nach der Vorab-Regel.** Vorab hatte ich das nur mit 0,2 erwartet.

## Sekundär (je Familie Holm über 4 Zellen)

| Nr. | Vergleich | MNIST d100 | Fashion d100 | MNIST d200 | Fashion d200 | Gesamt |
|---|---|---|---|---|---|---|
| S1 | LB − LBh, fp32 | −0,18 | −0,14 | −0,52 | −0,82 | kein Effekt (alle unklar, Tendenz weich besser, 8–10/12) |
| S2 | kL2B − kL2Bh, w8a4_best | −0,37 | +0,93 | +0,56 | −0,76 | **kein Effekt** |
| S3 | kL2B − kL2Bh, fp32 | **−0,35** | −0,06 | −0,57 | −0,19 | gemischt (nur MNIST d100 besser) |
| S4 | LB − LBh, w8a4a_best | **−1,24** | **−0,75** | **−0,94** | **−2,28** | **besser als LBh** |
| S5 | LB − LBh, w8a3_best | **−1,78** | +0,60 | +0,31 | +0,49 | gemischt |
| S6 | kL2Bh − LBh, fp32 | −0,19 | −0,03 | +0,24 | −0,11 | kein Effekt |
| S7 | kL2B − LB, w8a4_best | +4,25 | +4,31 | +4,80 | +2,45 | **schlechter als LB** (wie B1) |
| S8 | kL2B − LB, fp32 | **−0,53** | +0,16 | +0,24 | −0,41 | gemischt (nur MNIST d100 besser) |

Mediane in pp. Fett heißt nach Holm signifikant.

## Beschreibend (Median über 12 Seeds, Testfehler in %)

| Zelle | Form | fp32 | w8a4 | w8a4a | w8a3 | Bereich | Sättigung |
|---|---|---|---|---|---|---|---|
| MNIST d100 | LB | 5,49 | **9,21** | 9,43 | 48,1 | 2,22 | 15,1 % (> ρM) |
| | LBh | 5,68 | 10,36 | 10,20 | 50,2 | 2,30 | 6,5 % (> M) |
| | kL2B | 5,13 | 13,50 | 11,09 | 62,1 | 3,00 | 5,1 % |
| | kL2Bh | 5,46 | 14,01 | 11,29 | 62,7 | 3,00 | 2,2 % |
| Fashion d100 | LB | 15,50 | **18,93** | 19,24 | 51,6 | 2,22 | 15,4 % |
| | LBh | 15,72 | 19,85 | 20,17 | 52,0 | 2,30 | 7,0 % |
| | kL2B | 15,72 | 23,04 | 21,68 | 63,6 | 3,00 | 5,4 % |
| | kL2Bh | 15,78 | 22,34 | 21,69 | 63,3 | 3,00 | 2,5 % |
| MNIST d200 | LB | 7,05 | **11,70** | 11,48 | 56,9 | 2,45 | 14,8 % |
| | LBh | 7,65 | 13,43 | 13,05 | 56,4 | 2,54 | 7,2 % |
| | kL2B | 7,43 | 16,32 | 13,18 | 66,6 | 3,00 | 9,0 % |
| | kL2Bh | 7,94 | 15,80 | 16,32 | 64,9 | 3,00 | 4,2 % |
| Fashion d200 | LB | 18,23 | **21,99** | 22,68 | 60,0 | 2,45 | 15,1 % |
| | LBh | 18,54 | 24,64 | 25,78 | 59,3 | 2,54 | 7,8 % |
| | kL2B | 17,56 | 24,76 | 23,15 | 67,0 | 3,00 | 9,4 % |
| | kL2Bh | 18,52 | 25,61 | 26,41 | 65,9 | 3,00 | 4,4 % |

Die Sättigung ist bei weicher und harter Hülle verschieden definiert: weich |Kern| > ρM, hart |Kern| > M.

**3 Bit nach dem Training:** Alle Formen brechen ein, auf 42–67 % Fehler. Die Kosten (w8a3 − fp32) steigen bei LB von d100 auf d200 (MNIST 42,6 → 49,9 pp, Fashion 36,1 → 41,8 pp), bei BPreCap kaum. Das passt zur Richtung von Blumenfeld et al. 2019. 3 Bit ist hier aber schon bei d100 praktisch unbrauchbar, die Aussage ist also schwach.

## Vorab-Erwartung gegen Ergebnis

| Größe | Erwartung | Eingetreten |
|---|---|---|
| Primär „besser als LBh“ | 0,3 | **ja** |
| H1 hält | 0,7 | ja |
| Erfolg | 0,2 | **ja** |
| S1 „kein Effekt“ | 0,5 | ja |
| S6 „kein Effekt“ | 0,6 | ja |
| S7 „schlechter als LB“ | 0,85 | ja |
| S8 „kein Effekt“ | 0,7 | nein (gemischt) |
| 3-Bit-Kosten d200 > d100 bei allen Formen | 0,6 | nein (nur LB/LBh deutlich, BPreCap kaum) |
| Kontrolle bitgleich | 0,95 | ja |

## Deutung nach den Vorab-Regeln

1. **Erfolg:** Die weiche Lagma-Hülle hat gegenüber der kritischen hard-tanh einen eigenen Wert. Bei linearem Kern ist sie bei 4 Bit um 0,9–1,8 pp besser (symmetrisch) bzw. 0,8–2,3 pp (asymmetrisch, S4), ohne fp32-Preis. Die Lagma-Form ist als Baustein gerechtfertigt.
2. **S2/S3 nicht gleich gerichtet:** Mit PreCap-Kern (BPreCap) gibt es bei 4 Bit keinen Unterschied zwischen weich und hart. Der Befund hängt also vom Kern ab.
3. **S7:** B1 wiederholt sich auf neuen Seeds. LB ist bei 4 Bit um 2,5–4,8 pp besser als BPreCap.

## Eigene Einordnung (ehrlich, mit Claude)

- **Was B2 nicht trennt:** Verglichen wurde bei gleicher Kritikalität. Dadurch liegt die weiche Grenze etwas enger (LB 2,22/2,45 gegen LBh 2,30/2,54, also etwa 3,5 %). Ein engerer Bereich heißt bei Max-Kalibrierung etwa 7 % weniger Rundungsrauschen. Ein Teil des 4-Bit-Vorteils kann daher vom schmaleren Bereich kommen, nicht von der Weichheit selbst.
- **Ein Hinweis in diese Richtung ist S2:** Bei BPreCap ist M für weich und hart gleich (3), und es gibt keinen 4-Bit-Unterschied. Der Kern ist dort allerdings ein anderer, die Trennung ist also nicht sauber.
- **Für die Weichheit spricht:**
  - Die fp32-Tendenz (S1) zeigt in allen 4 Zellen zur weichen Hülle (8–10 von 12 Seeds), wenn auch nach Holm nicht signifikant.
  - Bei d200 ist der 4-Bit-Vorteil größer (MNIST 1,36, Fashion 1,80 bzw. asymmetrisch 2,28 pp), als 3,5 % engerer Bereich allein erwarten lassen. Das ist eine Überschlagsrechnung, kein Test.
- **Saubere Trennung als Nachfolgestudie (Vorschlag B4):** weich gegen hart **bei gleichem M** (dann ist eine der beiden Formen nicht mehr exakt kritisch) oder hart mit M_hart = M_weich. Erst dann ist die Aussage „weich ist besser“ eindeutig.

## Grenzen

- MNIST und Fashion, zwei Tiefen, 12 Seeds, MLP ohne Residuen.
- Lernrate nicht für die neuen Formen getunt (von kL2 übernommen).
- Weich und hart unterscheiden sich durch die gleiche Kritikalität leicht im Bereich (siehe oben).
- 3 Bit ist nach dem Training bei diesen Tiefen unbrauchbar. Die Blumenfeld-Beschreibung ist deshalb schwach.
- Schlafpausen im Lauf, nachgeprüft (2 von 8 Pausenläufen bitgleich).

## Folgen

- **Für Lagma:** Erstes Ergebnis, in dem die Lagma-Form φ einen messbaren eigenen Beitrag hat (bei linearem Kern), vorbehaltlich der Trennung von Weichheit und Bereich.
- **Bestes bisheriges Paket für Tiefe plus 4 Bit in MLPs:** LB (linearer Kern, weiche Lagma-Hülle, kritisch eingestellt). Bei d200 MNIST 7,05 % fp32 und 11,7 % w8a4.
- **Für BPreCap:** Bei 4 Bit hinter LB, in fp32 gemischt. Die Entscheidung über den Kern fällt in B3 (CIFAR-10).
