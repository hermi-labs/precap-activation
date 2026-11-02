# B4 – Auswertung: Ist es die weiche Form oder die engere Grenze?

Stand: 31.10.2026, etwa 03:40. Die Auswertung folgt strikt `B4_Praeregistrierung.md`. Lagma, PreCap und die Tiefenfamilie sind Aarons Ideen. Lauf und Auswertung habe ich mit Claude gemacht.

## Kurz

- **Erfolg nach der Vorab-Regel.** Bei gleicher (enger) Grenze ist die weiche Lagma-Hülle (LB) bei 4 Bit besser als der harte Clip (LBhM). Das Gesamturteil ist „besser als LBhM“: 2 Zellen signifikant, die anderen 2 in dieselbe Richtung. H1 hält, die weiche Form ist auch in fp32 in keiner Zelle schlechter, sondern in allen 4 Zellen besser (Median).
- **Die Grenze allein macht nichts aus.** Die Breite der Grenze wurde um etwa 3,5 % verändert (S2 und S3). Das ergab bei 4 Bit in 8 von 8 Zellen „unklar“. Der Vorbehalt aus B2 („der Vorteil kommt vielleicht nur vom engeren Bereich“) ist damit **nicht bestätigt**.
- **B2 wiederholt sich auf neuen Seeds** (S7: LB besser als LBh in 3 von 4 Zellen, die vierte mit −2,7 pp in dieselbe Richtung).
- **Einschränkung nach der Vorab-Regel:** Bei weiter Grenze (S1: LBM gegen LBh) ist der Form-Effekt nicht signifikant (4 × „unklar“, Mediane alle negativ). Nach der Deutungstabelle heißt das: „Die Form zählt nur bei enger Grenze.“ Die beschreibende Zerlegung spricht eher für einen allgemeinen Form-Effekt mit zu wenig Trennschärfe bei S1, siehe unten.

## Integrität

| Prüfung | Ergebnis |
|---|---|
| Läufe | 194 von 194 ok (w0 97, w1 97), 0 Ausfälle |
| Hash-Prüfung | `shasum -a 256 -c meta/frozen_sha256.txt`: 13 von 13 OK |
| Kontrolle | LB und LBh, MNIST d100, Seed 42072: bitgleich zu B2 (2 von 2) |
| Schlaf | Größte Lücke zwischen zwei Läufen: 98 s (w0) bzw. 99 s (w1). Kein Schlaf |
| Laufzeit | 01:22–03:17 |
| Sicherung | `sicherung/b4_2026-10-31/` (workers, analysis, meta, logs), 12 Dateien, `SHA256.txt` 12 von 12 OK |

Befehl: `~/PyCharmMiscProject/.venv/bin/python -I b4_analyze.py ~/PyCharmMiscProject/studien/b4`, Ergebnis in `analysis/b4_result.json`.

**Hinweis zu den Nulldifferenzen:** In 2 Zellen gab es je eine Differenz von genau 0. scipy wechselt dann von „exakt“ auf die Normalapproximation. Das sind S2 MNIST d100 und S4 Fashion d200. Ich habe beide exakt ohne die Null nachgerechnet:

| Zelle | p (verwendet) | p exakt ohne Null | Urteil |
|---|---|---|---|
| S2 MNIST d100 | 0,248 | 0,278 | unklar, unverändert |
| S4 Fashion d200 | 0,0058 | 0,0029 | besser, unverändert |

Kein Urteil ändert sich. Der Code ist derselbe wie in B2.

## Primärtest: LB − LBhM bei w8a4_best (gleiche enge Grenze)

| Zelle | Median | Siege/Niederlagen | Holm-p | Urteil |
|---|---|---|---|---|
| MNIST d100 | −0,50 pp | 8/4 | 0,176 | unklar |
| Fashion d100 | −0,82 pp | 10/2 | 0,021 | besser |
| MNIST d200 | −1,62 pp | 12/0 | 0,002 | besser |
| Fashion d200 | −1,17 pp | 9/3 | 0,085 | unklar |

- **Gesamturteil:** „besser als LBhM“. Die Regel wie in B1/B2 verlangt mindestens 2 Zellen in eine Richtung und keine in die andere.
- **H1** (fp32 LB − LBhM höchstens +0,5 pp je Zelle): hält. Die Werte sind −0,28, −0,56, −1,09 und −1,06 pp.
- **Erfolg: ja.** Die Vorab-Erwartung dafür lag bei 0,35.

## Sekundär (Holm je Familie über 4 Zellen)

| Nr. | Vergleich | MNIST d100 | Fashion d100 | MNIST d200 | Fashion d200 | Urteil |
|---|---|---|---|---|---|---|
| S1 | LBM − LBh, w4 (Form, weite Grenze) | −0,63 (9/3) | −0,53 (8/4) | −0,26 (7/5) | −1,90 (9/3) | kein Effekt (alle unklar, Holm 0,26) |
| S2 | LB − LBM, w4 (Grenze, weich) | −0,30 (8/3) | −0,40 (8/4) | −0,90 (7/5) | +0,76 (4/8) | kein Effekt |
| S3 | LBhM − LBh, w4 (Grenze, hart) | −0,10 (7/5) | +0,01 (6/6) | +0,13 (5/7) | +0,95 (5/7) | kein Effekt |
| S4 | LB − LBhM, fp32 | −0,28 (9/3) | −0,56 (11/1)* | −1,09 (10/2)* | −1,06 (10/1)* | besser als LBhM |
| S5 | LBM − LBh, fp32 | −0,44 (9/3)* | −0,53 (10/2)* | −0,25 (8/4) | −1,48 (9/3) | besser als LBh |
| S6 | LB − LBhM, w4a | −0,66 (10/2)* | −0,85 (10/2)* | −1,69 (11/1)* | −1,55 (11/1)* | besser als LBhM (4 von 4) |
| S7 | LB − LBh, w4 (B2 wiederholt) | −0,81 (10/2)* | −0,69 (9/3)* | −0,78 (11/1)* | −2,70 (9/3) | besser als LBh |

Mediane in pp, dazu Siege/Niederlagen. Ein * markiert Holm-p < 0,05.

## Zerlegung (beschreibend, Median über Seeds, pp)

| Zelle | Metrik | Form-Effekt | Grenz-Effekt | Wechselwirkung |
|---|---|---|---|---|
| MNIST d100 | w4 | −0,28 | −0,39 | −0,12 |
| MNIST d100 | fp32 | −0,46 | +0,14 | −0,05 |
| Fashion d100 | w4 | −0,58 | −0,17 | −0,24 |
| Fashion d100 | fp32 | −0,51 | +0,06 | −0,20 |
| MNIST d200 | w4 | −0,87 | −0,07 | −1,39 |
| MNIST d200 | fp32 | −0,55 | +0,20 | −0,95 |
| Fashion d200 | w4 | −1,87 | −0,55 | +1,19 |
| Fashion d200 | fp32 | −1,19 | −0,42 | +0,39 |

- **Form-Effekt:** in 8 von 8 Kombinationen negativ, die weiche Form ist also besser. Er beträgt 0,3–1,9 pp und wächst mit der Tiefe.
- **Grenz-Effekt:** klein (bis 0,55 pp) und mit wechselndem Vorzeichen.
- **Wechselwirkung:** wechselt das Vorzeichen (−1,39 bis +1,19). Ein klares Muster „die Form zählt nur bei enger Grenze“ ist darin nicht zu sehen.

## Beschreibende Mediane (Testfehler in %)

| Datensatz | Tiefe | Form | fp32 | w8a4 | w8a4a | w8a3 | w8a3a | Grenze M |
|---|---|---|---|---|---|---|---|---|
| MNIST | d100 | LB | 5,22 | 9,33 | 9,04 | 48,1 | 41,5 | 2,217 |
| MNIST | d100 | LBhM | 5,67 | 9,76 | 9,80 | 47,5 | 40,9 | 2,217 |
| MNIST | d100 | LBM | 5,11 | 9,60 | 9,43 | 48,9 | 42,8 | 2,299 |
| MNIST | d100 | LBh | 5,74 | 10,34 | 9,97 | 50,8 | 43,6 | 2,299 |
| Fashion | d100 | LB | 15,50 | 19,08 | 19,36 | 51,7 | 48,2 | 2,217 |
| Fashion | d100 | LBhM | 15,98 | 19,61 | 20,11 | 51,4 | 47,0 | 2,217 |
| Fashion | d100 | LBM | 15,38 | 19,33 | 19,52 | 51,9 | 47,0 | 2,299 |
| Fashion | d100 | LBh | 15,98 | 19,61 | 20,43 | 52,3 | 47,8 | 2,299 |
| MNIST | d200 | LB | 7,22 | 12,00 | 11,24 | 57,2 | 51,1 | 2,454 |
| MNIST | d200 | LBhM | 7,80 | 13,05 | 13,03 | 55,5 | 49,6 | 2,454 |
| MNIST | d200 | LBM | 7,43 | 12,28 | 12,20 | 56,6 | 51,9 | 2,537 |
| MNIST | d200 | LBh | 7,75 | 12,47 | 12,25 | 56,8 | 51,4 | 2,537 |
| Fashion | d200 | LB | 18,10 | 22,27 | 22,61 | 58,8 | 54,3 | 2,454 |
| Fashion | d200 | LBhM | 19,23 | 23,84 | 24,84 | 58,3 | 54,8 | 2,454 |
| Fashion | d200 | LBM | 17,57 | 22,05 | 21,88 | 57,9 | 56,0 | 2,537 |
| Fashion | d200 | LBh | 19,78 | 24,44 | 24,56 | 58,5 | 55,2 | 2,537 |

- **Bei 4 Bit** liegen die weichen Formen (LB, LBM) in jeder Zelle vor den harten (LBhM, LBh).
- **Bei 3 Bit** ist das nicht so (nicht getestet). Dort ist LBhM, also hart und eng, in MNIST beschreibend am besten. 3 Bit bleibt für alle Formen unbrauchbar (Fehler über 40 %).
- **Sättigung:** Die Raten der weichen und der harten Formen sind nicht direkt vergleichbar. Weich zählt ab |Kern| > ρM, hart ab |Kern| > M. Sie liegen bei etwa 0,14–0,15 (weich) und 0,07–0,08 (hart).

## Deutung nach den Vorab-Regeln

1. **Primär „besser“ und H1 hält: Erfolg.** Die weiche Lagma-Form ist bei gleicher Grenze bei 4 Bit besser als der harte Clip, und das ohne Preis in fp32.
2. **S1 „kein Effekt“:** Die Vorab-Tabelle deutet das als „die Form zählt nur bei enger Grenze, also bei mehr Sättigung“. So steht es im Ergebnis. Meine Einschätzung dazu, getrennt von der Regel:
   - S1 hat alle 4 Mediane negativ (zusammen 33 Siege gegen 15 Niederlagen).
   - Die Wechselwirkung hat kein festes Vorzeichen.
   - Wahrscheinlicher ist ein allgemeiner Form-Effekt, der bei weiter Grenze kleiner ist und mit 12 Seeds je Zelle nicht sicher nachweisbar. Bewiesen ist das nicht.
3. **S2 und S3 „kein Effekt“:** Der engere Bereich erklärt den B2-Vorteil nicht. Die Deutung „nur die Grenze zählt“ trifft nicht zu.
4. **S7 „besser“:** B2 wiederholt sich auf neuen Seeds.
5. **Vorbehalt χ1:** In beiden Form-Vergleichen hat die weiche Form das kleinere χ1 (LB 1 + 2/L gegen LBhM ≈ 1 + 2,5/L). Ein Hinweis dagegen, dass χ1 den Effekt trägt: S2 und S3 verändern innerhalb einer Form χ1 um ±0,4–0,5/L (zusammen mit M), und beide zeigen keinen Effekt. Ein χ1-Unterschied dieser Größe wirkt bei 4 Bit also nicht messbar. Getrennt ist das nicht, M und χ1 ändern sich in S2 und S3 gemeinsam. Bewertung: plausibel, nicht bewiesen.

## Abgleich mit der Vorab-Erwartung

| Größe | Erwartung | Ergebnis |
|---|---|---|
| Primär „besser“ | 0,35 | eingetreten |
| Primär „kein Effekt“ oder „gemischt“ | 0,5 | nein |
| Primär „schlechter“ | 0,15 | nein |
| H1 hält | 0,75 | ja |
| S2 oder S3 „besser“ | 0,4 | nein |
| S7 „besser“ | 0,7 | ja |
| Kontrolle bitgleich | 0,95 | ja |

Das Ergebnis fällt besser für die Lagma-Form aus, als ich vorher erwartet hatte.

## Antwort auf die Frage

Der B2-Vorteil der weichen Lagma-Hülle bei 4 Bit kommt von der Form, nicht von der um 3,5 % engeren Grenze. Bei gleicher Grenze ist weich besser als hart: um 0,5–1,6 pp bei w8a4, um 0,7–1,7 pp bei w8a4a und auch in fp32.

**Vertrauen: 0,75.** Für die Kernaussage „die Form trägt bei enger Grenze“ liegt es höher (etwa 0,85), für „sie trägt auch bei weiter Grenze“ niedriger (etwa 0,6).

## Grenzen

- MNIST und Fashion, MLP ohne Residuen, LR nicht getunt (wie B2), nur Quantisierung nach dem Training.
- χ1 ist in den neuen Armen nicht exakt kritisch (siehe Deutung, Punkt 5).
- Nur eine Weichheitsstufe (ρ = 0,8).
- 3 Bit wurde nicht getestet und ist für alle Formen unbrauchbar.
