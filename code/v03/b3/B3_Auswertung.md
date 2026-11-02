# B3 – Auswertung: Trägt der PreCap-Kern auf CIFAR-10 etwas bei?

Stand: 31.10.2026, 01:45. PreCap, die Tiefenfamilie und Lagma sind Aarons Ideen. BPreCap (= kL2B) ist ein Arbeitsname. Die Auswertung habe ich mit Claude gemacht, strikt nach `B3_Praeregistrierung.md` (eingefroren 30.10. 22:42:51). Lauf: 22:45 bis 01:19, ohne Schlafpause (`pmset -g log` ohne Sleep-Einträge, größte Lücke zwischen zwei Läufen 131 s).

## Integrität

| Prüfung | Ergebnis |
|---|---|
| Läufe | Tuning 192/192 ok, Bestätigung 97/97 ok, 0 Ausfälle |
| Hash-Prüfung (Code, Präregistrierung, LR-Tabellen, Paket, 6 CIFAR-Dateien) | alle 17 OK nach dem Lauf |
| Kontrolle kL2B, MNIST d100, Seed 42060, gegen B1 | **bitgleich** |
| Sicherung | `sicherung/b3_2026-10-31/` mit `SHA256.txt` (16 Dateien, alle OK) |

Befehl: `.venv/bin/python -I studien/b3/b3_analyze.py ~/PyCharmMiscProject/studien/b3`. Das Ergebnis steht in `analysis/b3_result.json`.

## Lernraten (Tuning, mediane Val-Genauigkeit über 4 Seeds)

| LR | kL2B d100 | LB d100 | kL2 d100 | TAT d100 | kL2B d200 | LB d200 | kL2 d200 | TAT d200 |
|---|---|---|---|---|---|---|---|---|
| 1e-5 | 43,0 | 39,6 | 43,8 | 45,1 | **39,6** | **34,3** | **43,5** | 46,6 |
| 2e-5 | **43,6** | **40,4** | **45,4** | 47,9 | 36,2 | 14,9 | 43,3 | **47,6** |
| 5e-5 | 39,8 | 19,3 | 42,8 | **48,7** | 14,5 | 13,0 | 13,2 | 46,5 |
| 1e-4 | 13,3 | 15,0 | 15,9 | 47,0 | 13,7 | 12,6 | 12,1 | 38,8 |
| 2e-4 | 13,1 | 14,1 | 10,8 | 38,6 | 13,2 | 12,3 | 10,2 | 18,5 |
| 5e-4 | 13,4 | 11,8 | 10,2 | 14,0 | 13,0 | 12,9 | 9,8 | 10,0 |

- **Randlage:** Bei d200 liegen kL2B, LB und kL2 am unteren Rand (1e-5). Nach der Vorab-Regel ist das eine Einschränkung, und es folgt ein Nachtuning in einer eigenen Studie (B5).
- **Auffällig:** Die Formen nach Gauß-Init haben eine scharfe Stabilitätskante. LB fällt bei d200 schon bei 2e-5 auf 15 % zusammen, kL2B hält dort noch 36 %, kL2 43 %. TAT ist über eine Dekade stabil.

## Primärtest (kL2B − LB, fp32, Holm über 2 Zellen)

| Zelle | Median | G/V | Holm | Urteil |
|---|---|---|---|---|
| CIFAR-10 d100 | **−2,95 pp** | 12/0 | 0,001 | besser |
| CIFAR-10 d200 | **−4,79 pp** | 12/0 | 0,001 | besser |

**Gesamturteil: besser als LB. Erfolg nach der Vorab-Regel.** Vorab hatte ich das mit 0,3 erwartet.

## Sekundär (je Familie Holm über 2 Zellen)

| Nr. | Vergleich | d100 | d200 | Gesamt |
|---|---|---|---|---|
| S1 | kL2B − LB, w8a4_best | −0,67 (9/3) | −0,03 (6/6) | **kein Effekt** |
| S2 | kL2B − kL2, fp32 | **+2,12** | **+5,04** | schlechter als kL2 |
| S3 | kL2B − kL2, w8a4_best | **−5,60** | **−10,12** | besser als kL2 |
| S4 | kL2B − TAT, fp32 | **+4,89** | **+9,30** | schlechter als TAT |
| S5 | kL2B − TAT, w8a4a_best | **+4,09** | **+1,49** | schlechter als TAT |
| S6 | LB − TAT, w8a4a_best | **+5,38** | **+4,03** | schlechter als TAT |
| S7 | kL2B − LB, w8a3_best | **+5,27** | **+2,21** | schlechter als LB |
| S8 | kL2 − LB, fp32 | **−5,26** | **−9,23** | besser als LB |

Mediane in pp, fett heißt nach Holm signifikant. Alle signifikanten Zellen haben 12/0 oder 0/12.

**Vorhersage p8 (fp32-Ordnung kL2 < kL2B < LB):** gilt in beiden Tiefen. Die volle Ordnung ist TAT < kL2 < kL2B < LB.

## Beschreibend (Median über 12 Seeds, Testfehler in %)

| Tiefe | Form | fp32 | w8a4 | w8a4a | w8a3 | 4-Bit-Kosten (w8a4a − fp32) | Bereich | Sättigung |
|---|---|---|---|---|---|---|---|---|
| d100 | kL2B | 56,4 | 62,0 | 60,9 | 79,2 | +4,5 | 3,00 | 5,5 % |
| | LB | 60,0 | 62,6 | 62,3 | 74,1 | +2,3 | 2,22 | 16,7 % |
| | kL2 | 54,1 | 67,8 | 65,3 | 86,8 | +11,2 | 6,75 | – |
| | TAT | **51,6** | **60,1** | **56,8** | 84,2 | +5,2 | 6,86 | – |
| d200 | kL2B | 61,1 | **69,9** | 68,3 | 86,4 | +7,2 | 3,00 | 6,0 % |
| | LB | 65,6 | 70,4 | 70,8 | 84,0 | +5,2 | 2,45 | 12,1 % |
| | kL2 | 56,3 | 79,7 | 79,4 | 89,7 | +23,1 | 7,85 | – |
| | TAT | **52,0** | 72,6 | **66,5** | 89,8 | +14,5 | 5,22 | – |

Zum Vergleich: Ein linearer Klassifikator schafft auf CIFAR-10 etwa 40 % Genauigkeit, also etwa 60 % Fehler. **LB liegt bei d200 mit 65,6 % Fehler darunter**, kL2B bei d200 etwa gleichauf.

## Vorab-Erwartung gegen Ergebnis

| Größe | Erwartung | Eingetreten |
|---|---|---|
| Primär „besser als LB“ (Erfolg) | 0,3 | **ja** |
| Primär „schlechter als LB“ | 0,15 | nein |
| S2 „schlechter als kL2“ | 0,6 | ja |
| S3 „besser als kL2“ | 0,9 | ja |
| S4 „schlechter als TAT“ | 0,75 | ja |
| S5 „besser als TAT“ | 0,35 | **nein** (schlechter) |
| S6 „besser als TAT“ | 0,4 | **nein** (schlechter) |
| S8 „besser als LB“ | 0,5 | ja |
| p8-Ordnung gilt in beiden Tiefen | 0,3 | **ja** |
| Kontrolle bitgleich | 0,95 | ja |

## Deutung nach den Vorab-Regeln

1. **Erfolg:** Der PreCap-Kern trägt auf schwereren Daten etwas bei, in fp32 um 3–5 pp gegenüber dem linearen Kern unter derselben Hülle, in 24 von 24 Paaren. BPreCap ist damit mehr als „kritisch und begrenzt“. Nach der Regel ist es ein Kandidat für v0.3, nach Wiederholung.
2. **S8 „besser“ und p8-Ordnung gilt:** Mehr Nichtlinearität bei der Init nützt auf CIFAR. Die Budget-Rechnung p8 hat die Reihenfolge richtig vorhergesagt. Das passt zum Befund aus Nonlinear Advantage (frühe Linearität bremst das Training).
3. **S3 „besser“:** Die Grenze hilft auch auf CIFAR bei 4 Bit deutlich (5,6 bzw. 10,1 pp gegenüber kL2).
4. **S5/S6 „schlechter“:** Bei fairer 4-Bit-Quantisierung schlägt keine der begrenzten Formen TAT. Ein eigener Beitrag gegenüber dem Stand der Technik ist auf CIFAR **nicht** gezeigt.
5. **LR am Rasterrand** (d200, drei Formen): Einschränkung, Nachtuning folgt in B5.

## Eigene Einordnung (ehrlich, mit Claude)

- **Was B3 zeigt:** Auf CIFAR ist der Kern nicht egal. In fp32 gilt klar: mehr Nichtlinearität ist besser (kL2 < kL2B < LB). Bei 4 Bit hebt sich der fp32-Vorsprung von kL2B gegenüber LB aber fast auf (S1 kein Effekt), weil kL2B etwa 2 pp mehr Quantisierungskosten hat. Bei 3 Bit ist LB besser (S7).
- **Das ist dasselbe Bild wie in v0.2, nur verschoben:** Nichtlinearität hilft fp32 und kostet Bits. BPreCap liegt zwischen kL2 und LB. Die Kopplung ist durch die Grenze nicht aufgehoben, sie ist **gemildert**: kL2B hat bei d200 nur +7,2 pp 4-Bit-Kosten, kL2 +23,1 und TAT +14,5.
- **TAT bleibt vorn:** TAT ist auf CIFAR in fp32 5–9 pp besser und auch bei asymmetrischen 4 Bit besser (1,5 pp bei d200, 4,1 pp bei d100). Nur bei symmetrischen 4 Bit und d200 liegt kL2B vor TAT (69,9 gegen 72,6). Das war kein Vorab-Test und zählt nur beschreibend.
- **Vorbehalt Rezept:** TAT läuft mit orthogonaler Init und Normierung je Beispiel, die anderen mit Gauß-Init. Der TAT-Vorsprung kann teils vom Rezept kommen. Das ist dieselbe Einschränkung wie in v0.2.
- **Vorbehalt Lernrate:** Bei d200 sitzen kL2B, LB und kL2 am unteren Rand. Die Stabilitätskante ist bei LB am schärfsten. Mit kleinerer LR könnte vor allem LB aufholen. Ob der Primärbefund bei d200 hält, klärt erst B5.
- **Vorbehalt Budget:** 1720 Schritte (etwa 5 Epochen) und Genauigkeiten um 35–48 %. Gemessen wird also die frühe Trainingsphase. Genau dort sollte Nichtlinearität nach Nonlinear Advantage am meisten zählen. Ob der Vorteil bei längerem Training bleibt, ist offen.

## Grenzen

- MLP ohne Faltungen, schwaches Modell für CIFAR, ein Datensatz, zwei Tiefen, 12 Seeds.
- Kurzes Budget, LR am Rand bei d200.
- Unterschiedliches Rezept bei TAT.
- Nur Quantisierung nach dem Training.

## Folgen

- **BPreCap hat eine Chance:** auf schwereren Daten in fp32 klar vor LB, bei 4 Bit gleichauf, bei 3 Bit dahinter. Das ist der erste Befund, in dem der PreCap-Kern unter der Lagma-Hülle einen messbaren eigenen Wert hat.
- **Nächster Schritt:** B5 (Nachtuning d200 mit kleineren LR, neue Seeds) nach der Vorab-Regel. Danach wäre CIFAR-100 der stärkere Test (Download nur mit Aarons Freigabe).
