# B6 – Auswertung: Kommt der TAT-Vorsprung vom Rezept oder von der Form? (CIFAR-10, d100)

Stand: 31.10.2026, morgens. Die Auswertung folgt strikt `B6_Praeregistrierung.md`. PreCap, die Tiefenfamilie und Lagma sind Aarons Ideen. Lauf und Auswertung habe ich mit Claude gemacht.

## Kurz

- **Erfolg, und zwar deutlich.** Im TAT-Rezept wird BPreCap in fp32 um 3,87 pp besser (12/0, p = 0,0005). Das Rezept ist die SUO-orthogonale Init mit Normierung je Beispiel.
- **Der größte Teil des TAT-Vorsprungs kam vom Rezept.** In fp32 schrumpft der Abstand zu TAT von 4,9 pp (B3) auf **1,0 pp**. Er bleibt signifikant (S2 „schlechter“, 1/11): Ein Rest geht auf die Form zurück.
- **Bei fairen 4 Bit ist BPreCap im gleichen Rezept besser als TAT:** asymmetrisch −0,76 pp (10/2, S3), symmetrisch −2,59 pp (12/0, S8). Vorab hatte ich für S3 nur 0,15 erwartet. Es ist der erste Befund der Nacht, in dem eine Form aus Aarons Familie TAT bei 4 Bit schlägt, und zwar in beiden Quantisierungen.
- **Der Kern-Befund aus B3 gilt auch im TAT-Rezept:** BPreCap ist besser als LB in fp32 (−1,32 pp, 12/0, S5).
- **Auch LB profitiert:** −5,03 pp in fp32 (S1). LB_tr ist bei w4a gleichauf mit TAT (S4 „unklar“, −0,02 pp).

## Integrität

| Prüfung | Ergebnis |
|---|---|
| Läufe | Tuning 48 von 48 ok, Bestätigung 61 von 61 ok |
| Hash-Prüfung | 18 von 18 OK |
| Kontrolle | kL2B MNIST d100 Seed 42060: bitgleich zu B1 |
| Referenzen gegen B3 d100 | kL2B, LB und TAT auf neuen Seeds mit fast gleichen Medianen (siehe unten) |
| Schlaf | Größte Lücke 65 s bzw. 86 s. Kein Schlaf |
| Laufzeit | 04:29–05:13 |
| Sicherung | `sicherung/b6_2026-10-31/`, 15 von 15 Dateien OK |

Befehl: `.venv/bin/python -I b6_analyze.py ~/…/studien/b6 ~/…/studien/b1`, Ergebnis in `analysis/b6_result.json`.

## LR-Tuning der neuen Varianten (mediane Val-Genauigkeit in %)

| Form | 1e-5 | 2e-5 | 5e-5 | 1e-4 | 2e-4 | 5e-4 | Wahl |
|---|---|---|---|---|---|---|---|
| kL2B_tr | 45,0 | 47,2 | **48,1** | 44,8 | 13,6 | 13,6 | 5e-5 |
| LB_tr | 40,8 | 44,3 | **45,9** | 43,0 | 13,2 | 13,0 | 5e-5 |

- Keine Wahl liegt am Rand.
- Im TAT-Rezept vertragen beide Formen eine 2,5-mal größere LR als mit Gauß-Init (2e-5 in B3), gleich wie TAT (5e-5). Das passt zur besseren Konditionierung durch orthogonale Gewichte (Pennington 2017).

## Primär und sekundär (12 Seeds)

| Test | Vergleich | Median | Siege/Niederlagen | p bzw. Holm | Urteil |
|---|---|---|---|---|---|
| **Primär** | kL2B_tr − kL2B, fp32 | **−3,87 pp** | 12/0 | 0,0005 | **besser** |
| S1 | LB_tr − LB, fp32 | −5,03 | 12/0 | 0,004 | besser |
| S2 | kL2B_tr − TAT, fp32 | +1,01 | 1/11 | 0,007 | schlechter |
| S3 | kL2B_tr − TAT, w4a | **−0,76** | 10/2 | 0,010 | **besser** |
| S4 | LB_tr − TAT, w4a | −0,02 | 7/5 | 0,47 | unklar |
| S5 | kL2B_tr − LB_tr, fp32 | −1,32 | 12/0 | 0,004 | besser |
| S6 | kL2B_tr − kL2B, w4 | −4,16 | 12/0 | 0,004 | besser |
| S7 | kL2B_tr − kL2B, w4a | −4,54 | 12/0 | 0,004 | besser |
| S8 | kL2B_tr − TAT, w4 | **−2,59** | 12/0 | 0,004 | **besser** |

## Mediane (Testfehler in %, d100)

| Form | Rezept | LR | fp32 | w8a4 | w8a4a | w8a3 | Bereich | 4-Bit-Preis (w4a − fp32) |
|---|---|---|---|---|---|---|---|---|
| **kL2B_tr** | TAT | 5e-5 | 52,2 | 57,6 | **56,1** | 77,1 | 3,0 | +3,8 |
| LB_tr | TAT | 5e-5 | 53,7 | **56,7** | 56,7 | 71,5 | 2,22 | +3,1 |
| TAT | TAT | 5e-5 | **51,3** | 60,3 | 57,1 | 84,3 | 6,84 | +5,8 |
| kL2B | Gauß (B3) | 2e-5 | 56,2 | 61,9 | 60,6 | 79,4 | 3,0 | +4,5 |
| LB | Gauß (B3) | 2e-5 | 58,7 | 61,8 | 61,8 | 74,1 | 2,22 | +3,1 |

**Wiederholung der Referenzen** (B3 d100 gegen B6 auf neuen Seeds, fp32 / w4a):

| Form | B3 | B6 |
|---|---|---|
| kL2B | 56,4 / 60,9 | 56,2 / 60,6 |
| LB | 60,0 / 62,3 | 58,7 / 61,8 |
| TAT | 51,6 / 56,8 | 51,3 / 57,1 |

Die Werte sind sehr stabil.

## Deutung nach den Vorab-Regeln

- **„Erfolg und S2 schlechter“:** Das Rezept hilft, ein Teil des Abstands kommt aber von der Form (TAT).
  - Der Rest in fp32 liegt bei etwa 1 pp (vorher 4,9 pp). Etwa 80 % des fp32-Abstands aus B3 kamen also vom Rezept, nicht von der Form.
- **„S3/S8 besser“:** Mit gleichem Rezept ist BPreCap bei 4 Bit besser als TAT, symmetrisch und asymmetrisch.
  - Grund (Messung, nicht getestet): Der Bereich von BPreCap ist auf 3,0 beschränkt, der von TAT liegt bei 6,8. Der 4-Bit-Preis ist bei BPreCap kleiner (+3,8 gegen +5,8 pp).
- **„S5 besser“:** Der Kern-Befund aus B3 (PreCap-Kern besser als linearer Kern) gilt auch im TAT-Rezept.
- **Zusatz:** LB_tr hat den kleinsten 4-Bit-Preis (+3,1) und ist bei 3 Bit klar am besten (71,5 %, nicht getestet). Auch dort bleibt 3 Bit aber unbrauchbar.

## Abgleich mit der Vorab-Erwartung

| Größe | Erwartung | Ergebnis |
|---|---|---|
| Primär „besser“ | 0,55 | eingetreten |
| S1 „besser“ | 0,5 | ja |
| S2 „schlechter“ | 0,7 | ja |
| S3 „besser“ | 0,15 | **ja** |
| S5 „besser“ | 0,8 | ja |
| eine _tr-LR am Rand | 0,2 | nein |
| Kontrolle bitgleich | 0,95 | ja |

## Antwort auf die Frage

- Der fp32-Vorsprung von TAT kam auf CIFAR-10 zum größten Teil vom Rezept (orthogonale Init und Normierung je Beispiel). Zum kleineren Teil (etwa 1 pp) kommt er von der Form.
- Im gleichen Rezept ist BPreCap bei 4 Bit besser als TAT.

**Vertrauen: 0,8** für „im gleichen Rezept ist BPreCap bei 4 Bit mindestens gleich gut wie TAT“. Für „klar besser“ liegt es bei 0,65: S3 ist mit −0,76 pp klein, die Holm-Schwelle wird mit p = 0,010 deutlich unterschritten.

## Grenzen

- Init und Normierung wurden als Paket geändert, welcher Teil wirkt, bleibt offen.
- Nur d100, kurzes Budget (dazu B7, aber im B3-Rezept), MLP ohne Faltungen, ein Datensatz.
- Die Referenzen wurden nicht neu getunt, TAT ebenfalls nicht (5e-5 aus B3, dort nicht am Rand).
- TAT ist hier unsere Nachbildung nach dks (Zhang et al. 2022), Leaky-ReLU-TAT ohne weitere Tricks.
