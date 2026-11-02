# B8 – Auswertung: Hält der B6-Befund bei Tiefe 200? (BPreCap im TAT-Rezept gegen TAT, CIFAR-10)

Stand: 31.10.2026, 10:50. Die Auswertung folgt strikt `B8_Praeregistrierung.md`. PreCap, die Tiefenfamilie und Lagma sind Aarons Ideen. Lauf und Auswertung habe ich mit Claude gemacht.

## Kurz

- **Erfolg.** Im TAT-Rezept ist BPreCap auch bei 200 Schichten bei fairen 4 Bit besser als TAT: w8a4a −0,98 pp, 11/1, p = 0,0015.
- **In fp32 gleichauf mit TAT** (+0,30 pp, 5/7, unklar). Bei d100 (B6) lag TAT noch 1,0 pp vorn.
- **Bei symmetrischen 4 Bit deutlich besser** als TAT (−3,34 pp, 12/0).
- **Nuance:** Bei d200 ist **LB_tr bei 4 Bit noch besser** als BPreCap (w4a +0,38, w4 +3,47 pp zugunsten von LB_tr), in fp32 aber 1,40 pp schlechter. Es gilt wieder die Front aus p9/p10: Der PreCap-Kern gewinnt in fp32, die engere Grenze von LB gewinnt bei wenigen Bits.
- **Ehrlich dazu:** Bei 200 Schichten ist 4 Bit für alle Formen teuer, etwa +11 bis +15 pp gegenüber fp32.

## Integrität

| Prüfung | Ergebnis |
|---|---|
| Läufe | Tuning 48 von 48 ok, Bestätigung 37 von 37 ok |
| Hash-Prüfung | 18 von 18 OK |
| Kontrolle | kL2B MNIST d100 Seed 42060: bitgleich zu B1 |
| Cache-Grenze | Mechanik-Test im Startskript: bitgleich mit und ohne Grenze, Cache 0,54 statt 2,51 GB. Worker je etwa 2,2 GB statt 4,1 GB |
| Schlaf | Größte Lücke 114 s bzw. 106 s. Kein Schlaf |
| Laufzeit | 09:53–10:42 |
| Sicherung | `sicherung/b8_2026-10-31/`, 15 von 15 Dateien OK |

Befehl: `.venv/bin/python -I b8_analyze.py ~/…/studien/b8 ~/…/studien/b1`, Ergebnis in `analysis/b8_result.json`.

## LR-Tuning (mediane Val-Genauigkeit in %, d200)

| Form | 5e-6 | 1e-5 | 2e-5 | 5e-5 | 1e-4 | 2e-4 | Wahl |
|---|---|---|---|---|---|---|---|
| kL2B_tr | 44,8 | 46,2 | **48,1** | 43,8 | 13,3 | 13,4 | 2e-5 |
| LB_tr | 40,2 | 43,2 | **45,9** | 41,9 | 16,1 | 13,9 | 2e-5 |

- Keine Wahl liegt am Rand. Im Rezept vertragen beide Formen bei d200 die doppelte LR wie mit Gauß-Init (B5: 1e-5 bzw. 7e-6), gleich wie TAT (2e-5).

## Primär und sekundär (d200, 12 Seeds)

| Test | Vergleich | Median | Siege/Niederlagen | p bzw. Holm | Urteil |
|---|---|---|---|---|---|
| **Primär** | kL2B_tr − TAT, w4a | **−0,98 pp** | 11/1 | 0,0015 | **besser** |
| S1 | kL2B_tr − TAT, fp32 | +0,30 | 5/7 | 0,47 | unklar |
| S2 | kL2B_tr − TAT, w4 | −3,34 | 12/0 | 0,004 | besser |
| S3 | LB_tr − TAT, w4a | −2,53 | 12/0 | 0,004 | besser |
| S4 | kL2B_tr − LB_tr, fp32 | −1,40 | 12/0 | 0,004 | besser |
| S5 | kL2B_tr − LB_tr, w4a | +0,38 | 2/10 | 0,010 | schlechter |
| S6 | LB_tr − TAT, fp32 | +1,52 | 0/12 | 0,004 | schlechter |
| S7 | LB_tr − TAT, w4 | −6,15 | 12/0 | 0,004 | besser |
| S8 | kL2B_tr − LB_tr, w4 | +3,47 | 0/12 | 0,004 | schlechter |

## Mediane (Testfehler in %, d200)

| Form | LR | fp32 | w8a4 | w8a4a | w8a3 | Bereich | 4-Bit-Preis (w4a − fp32) |
|---|---|---|---|---|---|---|---|
| kL2B_tr | 2e-5 | 52,7 | 69,5 | 66,2 | 89,3 | 3,0 | +13,5 |
| LB_tr | 2e-5 | 54,1 | **66,3** | **65,4** | 88,4 | 2,45 | +11,3 |
| TAT | 2e-5 | **52,5** | 72,3 | 67,8 | 89,8 | 5,09 | +15,3 |

**Zum Vergleich (nur Mediane, andere Seeds):**
- BPreCap mit Gauß-Init bei d200 (B5): fp32 61,3, w4a 69,0. Das Rezept bringt bei d200 also etwa **8,6 pp in fp32**, bei d100 (B6) waren es 3,9 pp. Die orthogonale Init hilft umso mehr, je tiefer das Netz ist (wie bei Pennington).
- TAT bei d200 in B5: 52,3 / 66,6, deckt sich mit B8.

## Deutung nach den Vorab-Regeln

- **Erfolg:** Der B6-Befund gilt auch bei d200. Im gleichen Rezept ist BPreCap bei fairen 4 Bit besser als TAT, in fp32 gleich.
- **S2 „besser“:** Bei symmetrischen 4 Bit ist BPreCap ebenfalls besser als TAT.
- **S4 „besser“:** Der Kern-Befund (PreCap-Kern besser als linearer Kern in fp32) gilt auch bei d200 im Rezept.
- **Zusatz:** Bei 4 Bit ist bei d200 LB_tr die beste Form (S3, S7, S5, S8). Für eine Empfehlung heißt das:
  - Bei d200 und 4 Bit ist die engere Grenze (LB) im Vorteil.
  - In fp32 und bei d100 gewinnt BPreCap.
  - Beide gehören zur selben Familie (Lagma-Hülle mit kritischem Kern). Die Wahl des Kerns ist ein Abwägen zwischen fp32-Güte und 4-Bit-Preis.

## Abgleich mit der Vorab-Erwartung

| Größe | Erwartung | Ergebnis |
|---|---|---|
| Primär „besser“ | 0,55 | eingetreten |
| Primär „unklar“ | 0,3 | nein |
| Primär „schlechter“ | 0,15 | nein |
| S1 „schlechter“ | 0,8 | **nein, unklar (gleichauf)** |
| S2 „besser“ | 0,85 | ja |
| S4 „besser“ | 0,8 | ja |
| eine _tr-LR am Rand | 0,2 | nein |
| Kontrolle bitgleich | 0,95 | ja |

## Antwort auf die Frage

Ja. Im gleichen Rezept ist BPreCap auch bei 200 Schichten bei 4 Bit besser als TAT (asymmetrisch −1,0 pp, symmetrisch −3,3 pp) und in fp32 gleichauf.

**Vertrauen: 0,8.** Einschränkungen: kurzes Budget (dazu B9 bei d100), ein Datensatz, TAT nicht neu getunt. Bei d200 ist die Form mit linearem Kern (LB_tr) bei 4 Bit noch etwas besser.

## Grenzen

- Kurzes Budget (1720 Schritte), nur Quantisierung nach dem Training.
- TAT mit der LR aus B3 (nicht am Rand), nicht im neuen Raster getunt.
- 4 Bit ist bei d200 für alle Formen teuer (+11 bis +15 pp). Für die Praxis ist das ein Vergleich auf schwachem Niveau.
