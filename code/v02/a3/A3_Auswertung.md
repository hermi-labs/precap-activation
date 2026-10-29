# A3 – Auswertung: kL2 mit TAT-Rezept gegen TAT

*Veröffentlichte Fassung vom 29.10.2026: Der Bericht ist inhaltlich unverändert. Entfernt wurden nur der Vermerk „privat“ und Querverweise auf noch unveröffentlichte Studien. Präregistrierung und Skripte sind die eingefrorenen Originale.*

Stand: 28.10.2026, 01:40. Ausgewertet strikt nach `A3_Praeregistrierung.md`. Hinweis: Der Zeitstempel „23:45“ in der eingefrorenen Präregistrierung ist falsch. Der tatsächliche Start war 27.10., 23:16 (erste Zeile in `logs/w0.log`: 23:17:11). Die Präregistrierung war vor dem Start eingefroren.

## Integrität

```
$ (cd studien/a3 && shasum -a 256 -c meta/frozen_sha256.txt)
a3_run.py: OK
a3_analyze.py: OK
lr_tat_from_a2.csv: OK
A3_Praeregistrierung.md: OK
```

- Tuning 96/96 ok, Bestätigung 102/102 ok, 0 Ausfälle.
- Reproduktion TAT (MNIST d100, Seeds 42010/42011) gegen A2: 0,00 pp. Bestanden.
- Rohdaten gesichert in `sicherung/a3_2026-10-28/` (mit `SHA256.txt`, 10 Dateien).
- Auswertung: `.venv/bin/python -I studien/a3/a3_analyze.py studien/a3 studien/a2 studien/a` → `analysis/a3_result.json`.

## Primärtest: kL2r gegen TAT, fp32-Testfehler, Holm über 8

Differenz = kL2r − TAT, negativ heißt kL2r besser. W/L = Seeds, in denen kL2r gewinnt bzw. verliert.

| Budget | Zelle | n | Median-Diff | W/L | Holm-p | Urteil |
|---|---|---|---|---|---|---|
| 1720 | MNIST d100 | 15 | +0,12 pp | 5/10 | 1,00 | unklar |
| 1720 | Fashion d100 | 15 | −0,35 pp | 12/2 | 0,046 | besser |
| 1720 | MNIST d200 | 15 | −0,12 pp | 8/7 | 1,00 | unklar |
| 1720 | Fashion d200 | 15 | −0,77 pp | 14/1 | 0,0015 | besser |
| 6880 | MNIST d100 | 10 | −0,08 pp | 6/4 | 1,00 | unklar |
| 6880 | Fashion d100 | 10 | +0,42 pp | 1/9 | 0,027 | schlechter |
| 6880 | MNIST d200 | 10 | +0,16 pp | 3/7 | 0,80 | unklar |
| 6880 | Fashion d200 | 10 | −0,19 pp | 7/3 | 1,00 | unklar |

**Gesamturteil nach Regel: „gemischt“** (2 Zellen besser, 1 schlechter). Nach Budget getrennt (beschreibend): 1720 Schritte „besser als TAT“, 6880 Schritte „gemischt“. Laut Präregistrierung gibt es bei „gemischt“ keine Gesamtaussage, die Zellen werden einzeln berichtet.

Vorab-Erwartung für „gemischt“ war 0,15, für „schlechter“ 0,40. Das Ergebnis ist also günstiger für kL2 als erwartet.

## Sekundär S1: kL2r gegen kL2 mit Gauß-Init (aus A/A2)

| Budget | MNIST d100 | Fashion d100 | MNIST d200 | Fashion d200 |
|---|---|---|---|---|
| 1720 | −1,34 (15/0) besser | −2,25 (15/0) besser | −1,89 (15/0) besser | −2,93 (15/0) besser |
| 6880 | −0,38 (8/2) unklar | −0,36 (8/2) besser | −1,76 (10/0) besser | −1,76 (10/0) besser |

Urteil S1: „besser als kL2-Gauß“ in 7 von 8 Zellen. Das Rezept (SUO-Init und Normierung je Beispiel) bringt kL2 1,3 bis 2,9 pp bei 1720 Schritten. Das ist etwa so viel wie der Abstand zu TAT in A2 (1,5 bis 2,5 pp).

## Beschreibend

Mediane Testfehler (%) und größter Aktivierungsbereich am Ende:

| Budget | Zelle | kL2 (Gauß) | kL2r | TAT | Bereich kL2r | Bereich TAT |
|---|---|---|---|---|---|---|
| 1720 | MNIST d100 | 5,04 | 3,70 | 3,52 | 7,2 | 8,3 |
| 1720 | Fashion d100 | 15,62 | 13,28 | 13,58 | 6,4 | 7,5 |
| 1720 | MNIST d200 | 5,44 | 3,80 | 3,86 | 7,7 | 6,9 |
| 1720 | Fashion d200 | 16,53 | 13,55 | 14,04 | 7,4 | 5,8 |
| 6880 | MNIST d100 | 3,21 | 2,84 | 2,84 | 7,7 | 16,8 |
| 6880 | Fashion d100 | 13,08 | 12,53 | 12,07 | 6,9 | 11,7 |
| 6880 | MNIST d200 | 4,71 | 3,15 | 3,01 | 9,1 | 10,3 |
| 6880 | Fashion d200 | 14,73 | 12,93 | 12,98 | 8,5 | 4,9 |

- Gradientengefälle bei Init (log10): kL2r 0,52 bis 0,55, TAT −0,09 bis +0,03. Die Vermutung aus der Präregistrierung, das steilere Gefälle von kL2 verhindere einen Gleichstand, hat sich nicht bestätigt.
- Train-Loss: kL2r liegt bei 1720 Schritten meist unter TAT (z. B. Fashion d100 0,224 gegen 0,293).
- LR am Rasterrand: kL2r bei d200 (MNIST und Fashion) auf 1e-5, also am unteren Rand. TAT bei Fashion d200 ebenfalls (aus A2). Bei d200 ist das Optimum beider eventuell nicht erreicht.

## Bedeutung

- Der klare Abstand aus A2 (8/8 Zellen schlechter) kam **überwiegend vom Rezept**, nicht von der Aktivierung. Mit gleichem Rezept liegen kL2r und TAT in 5 von 8 Zellen ununterscheidbar, kL2r gewinnt 2 (kurzes Budget, Fashion) und verliert 1 (langes Budget, Fashion d100).
- Belastbar formulierbar: „Mit dem TAT-Rezept ist kL2 in fp32 ungefähr gleichauf mit TAT; kein systematischer Vorteil in eine Richtung.“ Nicht formulierbar: „kL2 ist besser als TAT.“
- Tendenz: Bei 6880 Schritten verschiebt es sich leicht Richtung TAT (3 von 4 Medianen gleich oder schlechter für kL2r). Das ist nicht signifikant, sollte aber genannt werden.
- Am Ende des langen Trainings bleibt der Wertebereich von kL2r enger (6,9 bis 9,1) als bei TAT bei d100 (11,7 bis 16,8), bei d200 nicht. Relevant für Q2, aber hier nur beschreibend.

## Grenzen

- Adam statt K-FAC für TAT (wie A2). Mit K-FAC kann TAT besser sein.
- LR-Raster bei d200 am Rand für kL2r und teils für TAT.
- 6880 Schritte nur mit 10 Seeds, daher weniger Power.
- MLPs, Breite 256, MNIST/Fashion.

## Vertrauen

0,8 für „mit gleichem Rezept etwa gleichauf“. Einschränkungen: LR am Rand bei d200, Adam statt K-FAC, nur zwei einfache Datensätze.
