# B11 – Präregistrierung: Hält der Befund auf CIFAR-100? (BPreCap gegen TAT, praxisübliches Regime, d100, privat)

Stand: 31.10.2026, abends, vor dem Start. PreCap, die Tiefenfamilie und Lagma sind Aarons Ideen. BPreCap (= kL2B) ist ein Arbeitsname. Das Design habe ich mit Claude entworfen. Aaron hat B11 und den Download von CIFAR-100 am 31.10. freigegeben, nichts veröffentlichen.

## Ausgangslage

- **B9:** Im TAT-Rezept ist BPreCap bei w4a −3,76 pp besser als TAT (12/0), fp32 gleichauf. Getestet auf CIFAR-10 mit Adam und konstanter LR.
- **B10** (läuft beim Freeze von B11 noch): dieselbe Frage mit AdamW, Kosinus-LR und getuntem WD.
- **Offener Einwand:** nur ein Datensatz. CIFAR-100 ist schwerer, mit 100 Klassen und 500 Bildern je Klasse.

## Frage

Ist BPreCap im TAT-Rezept auf CIFAR-100 (d100) bei fairen 4 Bit besser als TAT? Training wie B10 (AdamW, Kosinus, LR und WD je Form getunt), Messung am eigenen besten Validierungspunkt.

## Eine Änderung gegenüber B10: der Datensatz

- **CIFAR-100**, Binärversion, feine Klassen (100).
  - Geladen am 31.10. MD5 des Archivs `03b5dce01913d631647c71ecec9e9cb8`, stimmt mit der offiziellen Seite überein. SHA-256 der Dateien in `data/cifar100/SHA256.txt`.
  - Aufteilung wie CIFAR-10: die letzten 5000 Trainingsbilder als Validierung, 45 000 zum Training, 10 000 Test.
  - Normierung wie CIFAR-10: Mittelwert und Standardabweichung aus dem Trainingsteil, für die Rezept-Formen zusätzlich die Normierung je Beispiel.
- Die Ausgabeschicht hat 100 statt 10 Einheiten, gleiche Init-Regel. Die verdeckten Schichten sind bei gleichem Seed identisch zu B10.
- **Raster:** wie B10, ergänzt um LR 5e-4.
  - Grund: TAT wählte in B10 2e-4 am oberen Rand.
  - Festgelegt am 31.10. um 18:05, nur nach den B10-Tuningdaten. Die B10-Bestätigungs- und Testdaten waren da noch nicht ausgewertet.
  - LR-Spitze {5e-5, 1e-4, 2e-4, 5e-4} × WD {0; 0,3; 3,0}, 2 Tuning-Seeds, Val-Genauigkeit am Ende. Wahlregel wie B10.
- **Konstanten wie B10:** Formen, Rezept, Netz (d100, Breite 256), Batch 128, 6880 Schritte, Messpunkte alle 860, Kalibrierung (1000 Trainingsbilder), Quantisierung, Cache-Grenze, frühes Stoppen.
- **Seeds:** neu 42180–42191.
- **Ausfall:** zählt als 99 % Fehler (Zufall bei 100 Klassen) statt 90 % wie bei CIFAR-10.

## Läufe

- Tuning: 3 Formen × 12 Kombinationen × 2 Seeds = 72.
- Bestätigung: 3 Formen × 12 Seeds = 36.
- **Kontrolle:** kL2B_tr und TAT auf CIFAR-10, Seed 42168, mit den B10-Hyperparametern (`hp_from_b10.csv`). Alle Kennzahlen an allen Messpunkten müssen bitgleich zu B10 sein.
- Zusammen 110 Läufe, etwa 3,5 h. Start nach B10, nur am Netzteil.

## Primärtest

**kL2B_tr − TAT bei w8a4a_best am Punkt des frühen Stoppens** (CIFAR-100). Wilcoxon zweiseitig exakt, eine Zelle, α = 0,05.

**Erfolg:** p < 0,05 und Median < 0 („besser“).

## Sekundär (Holm über alle 8, wie B9 und B10)

| Nr. | Vergleich |
|---|---|
| S1 | kL2B_tr − TAT, fp32, frühes Stoppen |
| S2 | kL2B_tr − TAT, w8a4_best, frühes Stoppen |
| S3 | LB_tr − TAT, w8a4a_best, frühes Stoppen |
| S4 | kL2B_tr − LB_tr, fp32, frühes Stoppen |
| S5 | kL2B_tr − LB_tr, w8a4a_best, frühes Stoppen |
| S6 | kL2B_tr − TAT, w8a4a_best, Ende |
| S7 | kL2B_tr − TAT, fp32, Ende |
| S8 | LB_tr − TAT, fp32, frühes Stoppen |

**Beschreibend:** gewählte Hyperparameter und Lage am Rand, Stopp-Punkte, Verlauf, Bereich, 4-Bit-Preis.

## Deutungsregeln (vorab)

| Ergebnis | Deutung |
|---|---|
| Erfolg | Der 4-Bit-Vorteil gilt auf einem zweiten, schwereren Datensatz |
| Primär „unklar“ | Auf CIFAR-100 ist kein Vorteil nachweisbar. Die Aussage für v0.3 bleibt auf CIFAR-10 beschränkt |
| Primär „schlechter“ | Auf CIFAR-100 ist TAT bei 4 Bit vorn. Die Aussage hängt am Datensatz |
| B10 und B11 widersprechen sich | Beide werden gemeinsam berichtet, keiner wird weggelassen |
| Kontrolle nicht bitgleich | Abweichung wird gesucht und berichtet, die Tests gelten |

## Vorab-Erwartung

| Größe | Erwartung |
|---|---|
| Primär „besser“ | 0,5 |
| Primär „unklar“ | 0,4 |
| Primär „schlechter“ | 0,1 |
| S1 „unklar“ (fp32 gleichauf) | 0,5 |
| S2 „besser“ | 0,75 |
| 4-Bit-Preis von TAT am Stopp-Punkt größer als bei BPreCap (Mediane) | 0,75 |
| Kontrolle bitgleich | 0,95 |

## Grenzen

- Nur 2 Tuning-Seeds. Das Raster wurde nach dem B10-Tuning erweitert (Regel und Zeitpunkt oben).
- Nur d100, MLP ohne Faltungen, nur Quantisierung nach dem Training.
- Kalibrierung mit 1000 Bildern. Bei 100 Klassen sind das etwa 10 je Klasse, gleich wie bei CIFAR-10 in absoluter Zahl.
- Absolutes Niveau bei CIFAR-100 mit einem MLP sehr schwach (Fehler vermutlich 75–80 %). Es geht um den Vergleich.

## Freeze

Eingefroren werden `b11_run.py`, `b11_analyze.py`, `test_b11_mechanik.py`, `test_b11_analyse.py`, diese Datei, `hp_from_b10.csv`, `src/precap/*.py` und die Dateien von CIFAR-10 und CIFAR-100.

Vorab gemachte Tests:

- **Selbsttest:** n_confirm 38, n_tune 72.
- **Auswertung an künstlichen Daten** (`test_b11_analyse.py`): AUSWERTUNG OK.
  - Erfolg, „schlechter“, Null mit 20 Wiederholungen (1/1 Fehlalarm), falsche Kontrolle am Ende und an einem Messpunkt, Ausfallregeln mit 99 %, relativer Pfad.
  - **Offen genannt:** Meine erste Prüfung der 99-%-Regel im Test war falsch nachgerechnet. Ich habe sie durch einen Bereichstest ersetzt: 42,2 statt etwa 33 bei 90 %. Die Auswertung selbst war richtig.
- **Mechanik** (`test_b11_mechanik.py`): läuft im Startskript nach dem Ende von B10, weil der Speicher neben den B10-Workern knapp ist. Bei Fehler bricht das Skript ab. Geprüft wird:
  - Formen und Init gleich B10, auch bei 100 Ausgaben.
  - CIFAR-100: Formen, alle 100 Klassen mit 500 bzw. 100 Bildern, Normierung, Normierung je Beispiel.
  - Raster, Laufzahlen, Seeds, Kontrolle mit den B10-Werten.
  - CIFAR-10-Pfad bitgleich zu B10 (mit Kosinus, WD und Zwischenmessung).
  - Kurzer CIFAR-100-Lauf ok. Die Zwischenmessung verändert das Training nicht.
- **Rauchtest:** im Startskript.
