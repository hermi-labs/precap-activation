# B10 – Präregistrierung: Hält B9 mit praxisüblichem Trainingsregime? (Kosinus-LR und Gewichtsabfall, CIFAR-10, d100, privat)

Stand: 31.10.2026, nachmittags, vor dem Start. PreCap, die Tiefenfamilie und Lagma sind Aarons Ideen. BPreCap (= kL2B) ist ein Arbeitsname. Das Design habe ich mit Claude entworfen. Aaron hat am 31.10. nachmittags B10 und B11 freigegeben, nichts veröffentlichen.

## Ausgangslage

- **B9:** Im TAT-Rezept, mit 6880 Schritten und frühem Stoppen ist BPreCap bei w4a −3,76 pp besser als TAT (12/0) und in fp32 gleichauf.
- Ursache laut B9: Der Bereich von TAT wächst beim Training (6,2 → 8,2), der von BPreCap ist durch die Hülle fest (3,0).
- **Stärkster offener Einwand:** B9 trainiert mit Adam bei konstanter LR, ohne Gewichtsabfall. In der Praxis nutzt man AdamW mit LR-Abfall. Beides kann das Wachsen der Gewichte und damit des Bereichs bei TAT bremsen und so den 4-Bit-Vorteil von BPreCap aufheben.

## Frage

Ist BPreCap im TAT-Rezept auf CIFAR-10 (d100) bei fairen 4 Bit auch dann besser als TAT, wenn alle Formen mit AdamW, Kosinus-Abfall der LR und für jede Form getuntem Gewichtsabfall trainieren und am eigenen besten Validierungspunkt gemessen werden?

## Eine Änderung gegenüber B9: das Trainingsregime

- **AdamW** (MLX, entkoppelter Gewichtsabfall: w ← w · (1 − lr · wd) vor dem Adam-Schritt, auf alle Parameter) statt Adam.
- **Kosinus-Abfall** der LR von der Spitze auf 0 über alle 6880 Schritte, ohne Anlauf.
- **Tuning für alle drei Formen** (auch TAT, fair) im gleichen Raster:
  - LR-Spitze {5e-5, 1e-4, 2e-4} × WD {0; 0,3; 3,0} = 9 Kombinationen.
  - 2 Tuning-Seeds (42000, 42001), Kriterium: höchste mediane Val-Genauigkeit am Ende (fp32). Bei Gleichstand gewinnt die kleinere LR, dann der kleinere WD. Ausfall zählt als 10 % Genauigkeit.
  - Zur Einordnung von WD: Bei lr 1e-4 und wd 3 schrumpfen die Gewichte ohne Gradient über den Lauf etwa auf exp(−1) ≈ 0,37. Bei wd 0,3 auf etwa 0,9.
- **Konstanten wie B9:** Formen, Rezept (SUO-orthogonale Init × Gain, Normierung je Beispiel), Netz (d100, Breite 256), Batch 128, 6880 Schritte, Messpunkte alle 860 Schritte, Daten, Kalibrierung, Quantisierung, Cache-Grenze 512 MB.
- **Frühes Stoppen wie B9:** je Lauf der Messpunkt mit dem kleinsten Val-Fehler (fp32), bei Gleichstand der frühere. Test- und 4-Bit-Werte an diesem Punkt. Ausfallregeln wie B9.
- **Seeds:** neu 42168–42179.

## Läufe

- Tuning: 3 Formen × 9 × 2 = 54 Läufe (ohne Zwischenmessungen).
- Bestätigung: 3 Formen × 12 Seeds = 36 Läufe.
- **Kontrolle:** kL2B_tr und TAT, Seed 42156, konstante LR 5e-5, WD 0 über den neuen Code-Pfad (AdamW). Alle 7 Kennzahlen an allen 8 Messpunkten müssen bitgleich zu B9 sein.
- Zusammen 92 Läufe, etwa 3 h. Nur am Netzteil.

## Primärtest

**kL2B_tr − TAT bei w8a4a_best am Punkt des frühen Stoppens.** Wilcoxon zweiseitig exakt, eine Zelle, α = 0,05.

**Erfolg:** p < 0,05 und Median < 0 („besser“).

## Sekundär (Holm über alle 8, wie B9)

| Nr. | Vergleich |
|---|---|
| S1 | kL2B_tr − TAT, fp32, frühes Stoppen |
| S2 | kL2B_tr − TAT, w8a4_best, frühes Stoppen |
| S3 | LB_tr − TAT, w8a4a_best, frühes Stoppen |
| S4 | kL2B_tr − LB_tr, fp32, frühes Stoppen |
| S5 | kL2B_tr − LB_tr, w8a4a_best, frühes Stoppen |
| S6 | kL2B_tr − TAT, w8a4a_best, Ende (6880) |
| S7 | kL2B_tr − TAT, fp32, Ende (6880) |
| S8 | LB_tr − TAT, fp32, frühes Stoppen |

**Beschreibend:**
- gewählte LR und WD je Form, Lage am Rand
- Stopp-Punkte
- Verlauf über die Messpunkte
- Bereich (wächst er bei TAT noch?)
- 4-Bit-Preis
- Vergleich der Mediane mit B9 (andere Seeds, nur beschreibend)

## Deutungsregeln (vorab)

| Ergebnis | Deutung |
|---|---|
| Erfolg | Der 4-Bit-Vorteil hält auch mit praxisüblichem Regime. Er hängt nicht an konstanter LR oder fehlendem WD |
| Primär „unklar“ | Mit LR-Abfall und WD holt TAT bei 4 Bit auf. Der Vorteil aus B9 hing am einfachen Regime. Für v0.3 hieße das nur noch „robuster bei einfachem Training“ |
| Primär „schlechter“ | Mit praxisüblichem Regime ist TAT bei 4 Bit vorn |
| Bereich von TAT wächst nicht mehr | Der Mechanismus aus B9 wird durch WD und Kosinus abgeschaltet. Dann ist zu prüfen, ob ein verbleibender Vorteil anders erklärt werden muss |
| WD-Wahl am Rand (3,0) | Grenze nennen. Ein größerer WD könnte TAT weiter helfen |
| Kontrolle nicht bitgleich | Abweichung wird gesucht und berichtet, die Tests gelten |

## Vorab-Erwartung

| Größe | Erwartung |
|---|---|
| Primär „besser“ (Erfolg) | 0,55 |
| Primär „unklar“ | 0,35 |
| Primär „schlechter“ | 0,10 |
| S1 „unklar“ (fp32 gleichauf) | 0,6 |
| S2 „besser“ | 0,8 |
| Bereich von TAT am Stopp-Punkt kleiner als in B9 (Median < 7,5) | 0,6 |
| mindestens eine Wahl am Rand | 0,5 |
| Kontrolle bitgleich | 0,95 |

## Grenzen

- Nur 2 Tuning-Seeds, Raster 3 × 3.
- Kosinus ohne Anlauf, WD auf alle Parameter (auch Biases). Kein weiteres Regime (z. B. Label-Smoothing, Augmentierung).
- Nur Quantisierung nach dem Training. Nur d100, ein Datensatz, MLP ohne Faltungen. TAT ist unsere Nachbildung nach dks.

## Freeze

Eingefroren werden `b10_run.py`, `b10_analyze.py`, `test_b10_mechanik.py`, `test_b10_analyse.py`, diese Datei, `src/precap/*.py` und die CIFAR-Dateien.

Vorab gemachte Tests (31.10., nachmittags):

- **Selbsttest:** n_confirm 38, n_tune 54, 6880 Schritte.
- **Mechanik** (`test_b10_mechanik.py`): **MECHANIK OK.** Das Startskript führt den Test erneut aus und bricht bei Fehler ab. Geprüft wird:
  - Formen, Init und Daten gleich B9. Raster, Laufzahlen, Seeds, Spalten.
  - Die Wahlregel an künstlichen Tuning-Daten, mit Gleichstand und Ausfall.
  - AdamW mit konstanter LR und WD 0 ist bitgleich zu Adam in B9, auch an der Zwischenmessung.
  - Der Kosinus wirkt unter `mx.compile`: Die LR am Ende ist wie berechnet, das Ergebnis weicht von konstanter LR ab. WD wirkt.
  - Die Zwischenmessung verändert das Training mit Kosinus und WD nicht.
  - **Offen genannt:** Meine erste Prüfung der konstanten End-LR war zu eng (|lr − 5e-5| < 1e-12), weil 5e-5 als float32 4,99999987e-05 ist. Ich habe sie durch einen exakten Vergleich mit dem float32-Wert ersetzt. Die Bitgleichheit zu B9 hat diese Prüfung nicht berührt.
- **Auswertung an künstlichen Daten** (`test_b10_analyse.py`): AUSWERTUNG OK.
  - „BPreCap vorn“ wird als Erfolg erkannt, „TAT vorn“ als „schlechter“.
  - Null mit 20 Wiederholungen: 1 Fehlalarm primär, 1 sekundär.
  - Eine falsche Kontrolle wird erkannt, am Ende wie an einem einzelnen Messpunkt.
  - Die Ausfallregeln greifen. WD wird gelesen, der relative Pfad findet B9.
- **Rauchtest:** läuft im Startskript (Tuning 12 Läufe und Bestätigung 5 Läufe mit 40 Schritten, Zwischenwerte bei 10 und 20).
