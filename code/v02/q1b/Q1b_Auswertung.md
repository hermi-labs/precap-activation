# Q1b – Auswertung: faire Kalibrierung bei 4-Bit-Aktivierungen

*Veröffentlichte Fassung vom 29.10.2026: Der Bericht ist inhaltlich unverändert. Entfernt wurden nur der Vermerk „privat“ und Querverweise auf noch unveröffentlichte Studien. Präregistrierung und Skripte sind die eingefrorenen Originale.*

Stand: 27.10.2026.

## Kurzfassung

- **Gesamturteil nach Präregistrierung: "gemischt"**
  - 3 von 4 Zellen fallen "SPreCap robuster" aus, 1 Zelle (MNIST 50) "anfälliger".
- **Faire Kalibrierung:** Wenn jede Funktion ihre beste Kalibrierung bekommt (meist den MSE-Clip), schrumpft der Vorteil von SPreCap-cs gegenüber SELU stark. In Q1 waren es 6–19 pp, jetzt sind es noch 1,3–2,4 pp.
- **Tiefe 100:** Der Vorteil bleibt klar und einheitlich, 15/15 Seeds in beiden Datensätzen.
- **Tiefe 50:** Das Bild ist gemischt. Bei Fashion 50 ist cs robuster, bei MNIST 50 verliert SELU weniger.
- **Q1 für sich genommen:** Das Ergebnis aus Q1 ("robuster unter max-Kalibrierung") stimmt. Es lag aber zum großen Teil an der Kalibrierung und nicht nur an der Funktion.
- **Zuversicht:**
  - "Der Vorteil bleibt bei Tiefe 100 mit fairer Kalibrierung": 0,9
  - "Der Vorteil gilt allgemein": 0,3

## Prüfung

- **Läufe:** 240 von 240 OK, keine Ausfälle.
- **Hashes:** Die eingefrorenen Dateien (Skripte, Präregistrierung, LR-Tabelle, Paket) waren nach dem Lauf unverändert.
- **Wiederholbarkeit gegen Q1:** 960 von 960 Werten sind identisch (fp32, w8a8, w8a4, w4a8). Die maximale Abweichung beträgt 0,0 pp. Das Training auf MLX/Metal ist also bitgenau reproduzierbar.
- **Daten:** w0 sha256 6702600f…, w1 sha256 2a3eab60…

## Primärtest: w8a4_best, Δcs − Δselu

| Zelle | Median-Differenz | Siege/Niederlagen | Holm p | Urteil |
|---|---|---|---|---|
| MNIST 50 | +0,71 pp | 1/14 | 0,0006 | anfälliger |
| Fashion 50 | −1,33 pp | 15/0 | 0,0002 | robuster |
| MNIST 100 | −2,35 pp | 15/0 | 0,0002 | robuster |
| Fashion 100 | −1,92 pp | 15/0 | 0,0002 | robuster |

## Verlust je Funktion und Kalibrierung (Median Δ, pp)

| Zelle | Funktion | max (Q1) | p99,9 | p99,99 | MSE | beste |
|---|---|---|---|---|---|---|
| MNIST 50 | cs | 3,20 | 2,59 | 3,14 | 1,93 | 1,94 |
| | e03 | 9,73 | 5,52 | 7,66 | 3,53 | 3,53 |
| | SELU | 9,28 | 2,08 | 3,66 | 1,15 | 1,31 |
| | ELU | 10,97 | 5,74 | 6,92 | 2,83 | 2,83 |
| Fashion 50 | cs | 0,99 | 0,92 | 0,92 | 0,74 | 0,70 |
| | e03 | 9,25 | 5,35 | 7,16 | 3,59 | 3,60 |
| | SELU | 7,84 | 2,23 | 3,76 | 1,66 | 2,03 |
| | ELU | 31,14 | 21,43 | 29,02 | 7,06 | 7,06 |
| MNIST 100 | cs | 4,36 | 3,94 | 4,23 | 2,53 | 2,79 |
| | e03 | 8,78 | 5,30 | 7,63 | 3,30 | 3,30 |
| | SELU | 23,13 | 11,11 | 16,70 | 5,00 | 5,00 |
| | ELU | 48,16 | 42,63 | 45,44 | 19,77 | 19,77 |
| Fashion 100 | cs | 4,69 | 4,76 | 4,67 | 3,17 | 3,29 |
| | e03 | 8,58 | 6,00 | 7,44 | 4,01 | 4,41 |
| | SELU | 20,70 | 10,42 | 14,58 | 5,23 | 5,23 |
| | ELU | 69,74 | 64,43 | 68,94 | 40,90 | 40,90 |

**Was gewählt wurde:** Meist war es der MSE-Clip. Bei SELU auf Tiefe 50 wurde teils p99,9 gewählt. Die Wahl über die Kalibrierbilder ist darum manchmal etwas schlechter als MSE allein.

**Wo der MSE-Clip liegt (Median):** cs 2,7–2,9, SELU 3,6–4,0, e03 4,2–4,6, ELU 3,0–5,2. Auch für cs lohnt sich das Beschneiden noch, die tanh-Grenze bei 3,35 ist für 4 Bit also nicht optimal.

## Sekundär

- **e03 gegen SELU (beste):** Bei Tiefe 50 ist e03 anfälliger (+2,1 / +1,7 pp), bei Tiefe 100 robuster (−1,8 / −0,9 pp). Der breitere Bereich von e03 (Grenze weiter außen) schadet bei 4 Bit.
- **cs gegen ELU:** In allen 4 Zellen und bei jeder Kalibrierung ist cs robuster (−0,9 bis −38 pp). ELU bleibt bei Tiefe 100 auch mit MSE-Clip schlecht (20–41 pp Verlust).
- **Perzentil-Kalibrierung:** cs gegen SELU ist bei Tiefe 100 deutlich robuster (−5,6 bis −12 pp). Bei MNIST 50 mit p99,9 ist cs anfälliger.
- **w8a8 mit MSE-Clip:** Alles ist unklar, alle liegen bei etwa 0 pp Verlust.

## Einordnung

- **Q1 und Q1b zusammen:**
  - Die beidseitige Begrenzung macht tiefe MLPs ohne Normierung bei 4-Bit-Aktivierungen robuster, besonders bei Tiefe 100.
  - Mit fairer Kalibrierung ist der Vorteil gegenüber SELU klein (etwa 2 pp), aber bei Tiefe 100 sehr einheitlich.
  - Gegenüber ELU ist der Vorteil groß.
- **Grenzen:**
  - Die Quantisierung ist simuliert, pro Tensor, nach dem Training.
  - Getestet sind nur MLPs auf MNIST/Fashion.
  - Quantisierungsbewusstes Training (QAT) und pro-Kanal-Kalibrierung der Aktivierungen wurden nicht geprüft.
- **Für eine spätere Veröffentlichung:** Q1 und Q1b gehören immer zusammen berichtet. Q1 allein würde den Effekt stark überzeichnen.
