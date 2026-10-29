# Q1 – Auswertung: Quantisierung nach dem Training

*Veröffentlichte Fassung vom 29.10.2026: Der Bericht ist inhaltlich unverändert. Entfernt wurden nur der Vermerk „privat“ und Querverweise auf noch unveröffentlichte Studien. Präregistrierung und Skripte sind die eingefrorenen Originale.*

Stand: 27.10.2026. Q1 wird immer zusammen mit Q1b berichtet.

## Kurzfassung

- **Gesamturteil nach Präregistrierung: "bestätigt"**
  - 4 von 8 Primärtests fallen "SPreCap robuster" aus, keiner "anfälliger".
- **w8a4 (int8-Gewichte, int4-Aktivierungen):**
  - SPreCap-cs verliert in allen 4 Zellen viel weniger Genauigkeit als SELU, und das bei 15 von 15 Seeds (Holm p = 0,0005).
  - Der Vorteil beträgt 6 bis 19 Prozentpunkte.
- **w8a8:** Es gibt keinen Unterschied. Alle Funktionen verlieren praktisch nichts (|Δ| ≤ 0,16 pp).
- **Wichtigster Vorbehalt:** Kalibriert wurde mit dem Maximum. Das benachteiligt SELU und ELU, weil sie unbeschränkt sind. Q1b prüft das deshalb mit fairer Kalibrierung (Perzentil, MSE-Clip, beste Wahl je Lauf). Erst danach ist das Ergebnis belastbar.
- **Zuversicht:**
  - Das Ergebnis "unter max-Kalibrierung robuster" ist sicher: 0,95.
  - Der Vorteil bleibt auch bei fairer Kalibrierung: etwa 0,5.

## Ablauf und Prüfung

- **Umfang:** 240 von 240 Läufen OK, keine Ausfälle. MLX 0.32.4 auf dem M1 Pro, Metal-Kernel.
- **Eingefroren vor dem Start:**
  - Die SHA-256-Werte von `q1_run.py`, `q1_analyze.py`, Präregistrierung, LR-Tabelle und Paket waren nach dem Lauf identisch.
  - `meta/` und `logs/` auf dem Mac wurden um 07:52 durch einen Dateiabgleich geleert. Die Ergebnisse in `workers/` blieben vollständig erhalten. Eine Sicherung liegt in `~/PyCharmMiscProject/q1_sicherung/`:
    - w0 sha256 53f55d25…
    - w1 sha256 59d9c295…
- **Korrektur vor dem Start:** Die fp8-Rundung wurde auf exakte Bit-Rechnung umgestellt (siehe Präregistrierung).
- **Abweichung vom Plan:** Zusätzlich zum synthetischen Rauchtest lief ein Rauchtest auf MNIST mit 40 Schritten und 4 Läufen. Er diente nur der Funktionsprüfung und hatte keinen Einfluss auf Einstellungen.

## Primärtest: Δcs − Δselu (Median der Paardifferenz, pp)

| Zelle | w8a8 | Siege/Niederl. | Holm p | w8a4 | Siege/Niederl. | Holm p |
|---|---|---|---|---|---|---|
| MNIST 50 | +0,00 | 6/7 | 1,0 | **−6,01** | 15/0 | 0,0005 |
| Fashion 50 | −0,06 | 10/4 | 0,41 | **−7,01** | 15/0 | 0,0005 |
| MNIST 100 | +0,01 | 6/8 | 1,0 | **−19,27** | 15/0 | 0,0005 |
| Fashion 100 | +0,01 | 7/8 | 1,0 | **−16,22** | 15/0 | 0,0005 |

## Genauigkeitsverlust je Funktion (Median Δ, pp)

| Zelle | Funktion | fp32-Fehler | max. Wertebereich | w8a8 | w8a4 | w4a8 | fp8 |
|---|---|---|---|---|---|---|---|
| MNIST 50 | cs | 4,60 | 3,35 | −0,02 | 3,20 | 3,42 | 0,20 |
| | e03 | 4,00 | 5,82 | 0,00 | 9,73 | 2,72 | 0,16 |
| | SELU | 4,29 | 7,60 | −0,01 | 9,28 | 1,28 | 0,07 |
| | ELU | 4,10 | 8,10 | −0,01 | 10,97 | 1,39 | 0,07 |
| Fashion 50 | cs | 14,81 | 3,37 | 0,00 | 0,99 | 1,13 | 0,05 |
| | e03 | 14,19 | 5,81 | −0,01 | 9,25 | 2,84 | 0,28 |
| | SELU | 14,98 | 7,67 | 0,04 | 7,84 | 1,22 | 0,23 |
| | ELU | 14,33 | 6,40 | −0,02 | 31,14 | 4,81 | 0,49 |
| MNIST 100 | cs | 6,21 | 3,35 | 0,01 | 4,36 | 4,47 | 0,29 |
| | e03 | 5,41 | 5,87 | 0,05 | 8,78 | 2,75 | 0,19 |
| | SELU | 6,46 | 6,18 | 0,00 | 23,13 | 5,55 | 0,35 |
| | ELU | 5,92 | 6,95 | 0,05 | 48,16 | 7,91 | 0,30 |
| Fashion 100 | cs | 17,25 | 3,34 | −0,05 | 4,69 | 6,39 | 0,27 |
| | e03 | 16,43 | 5,88 | 0,02 | 8,58 | 3,52 | 0,27 |
| | SELU | 16,64 | 6,26 | −0,04 | 20,70 | 6,42 | 0,35 |
| | ELU | 16,64 | 5,62 | 0,16 | 69,74 | 36,09 | 0,95 |

Der Wertebereich ist das Maximum des Betrags am Eingang jeder Linearschicht, als Median über die Schichten. cs ist durch die tanh-Grenze fest auf etwa 3,35 begrenzt.

## Sekundär

- **e03 gegen SELU bei w8a4:**
  - Bei Tiefe 100 ist e03 robuster: MNIST −14,85 pp, Fashion −12,52 pp, je 15/0.
  - Bei Tiefe 50 ist der Unterschied unklar (+0,75 und +1,0 pp).
  - e03 hat einen größeren Wertebereich (5,8) als cs, weil die Grenze bei e03 weiter außen liegt.
- **cs gegen ELU:** Bei w8a4 ist cs in allen 4 Zellen robuster (−7 bis −65 pp, 15/0). ELU bricht bei int4 stark ein.
- **w4a8 (4-Bit-Gewichte):** Das Bild ist gemischt.
  - cs gegen SELU ist bei MNIST 50 schlechter (+2,30 pp, 0/15, Holm 0,001), in den anderen Zellen unklar.
  - Die Begrenzung hilft also bei den Aktivierungen, nicht bei den Gewichten.
- **fp8 und w8a8 mit 99,99-Perzentil:** Es gibt keine Unterschiede zu SELU. Nur gegen ELU ist cs bei Fashion etwas robuster (fp8, −0,3 bis −0,6 pp).
- **Zusammenhang Wertebereich und Verlust (Spearman):** bei w8a4 ρ = 0,51, bei w8a8 ρ ≈ 0. Das passt zum Mechanismus: kleinerer Bereich, feinere 4-Bit-Stufen.
- **Kontrolle fp32 gegen 3a-3 (Median):** Alle Abweichungen liegen innerhalb von 0,65 pp. Die Grenze von 1 pp wurde nicht überschritten, MLX reproduziert 3a-3 also gut.
- **Tempo (ms pro Trainingsschritt, Probe-Lauf):**

  | Tiefe | Kernel | Referenz | SELU |
  |---|---|---|---|
  | 50 | 5,43 | 7,28 | 5,56 |
  | 100 | 9,79 | 13,55 | 10,49 |

  Der Kernel ist im Training etwa 1,35-mal schneller als die Referenz und gleichauf mit SELU.

## Einordnung

- **Was belegt ist:** Mit einfacher max-Kalibrierung macht die beidseitige Begrenzung tiefe MLPs ohne Normierung bei 4-Bit-Aktivierungen deutlich robuster. Bei 8 Bit bringt sie nichts, weil dann alle praktisch verlustfrei sind.
- **Was offen ist:** Gängige Verfahren (Perzentil- oder MSE-Clip) beschneiden die Ausreißer von SELU/ELU. Ob der Vorteil dann bleibt, prüft Q1b. Ebenso offen ist, ob das Ergebnis auf CNNs und Transformer übertragbar ist und ob echte int4-Hardware dasselbe zeigt. Hier wurde die Quantisierung nur simuliert.
- **Nicht übertreiben:** Die Aussage gilt für simulierte Post-Training-Quantisierung, pro Tensor, max-kalibriert, in MLPs auf MNIST/Fashion.
