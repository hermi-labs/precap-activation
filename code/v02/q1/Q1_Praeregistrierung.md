# Q1 – Präregistrierung: Quantisierung nach dem Training (privat)

Stand: 27.10.2026. Das Dokument wird zusammen mit `q1_run.py` und `q1_analyze.py` per SHA-256 eingefroren, bevor der erste Hauptlauf startet. Bis zur Freigabe bleibt es privat.

Korrektur vor dem Start (27.10.2026, vor dem ersten Hauptlauf): In `q_fp8` wurden Exponent und Stufe von `mx.log2`/`mx.power` auf exakte Bit-Operationen umgestellt, weil Metal `pow(2, negativ)` nicht exakt liefert (Selbsttest `q_fp8_grid_ok` war auf der GPU falsch). Die Definition von E4M3 bleibt unverändert.

## Frage

Verliert ein tiefes MLP ohne Normierung mit SPreCap weniger Genauigkeit als mit SELU, wenn man es nach dem Training quantisiert? SPreCap ist beidseitig beschränkt. Deshalb sollte der Wertebereich der Aktivierungen kleiner sein, und die Rundungsstufen werden feiner.

## Aufbau

- **Training wie 3a-3:**
  - MNIST und FashionMNIST, Tiefe 50 und 100, Breite 256
  - Batch 128, 1720 Schritte, Adam (β = 0.9/0.999, ε = 1e-8, mit Bias-Korrektur)
  - varianzerhaltende Init, Validierung = letzte 5000 Trainingsbilder
- **Lernraten:** werden unverändert aus der 3a-3-Auswahl übernommen (`lr_from_p3a3.csv`). Es gibt kein neues Tuning.
- **Funktionen:** spreccap_cs, spreccap_e03, selu, elu
- **Seeds:** die 15 Bestätigungs-Seeds aus 3a-3 (42010 bis 42024), also 240 Läufe
- **Rechner:** MLX auf dem M1 Pro. PreCap läuft mit dem Metal-Kernel aus dem Paket, SELU und ELU mit den nativen MLX-Funktionen. Alle Funktionen laufen auf demselben Gerät.
- **Kalibrierung:** auf den ersten 1000 Trainingsbildern, nie auf Test oder Validierung.
- **Simulierte Quantisierung:** Gerechnet wird in fp32 mit gerundeten Werten.
  - Gewichte: pro Ausgangskanal symmetrisch
  - Eingänge jeder Linearschicht, inklusive der ersten: pro Tensor symmetrisch mit dem kalibrierten Maximum

| Stufe | Gewichte | Aktivierungen | Rolle |
|---|---|---|---|
| w8a8 | int8 | int8 (Max) | primär |
| w8a4 | int8 | int4 (Max) | primär |
| w4a8 | int4 | int8 (Max) | sekundär |
| fp8 | FP8 E4M3 | FP8 E4M3 (Max) | sekundär |
| w8a8_p9999 | int8 | int8 (99,99-Perzentil) | sekundär |

## Messgröße

Δ = Testfehler quantisiert − Testfehler fp32, je Lauf, in Prozentpunkten.

## Primärtest

- **Vergleich:** spreccap_cs gegen selu, gepaart über den Seed, zweiseitiger Wilcoxon-Vorzeichen-Rang-Test auf Δ (scipy, zero_method='wilcox', method='auto'; alle Paardifferenzen 0 → p = 1).
- **Zellen:** 4 (Datensatz × Tiefe) mal 2 Stufen (w8a8, w8a4), also 8 Tests, Holm-korrigiert, α = 0,05.
- **Urteil je Test:**
  - "SPreCap robuster": Holm-p < 0,05 und Median der Paardifferenz Δcs − Δselu < 0
  - "SPreCap anfälliger": Holm-p < 0,05 und Median > 0
  - sonst "unklar"
- **Gesamturteil:**
  - "bestätigt": mindestens 2 der 8 Tests "robuster" und keiner "anfälliger"
  - "widerlegt": mindestens 1 Test "anfälliger" und keiner "robuster"
  - sonst "offen"

## Sekundär (nur beschreibend oder Holm innerhalb der Familie)

- e03 gegen selu, cs gegen elu und e03 gegen elu, auf w8a8 und w8a4
- alle Paare auf w4a8, fp8 und w8a8_p9999
- Zusammenhang zwischen Wertebereich (`act_range_max_median`) und Δ
- **Kontrolle:** fp32-Testfehler in MLX gegen die 3a-3-Mediane. Eine Abweichung über 1 pp im Median wird berichtet, ändert aber nicht die Auswertung.
- **Tempo:** ms pro Trainingsschritt, Kernel gegen Referenz gegen SELU (getrennter Probe-Lauf)

## Abbruch und Ausfälle

Nicht-endlicher Loss ergibt `status` ≠ ok. Δ wird dann als fehlend gewertet, und das Paar entfällt im Test. Die Zahl der Ausfälle wird je Funktion berichtet. Es gibt keine Wiederholung mit anderen Seeds.

## Was vorher bekannt ist

Die Lernraten stammen aus 3a-3. Quantisierte Ergebnisse gibt es bisher nicht. Sie wurden auch nicht in einem Testlauf angeschaut, abgesehen vom Rauchtest auf synthetischen Daten mit 40 Schritten.
