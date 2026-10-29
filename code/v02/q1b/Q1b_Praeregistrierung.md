# Q1b – Präregistrierung: faire Kalibrierung bei 4-Bit-Aktivierungen (privat)

Stand: 27.10.2026. Das Dokument wird zusammen mit `q1b_run.py` und `q1b_analyze.py` per SHA-256 eingefroren, bevor der erste Hauptlauf startet. Bis zur Freigabe bleibt es privat.

## Anlass

In Q1 war SPreCap-cs bei w8a4 in allen 4 Zellen und für 15 von 15 Seeds robuster als SELU. Die Unterschiede lagen zwischen −6 und −19 pp. Die Aktivierungen wurden in Q1 aber mit ihrem Maximum kalibriert. Das benachteiligt unbeschränkte Funktionen wie SELU und ELU, weil einzelne Ausreißer die Stufen grob machen. Q1b prüft deshalb, ob der Vorteil auch dann bleibt, wenn jede Funktion eine gute Kalibrierung bekommt.

## Aufbau

- **Training:** identisch zu Q1. Es gelten dieselben Daten, Lernraten, Seeds (42010–42024) und derselbe Code, also 240 Läufe auf dem M1 Pro mit MLX.
- **Kalibrierung:** auf den ersten 1000 Trainingsbildern, mit fp32-Eingängen jeder Linearschicht, pro Tensor symmetrisch. Gewichte werden pro Ausgangskanal mit dem Maximum quantisiert.
- **Clip-Verfahren für die Aktivierungen:**
  - max
  - 99,9-Perzentil
  - 99,99-Perzentil
  - MSE-optimaler Clip auf dem Gitter 1–100 % des Maximums, getrennt für 4 und 8 Bit

| Stufe | Gewichte | Aktivierungen | Rolle |
|---|---|---|---|
| w8a4_best | int8 | int4, je Lauf das Verfahren mit dem kleinsten Fehler auf den Kalibrierbildern (Kandidaten: max, p99,9, p99,99, MSE; bei Gleichstand gilt die Reihenfolge der Liste) | primär |
| w8a4_mse | int8 | int4, MSE-Clip | sekundär |
| w8a4_p999, w8a4_p9999 | int8 | int4, Perzentil | sekundär |
| w8a8_mse | int8 | int8, MSE-Clip | sekundär |
| w8a8, w8a4, w4a8 | wie Q1 (max) | | Wiederholungskontrolle |

## Messgröße

Δ = Testfehler quantisiert − Testfehler fp32, je Lauf, in Prozentpunkten.

## Primärtest

- **Vergleich:** spreccap_cs gegen selu auf w8a4_best, gepaart über den Seed, zweiseitiger Wilcoxon-Test (zero_method='wilcox', method='auto'; alle Paardifferenzen 0 → p = 1).
- **Umfang:** 4 Zellen, also 4 Tests, Holm, α = 0,05.
- **Urteil je Test:** wie Q1 ("robuster", "anfälliger" oder "unklar").
- **Gesamturteil:**
  - "Vorteil bleibt": mindestens 2 von 4 Zellen "robuster" und keine "anfälliger"
  - "Vorteil verschwindet": keine Zelle "robuster"
  - "Vorteil kehrt sich um": mindestens 1 Zelle "anfälliger" und keine "robuster"
  - sonst "gemischt"

## Sekundär

- e03 gegen selu, cs gegen elu und e03 gegen elu auf w8a4_best
- alle Paare auf w8a4_mse, w8a4_p999, w8a4_p9999 und w8a8_mse, Holm je Stufe
- Wie oft wird welches Verfahren je Funktion gewählt?
- **Kontrolle Wiederholbarkeit:** w8a8, w8a4, w4a8 und fp32 gegen Q1, je Lauf. Berichtet wird der Anteil identischer Testfehler und die maximale Abweichung. Sie dient nur der Beschreibung.
- **Spannweite:** Abstand zwischen bester Kalibrierung und max je Funktion, also wie viel die faire Kalibrierung SELU/ELU hilft.

## Erwartung vor dem Lauf

- SELU und ELU profitieren stark von der Clip-Kalibrierung, cs kaum, weil sein Wertebereich fest begrenzt ist.
- Geschätzte Chance, dass der Vorteil in mindestens 2 Zellen bleibt: etwa 0,5.

## Ausfälle

Wie Q1. Ein nicht-endlicher Loss bedeutet, dass das Paar entfällt. Es gibt keine Wiederholung.

## Was vorher bekannt ist

- Die Q1-Ergebnisse mit max-Kalibrierung sind bekannt.
- Die neuen Kalibrierverfahren wurden auf echten Daten noch nicht ausgewertet. Der Rauchtest läuft auf synthetischen Daten und auf MNIST mit 40 Schritten. Er prüft nur, ob der Code läuft.
