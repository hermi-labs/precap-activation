# B15 – Entwurf (überholt, siehe B15_Praeregistrierung.md)

Stand: 01.11.2026, 02:05, Entwurf mit Claude. **Start nur nach Aarons Freigabe** (GPU etwa 4 h, über der 1-h-Grenze). Nichts veröffentlichen.

## Warum

- **B12:** Bei d100 im TAT-Rezept ist SELU nur 0,95 pp hinter BPreCap in fp32 und bei 4 Bit gleichauf. Sein 4-Bit-Preis ist ebenso klein (+2,5 pp).
- Die Tiefenkalibrierung (χ1 = 1 + 2/L) ist der Teil von BPreCap, der bei größerer Tiefe zählen sollte. SELU hat festes χ1 = 1,07. Bei d200 wächst der Gradient bei der Init damit etwa um 1,0716^99,5 ≈ 970-fach (wie in v0.2). In v0.2 ist SELU bei d200 mit Gauß-Init und kurzem Budget gescheitert.
- Offen ist, ob das auch mit dem Rezept (orthogonale Init, Normierung je Beispiel) und dem B10-Regime gilt. Die Antwort entscheidet, wie v0.3 BPreCap gegenüber SELU einordnet.

## Frage

Ist BPreCap bei d200 im TAT-Rezept und im B10-Regime auf CIFAR-10 in fp32 und bei 4 Bit besser als SELU?

## Design (eine Änderung gegenüber B12: Tiefe 200, dafür nur SELU)

- **Formen:** kL2B_tr (Konstanten für d200, χ1 = 1,01), SELU_tr (Gain 1,0).
- **Regime wie B10:** AdamW, Kosinus auf 0 über 6880 Schritte, frühes Stoppen nach Val, Messpunkte alle 860.
- **Tuning für beide Formen neu,** weil es für BPreCap bei d200 im B10-Regime noch keine Wahl gibt. LR {2e-5, 5e-5, 1e-4, 2e-4} × WD {0; 0,3; 3,0}, 2 Tuning-Seeds.
  - Das Raster liegt eine Stufe tiefer als bei d100, weil B5 und B8 bei d200 kleinere LR wählten.
- **Bestätigung:** 12 neue Seeds (42192–42203), gepaart. **Kontrolle:** kL2B_tr d100, Seed 42168, B10-Wahl, bitgleich zu B10.
- **Läufe:** Tuning 2 × 12 × 2 = 48, Bestätigung 24 + 1.
- **Laufzeit:** etwa 4 h (ein d200-Lauf über 6880 Schritte etwa 7 min, 2 Worker).

## Primärtest

kL2B_tr − SELU_tr bei w8a4a_best am Stopp-Punkt, d200. Wilcoxon zweiseitig exakt, α = 0,05.

## Sekundär (Holm über 3)

| Nr. | Vergleich |
|---|---|
| S1 | fp32 |
| S2 | w8a4 |
| S3 | w8a3 |

## Deutungsregeln (Entwurf)

| Ergebnis | Deutung für v0.3 |
|---|---|
| Primär und S1 „besser“ | Bei größerer Tiefe trägt die Tiefenkalibrierung. BPreCap ist SELU dort überlegen |
| S1 „besser“, Primär „unklar“ | BPreCap trainiert tiefer besser, der 4-Bit-Preis ist gleich |
| beide „unklar“ | Im Rezept hält SELU bis d200 mit. v0.3 nennt das so, BPreCaps Vorteil gegenüber SELU ist dann nur klein |
| „schlechter“ | SELU ist besser. Die Kernaussage von v0.3 muss eingeschränkt werden |

## Vorab-Erwartung (Entwurf)

| Größe | Erwartung |
|---|---|
| S1 „besser“ | 0,6 |
| Primär „besser“ | 0,5 |
| SELU divergiert in mindestens einer Tuning-Kombination | 0,8 |

## Vor dem Start noch zu tun

- Code aus `b12_run.py`, Tiefe 200, Seeds, Raster.
- Mechanik-Test, insbesondere bitgleiche Kontrolle und Konstanten für d200.
- Auswertungsskript an künstlichen Daten testen.
- Freeze, dann Rauchtest.
