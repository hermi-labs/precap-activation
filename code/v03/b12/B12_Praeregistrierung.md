# B12 – Präregistrierung: Basislinien ReLU, GELU, SELU im gleichen Rezept und Regime (CIFAR-10, d100, privat)

Stand: 31.10.2026, abends, vor dem Start. PreCap, die Tiefenfamilie und Lagma sind Aarons Ideen. BPreCap (= kL2B) ist ein Arbeitsname. Das Design habe ich mit Claude entworfen. Aaron hat B12 und B13 am 31.10. abends für die Nacht freigegeben („wenn jetzt alles bprecap test durchlaufen bis morgen früh“, „übersürze bitte nichts“). Nichts veröffentlichen.

## Ausgangslage und Zweck

- Für v0.3 fehlt der Vergleich mit gewöhnlichen Aktivierungen im selben Aufbau. Das ist die erste Frage jedes Lesers.
- Die Theorie (`math/p12_tiefe_kollaps.py`) sagt für feste Formen bei d100 einen Kollaps (ReLU) oder Chaos (GELU, SELU) der Eingabe-Ähnlichkeit voraus. Die orthogonale Init und die Normierung je Beispiel mildern das nur teilweise.
- B12 misst, wie ReLU, GELU und SELU **im selben Rezept** (orthogonale Init × Gain, Normierung je Beispiel) und **im selben Regime wie B10** abschneiden, bei fp32 und bei 4 Bit.

## Frage

Ist BPreCap bei d100 auf CIFAR-10 bei fairen 4 Bit besser als ReLU, GELU und SELU, wenn alle im gleichen Rezept und Regime trainieren und für jede Form LR und WD getunt sind?

## Eine Änderung gegenüber B10: die Formen

- **Neue Formen** im TAT-Rezept:
  - **ReLU:** Gain √2, also He-Skalierung.
  - **GELU:** exakt mit erf, Gain 1,5335.
  - **SELU:** Gain 1,0000.
  - Die Gains folgen der Regel aus 3a-3: g = 1/√E[f(Z)²]. Effektives χ1 = g²·E[f′²]: ReLU 1,00, GELU 1,07, SELU 1,07.
  - Biases bei 0, Ausgabeschicht wie bei allen Rezept-Formen.
- **Regime wie B10:** AdamW, Kosinus auf 0 über 6880 Schritte, frühes Stoppen nach Val, Messpunkte alle 860 Schritte.
- **Tuning je Form:** LR-Spitze {5e-5, 1e-4, 2e-4, 5e-4} × WD {0; 0,3; 3,0}, 2 Tuning-Seeds, Wahlregel wie B10.
  - Das LR-Raster ist eine Stufe höher als in B10 (plus 5e-4), weil ReLU-Netze oft höhere LR vertragen. BPreCaps Wahl aus B10 (1e-4) liegt damit im Inneren.
- **Vergleichsform:** kL2B_tr aus B10 (1e-4, WD 0,3), dieselben 12 Seeds 42168–42179, gepaart.
- **Konstanten wie B10:** Netz (d100, Breite 256), Batch, Daten, Kalibrierung, Quantisierung, Cache-Grenze.

## Läufe

- Tuning 3 × 12 × 2 = 72.
- Bestätigung 3 × 12 = 36.
- **Kontrolle:** kL2B_tr, Seed 42168, mit der B10-Wahl. Alle Kennzahlen an allen Messpunkten müssen bitgleich zu B10 sein.
- Zusammen 109 Läufe, etwa 2,5 h. Start nach B10b, nur am Netzteil.

## Primärtests (Holm über 3, α = 0,05)

| Nr. | Vergleich |
|---|---|
| P1 | kL2B_tr − ReLU_tr, w8a4a_best, frühes Stoppen |
| P2 | kL2B_tr − GELU_tr, w8a4a_best, frühes Stoppen |
| P3 | kL2B_tr − SELU_tr, w8a4a_best, frühes Stoppen |

Wilcoxon zweiseitig exakt, gepaart.

**Erfolg:** alle drei „besser“.

## Sekundär (Holm über 6)

- S1–S3: dieselben drei Vergleiche in fp32.
- S4–S6: dieselben drei Vergleiche bei w8a4_best (symmetrisch).

**Beschreibend:**
- Gewählte LR und WD, Lage am Rand.
- Stopp-Punkte, Bereich am Stopp-Punkt, 4-Bit-Preis.
- Ob eine Form in fp32 deutlich hinter BPreCap liegt (Hinweis auf schlechtere Trainierbarkeit).

## Deutungsregeln (vorab)

| Ergebnis | Deutung |
|---|---|
| Erfolg | Gewöhnliche Aktivierungen sind im selben Aufbau bei 4 Bit schlechter als BPreCap |
| fp32 „besser“ für BPreCap | Gewöhnliche Formen trainieren in diesem tiefen Netz schlechter, wie von der Theorie erwartet |
| fp32 „unklar“, w4a „besser“ | Sie trainieren gleich gut, verlieren aber bei 4 Bit, wie TAT. Der Vorteil liegt in der Grenze |
| eine Form bei w4a „unklar“ oder „schlechter“ | Wird so berichtet. Die Aussage in v0.3 gilt dann nur gegen die anderen Formen |
| Kontrolle nicht bitgleich | Abweichung wird gesucht und berichtet |

## Vorab-Erwartung

| Größe | Erwartung |
|---|---|
| P1 ReLU „besser“ | 0,8 |
| P2 GELU „besser“ | 0,8 |
| P3 SELU „besser“ | 0,7 |
| alle drei „besser“ (Erfolg) | 0,55 |
| fp32: mindestens eine Form „besser“ für BPreCap | 0,6 |
| mindestens eine LR-Wahl am oberen Rand (5e-4) | 0,35 |
| Kontrolle bitgleich | 0,95 |

## Grenzen

- Die gewöhnlichen Formen bekommen unser Rezept (orthogonale Init, Normierung je Beispiel). Mit Gauß-Init wären sie vermutlich schlechter. Gemessen wird also ihre beste uns bekannte Variante ohne Residuen und ohne Normierungsschichten.
- Nur 2 Tuning-Seeds. Nur d100, CIFAR-10, Quantisierung nach dem Training.
- Die BPreCap-Läufe stammen aus B10, ihre Ergebnisse waren beim Entwurf bekannt.

## Freeze

Eingefroren werden `b12_run.py`, `b12_analyze.py`, `test_b12_mechanik.py`, `test_b12_analyse.py`, diese Datei, `hp_from_b10.csv`, `src/precap/*.py` und die CIFAR-10-Dateien.

Vorab gemachte Tests (31.10., abends):

- **Selbsttest:** n_confirm 37, n_tune 72. Neue Formen: MLX gegen NumPy unter 1e-5.
- **Auswertung an künstlichen Daten** (`test_b12_analyse.py`): AUSWERTUNG OK.
  - Erfolg und „schlechter“ werden erkannt. Null mit 20 Wiederholungen: 0 Fehlalarme.
  - Eine falsche Kontrolle wird erkannt, am Ende wie an einem Messpunkt.
  - Ein Ausfall ohne Messpunkt zählt als 90 %. Der relative Pfad findet B10.
- **Mechanik** (`test_b12_mechanik.py`): **MECHANIK OK** (31.10., 21:29). Das Startskript führt den Test erneut aus und bricht bei Fehler ab.
- **Rauchtest:** im Startskript.
