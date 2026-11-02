# B13 – Präregistrierung: Weich gegen hart im TAT-Rezept (CIFAR-10, d100, privat)

Stand: 31.10.2026, abends, vor dem Start. PreCap, die Tiefenfamilie und Lagma sind Aarons Ideen. BPreCap (= kL2B) ist ein Arbeitsname. Das Design habe ich mit Claude entworfen. Aaron hat B12 und B13 am 31.10. abends für die Nacht freigegeben. Nichts veröffentlichen.

## Ausgangslage

- **B4:** Bei gleicher Grenze und gleicher Kritikalität ist die weiche Hülle bei 4 Bit besser als ein harter Clip. Gezeigt nur mit **Gauß-Init** und Adam bei konstanter LR, auf MNIST und Fashion.
- Für v0.3 soll diese Nebenaussage (K1) im **TAT-Rezept** und im **Regime von B10** gelten. Das ist bisher nicht geprüft.

## Frage

Ist die weiche BPreCap-Form im TAT-Rezept auf CIFAR-10 (d100) bei fairen 4 Bit besser als dieselbe Form mit hartem Clip an der gleichen Grenze?

## Eine Änderung gegenüber B10: harte statt weicher Hülle

- **kL2Bh_tr:** PreCap-Kern, harter Clip bei M = 3, Konstanten so, dass χ1 = 1 + 2/L (`math/p7_hard_bound.py`, wie B4).
- **LBh_tr:** linearer Kern, harter Clip, χ1 = 1,02 (wie B4).
- Rezept, Regime, Messpunkte, frühes Stoppen und Quantisierung wie B10.
- **Tuning je Form:** LR-Spitze {5e-5, 1e-4, 2e-4, 5e-4} × WD {0; 0,3; 3,0}, 2 Tuning-Seeds, Wahlregel wie B10.
- **Vergleichsformen:** kL2B_tr und LB_tr aus B10 (je 1e-4, WD 0,3), dieselben 12 Seeds 42168–42179, gepaart.

## Läufe

- Tuning 2 × 12 × 2 = 48.
- Bestätigung 2 × 12 = 24.
- **Kontrolle:** kL2B_tr, Seed 42168, mit der B10-Wahl, bitgleich zu B10.
- Zusammen 73 Läufe, etwa 1,7 h. Start nach B12, nur am Netzteil.

## Primärtest

**kL2B_tr − kL2Bh_tr bei w8a4a_best am Punkt des frühen Stoppens.** Wilcoxon zweiseitig exakt, gepaart, α = 0,05.

**Erfolg:** p < 0,05 und Median < 0.

## Sekundär (Holm über 7)

| Nr. | Vergleich |
|---|---|
| S1 | LB_tr − LBh_tr, w8a4a |
| S2 | kL2B_tr − kL2Bh_tr, w8a4 |
| S3 | LB_tr − LBh_tr, w8a4 |
| S4 | kL2B_tr − kL2Bh_tr, fp32 |
| S5 | LB_tr − LBh_tr, fp32 |
| S6 | kL2B_tr − kL2Bh_tr, w8a3 |
| S7 | LB_tr − LBh_tr, w8a3 |

Alle beim frühen Stoppen.

**Beschreibend:** gewählte LR und WD, Bereich, 4-Bit-Preis, Stopp-Punkte.

## Deutungsregeln (vorab)

| Ergebnis | Deutung |
|---|---|
| Erfolg | K1 gilt auch im TAT-Rezept und praxisüblichen Regime |
| Primär „unklar“ | Im Rezept ist kein Vorteil der weichen Form nachweisbar. K1 bleibt in v0.3 auf Gauß-Init beschränkt (B4) |
| Primär „schlechter“ | Im Rezept ist die harte Form bei 4 Bit besser. K1 wird für v0.3 eingeschränkt bzw. gestrichen |
| S4 „schlechter“ | Die weiche Form kostet im Rezept fp32-Güte. Wird berichtet |

## Vorab-Erwartung

| Größe | Erwartung |
|---|---|
| Primär „besser“ | 0,45 |
| Primär „unklar“ | 0,45 |
| Primär „schlechter“ | 0,1 |
| S1 „besser“ | 0,45 |
| S4 „unklar“ (fp32 gleich) | 0,6 |
| Kontrolle bitgleich | 0,95 |

Die Erwartung ist zurückhaltend. In B4 war der Effekt klein (0,5–1,6 pp) und bei weiter Grenze nicht signifikant.

## Grenzen

- Nur 2 Tuning-Seeds. Nur d100, CIFAR-10, Quantisierung nach dem Training.
- Die weichen Läufe stammen aus B10, ihre Ergebnisse waren beim Entwurf bekannt.
- Ein gelernter harter Clip (PACT) ist nicht dabei. Er braucht lernbare Grenzen und kommt, wenn überhaupt, in einer eigenen Studie.

## Freeze

Eingefroren werden `b13_run.py`, `b13_analyze.py`, `test_b13_mechanik.py`, `test_b13_analyse.py`, diese Datei, `hp_from_b10.csv`, `src/precap/*.py` und die CIFAR-10-Dateien.

Vorab gemachte Tests (31.10., abends):

- **Selbsttest:** n_confirm 25, n_tune 48.
- **Auswertung an künstlichen Daten** (`test_b13_analyse.py`): AUSWERTUNG OK.
  - Erfolg und „schlechter“ werden erkannt. Null mit 20 Wiederholungen: 1 Fehlalarm.
  - Eine falsche Kontrolle wird erkannt, am Ende wie an einem Messpunkt.
  - Ein Ausfall ohne Messpunkt zählt als 90 %. Der relative Pfad findet B10.
- **Mechanik** (`test_b13_mechanik.py`): **MECHANIK OK** (31.10., 21:29). Das Startskript führt den Test erneut aus und bricht bei Fehler ab.
- **Rauchtest:** im Startskript.
