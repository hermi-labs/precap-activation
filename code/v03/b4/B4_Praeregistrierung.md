# B4 – Präregistrierung: Ist es die weiche Form oder die engere Grenze? (2×2, privat)

Stand: 30.10.2026, nachts, vor dem Start. PreCap, die Tiefenfamilie und Lagma sind Aarons Ideen. Das Design habe ich mit Claude entworfen, mit Aarons Freigabe für die Nacht („freie Hand zu testen und zu entwickeln, exakt begründet, nichts veröffentlichen oder pushen“).

## Ausgangslage

- **B2 (Erfolg):** LB (weiche Lagma-Hülle) war bei 4 Bit in 4 von 4 Zellen besser als LBh (harte Grenze), um 0,9–1,8 pp, fp32 nicht schlechter.
- **Offener Vorbehalt aus B2:** Bei gleicher Kritikalität liegt die weiche Grenze etwa 3,5 % enger (d100 2,2165 gegen 2,2992; d200 2,4543 gegen 2,5369). Der Vorteil kann also von der Form oder vom engeren Bereich kommen.
- **Rechnung p11** (`math/p11_gleiches_M.py`):
  - Bei Mittel 0, Varianz 1 und gleichem χ1 sind Weichheit und Grenze gekoppelt. Eine Familie „weich mit gleicher Grenze und gleicher Kritikalität“ gibt es mit diesen Formen nicht.
  - Hält man M gleich und löst β nur für Varianz 1, weicht χ1 ab:
    - LBhM (hart, enge Grenze): χ1 = 1,02534 (d100) bzw. 1,01276 (d200), also etwa 1 + 2,5/L
    - LBM (weich, weite Grenze): χ1 = 1,01572 bzw. 1,00784, also etwa 1 + 1,6/L
  - Beides liegt im Bereich der Tiefenfamilie (kL1 bis knapp über kL2).

## Frage

Ist die weiche Lagma-Hülle bei **gleicher Grenze** bei 4 Bit besser als die harte, oder erklärt der engere Bereich den Vorteil aus B2?

## Eine Änderung gegenüber B2: Grenze und Kritikalität entkoppelt (2×2)

| | enge Grenze (M wie LB) | weite Grenze (M wie LBh) |
|---|---|---|
| **weich** (Lagma-Bremse φ_ρ, ρ = 0,8) | LB (B2, kritisch, χ1 = 1 + 2/L) | **LBM** (neu, χ1 ≈ 1 + 1,6/L) |
| **hart** (clip) | **LBhM** (neu, χ1 ≈ 1 + 2,5/L) | LBh (B2, kritisch, χ1 = 1 + 2/L) |

Konstanten der neuen Formen (p11, Varianz 1 exakt):

| | d100 | d200 |
|---|---|---|
| LBhM | β = 1,028768, M = 2,216509 | β = 1,014266, M = 2,454294 |
| LBM | β = 1,028322, M = 2,299246 | β = 1,014988, M = 2,536873 |

**Konstanten wie B2:** Netz, Optimierer, Budget (1720 Schritte), LR (die von kL2 aus A, wie B1/B2), Kalibrierung, Quantisierung (4 und 3 Bit, symmetrisch und asymmetrisch), MNIST und Fashion, d100 und d200.

**Seeds:** neu 42096–42107 (12 je Zelle), gepaart. 4 · 4 · 12 = 192 Läufe.

**Kontrolle:** LB und LBh, MNIST d100, Seed 42072, bitgleich zu B2. Zusammen 194 Läufe, etwa 2,5 h. Ein Ausfall zählt als 90 %.

## Primärtest

**LB − LBhM bei w8a4_best**, also die Form bei gleicher (enger) Grenze. Je Zelle Wilcoxon zweiseitig exakt, Holm über die 4 Zellen. Das Gesamturteil folgt derselben Regel wie in B1/B2.

**H1:** Die mediane fp32-Differenz LB − LBhM ist in jeder Zelle höchstens +0,5 pp.

**Erfolg (die Form zählt):** Primär „besser als LBhM“ und H1 hält.

## Sekundär (je Familie Holm über 4 Zellen)

| Nr. | Vergleich | Frage |
|---|---|---|
| S1 | LBM − LBh, w8a4_best | Form bei weiter Grenze |
| S2 | LB − LBM, w8a4_best | Grenze bei weicher Form |
| S3 | LBhM − LBh, w8a4_best | Grenze bei harter Form |
| S4 | LB − LBhM, fp32 | Form bei enger Grenze in fp32 (zugleich H1) |
| S5 | LBM − LBh, fp32 | Form bei weiter Grenze in fp32 |
| S6 | LB − LBhM, w8a4a_best | Form bei asymmetrischer Quantisierung |
| S7 | LB − LBh, w8a4_best | Wiederholung des B2-Primärtests auf neuen Seeds |

**Beschreibend:** Zerlegung je Seed und Zelle für w8a4_best und fp32:
- Form-Effekt = ½[(LB − LBhM) + (LBM − LBh)]
- Grenz-Effekt = ½[(LB − LBM) + (LBhM − LBh)]
- Wechselwirkung = (LB − LBhM) − (LBM − LBh)

Dazu Mediane aller Metriken, Sättigung und Bereich.

## Deutungsregeln (vorab)

| Ergebnis | Deutung |
|---|---|
| Erfolg und S1 gleich gerichtet | Die weiche Form ist bei gleicher Grenze besser. Die Lagma-Form hat einen eigenen Beitrag, nicht nur über den Bereich |
| Primär „kein Effekt“, S2/S3 „besser“ (enge Grenze besser) | Der Vorteil aus B2 kommt vom engeren Bereich. Weich ist nur besser, weil es bei gleicher Kritikalität enger sein darf. Das ist ein indirekter, aber echter Vorteil |
| Primär „besser“, S1 „kein Effekt“ | Die Form zählt nur bei enger Grenze, also bei mehr Sättigung. Plausibel, weil dort mehr Werte im gesättigten Bereich liegen |
| Primär „schlechter“ | Bei gleicher Grenze ist hart besser. Der B2-Vorteil kam ganz vom Bereich |
| S7 „besser“ | B2 wiederholt sich auf neuen Seeds |
| Kontrolle nicht bitgleich | Abweichung wird gesucht und berichtet, die Tests gelten |

**Vorbehalt in jeder Deutung:** In den Vergleichen bei gleichem M unterscheidet sich χ1 (LB 1 + 2/L gegen LBhM ≈ 1 + 2,5/L; LBM ≈ 1 + 1,6/L gegen LBh 1 + 2/L). In beiden Form-Vergleichen hat die weiche Form das kleinere χ1. Ein Form-Effekt ist also nicht völlig von χ1 getrennt. Aus v0.2 (kL1 gegen kL2) ist bekannt, dass ein solcher Unterschied in χ1 bei 4 Bit wenig ausmacht. Das wird im Bericht berücksichtigt.

## Vorab-Erwartung

| Größe | Erwartung |
|---|---|
| Primär „besser als LBhM“ | 0,35 |
| Primär „kein Effekt“ oder „gemischt“ | 0,5 |
| Primär „schlechter“ | 0,15 |
| H1 hält | 0,75 |
| S2 oder S3 „besser“ (Grenze zählt) | 0,4 |
| S7 „besser als LBh“ (B2 wiederholt) | 0,7 |
| Kontrolle bitgleich | 0,95 |

## Neuheit und Literatur

Siehe `docs/literatur/Recherche_2026-10-30_BPreCap.md`. Eine Studie, die weich gegen hart bei gleicher Grenze und bei gleicher Kritikalität getrennt misst, habe ich nicht gefunden. Kein Neuheitsanspruch, es geht um eine interne Klärung.

## Grenzen

- Wie B2: einfache Daten, MLP ohne Residuen, LR nicht getunt, nur Quantisierung nach dem Training.
- χ1 ist in den neuen Armen nicht exakt kritisch (siehe Vorbehalt).
- Nur eine Weichheitsstufe (ρ = 0,8).

## Freeze

Eingefroren werden `b4_run.py`, `b4_analyze.py`, `test_b4_mechanik.py`, `test_b4_analyse.py`, diese Datei, `lr_from_a.csv`, `lr_from_p3a3.csv`, `p7_hard_bound.json`, `p11_gleiches_M.json` und `src/precap/*.py`.

Vorab gemachte Tests (30.10., nachts):

- **Selbsttest:** n_confirm 194. LBhM und LBM mit Varianz 1,000 und χ1 genau wie p11.
- **Mechanik** (`test_b4_mechanik.py`): MECHANIK OK.
  - Varianz 1 und χ1 wie p11 auf 2e-4. Die Grenzen sind exakt gleich den Partnern, die Formen gleich clip bzw. Lagma-Bremse, MLX gleich numpy.
  - LB und LBh gleich B2, `execute` bitgleich zu B2.
  - Lernraten wie B2, Seeds neu, 194 Läufe.
- **Auswertung an künstlichen Daten** (`test_b4_analyse.py`): AUSWERTUNG OK.
  - „Form zählt“ wird erkannt, mit negativem Form-Effekt in allen Zellen.
  - „Weich teuer“ ergibt H1 verletzt.
  - **„Nur die Grenze zählt“ ergibt Primär „kein Effekt“**, einen Grenz-Effekt von etwa −1,5 pp und S2 „besser“.
  - „Null“ ergibt 6 von 6 „kein Effekt“.
  - Eine falsche Kontrolle wird erkannt, ein Ausfall gezählt, der relative Pfad findet B2.
