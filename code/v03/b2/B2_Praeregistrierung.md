# B2 – Präregistrierung: weiche Lagma-Hülle gegen harte Grenze (privat)

Stand: 30.10.2026, vormittags, vor dem Start. PreCap, die Tiefenfamilie und Lagma sind Aarons Ideen. Aaron will die neue Form als Variante von PreCap führen. „BPreCap“ ist dafür ein Arbeitsname (Vorschlag mit Claude, noch nicht von Aaron bestätigt). Das Design habe ich mit Claude entworfen, nach Aarons Auftrag für heute: „schauen, ob die neue Variante eine Chance hat, ob Lagma eine Chance hat“. Aaron hat Studien für heute freigegeben (nichts veröffentlichen, nichts pushen).

## Ausgangslage

- **B1 (30.10.):** Eine kritisch eingestellte Form mit fester, schmaler Grenze trainiert MLPs mit 100–200 Schichten und senkt den 4-Bit-Fehler stark.
  - BPreCap (= kL2B = M · φ_ρ(PreCap/M)) und LB (= M · φ_ρ(βx/M)) waren in fp32 gleich, LB war bei 4 Bit besser.
  - Die Grenze ist in beiden Fällen die **weiche Lagma-Bremse φ_ρ** (ρ = 0,8).
- **Offen:** Trägt die weiche Form φ etwas bei, oder reicht hartes Abschneiden? Hartes Abschneiden bei Kritikalität ist die bekannte kritische hard-tanh (Pennington et al. 2017). Eine weiche Hülle hält den Gradienten im gesättigten Bereich ungleich null, eine harte setzt ihn auf null.
- **Literatur** (`docs/literatur/Recherche_2026-10-30_BPreCap.md`): Eine direkte Studie „weich gegen hart“ für kritisch eingestellte tiefe Netze bei Quantisierung nach dem Training habe ich nicht gefunden. Blumenfeld et al. 2019 sagen eine Tiefengrenze für quantisierte Aktivierungen voraus, die bei 3 Bit im Bereich unserer Tiefen liegt (ξ ≈ 26).

## Frage

Ist eine **weiche Lagma-Hülle** bei gleicher Kritikalität besser als eine **harte Grenze**, gemessen am 4-Bit-Fehler nach dem Training, ohne fp32 zu verschlechtern?

## Eine Änderung: die Hülle (weich oder hart)

| Variante | Formel | Konstanten |
|---|---|---|
| LB | M · φ_ρ(βx/M) | wie B1 (p5): d100 β = 1,0353, M = 2,2165; d200 β = 1,0187, M = 2,4543 |
| **LBh** | clip(βx, −M, M) | neu (p7): d100 β = 1,02258, M = 2,29925; d200 β = 1,01113, M = 2,53687 |
| kL2B = BPreCap | 3 · φ_ρ(f_{λ,α,b}(x)/3) | wie B1 (p4) |
| **kL2Bh** = BPreCap hart | clip(f_{λ,α,b}(x), −3, 3) | neu (p7): d100 λ = 3,21416, α = 1,37936, b = 0,27047; d200 λ = 5,45039, α = 1,22531, b = 0,16740 |

- **Alle Konstanten** sind exakt für Mittel 0, Varianz 1 und χ1 = 1 + 2/L gelöst (`math/p7_hard_bound.py`; LBh analytisch über erf, kL2Bh numerisch, beide am Gitter gegengeprüft).
- **Folge der gleichen Kritikalität:** Die harte Grenze liegt etwas weiter außen (LBh 2,30 statt 2,22). Am Fixpunkt sättigen 2,4 % (LBh) statt 8,7 % (LB, Anteil |βx| > ρM). Das ist gewollt: Verglichen wird bei gleicher Signalausbreitung, nicht bei gleichem M.
- **Konstanten aus B1:** Netz (MLP, Breite 256, ohne Normierung und Residuen), Adam, Batch 128, 1720 Schritte, Init-Gain, Kalibrierbilder, Quantisierung (w8a4_best, w8a4a_best wie B1). Die Lernrate ist für alle Formen die von kL2 aus A (wie B1, nicht neu getunt).
- **Neu in der Messung, ohne Einfluss auf das Training:** 3-Bit-Aktivierungen nach dem Training, symmetrisch (w8a3_best, 7 Stufen) und asymmetrisch (w8a3a_best, 8 Stufen), mit derselben Wahlregel aus {max, p999, p9999, mse3}.

## Zellen, Seeds, Kontrolle

- MNIST und FashionMNIST, Tiefe 100 und 200. **Neue Seeds 42072–42083** (12 je Zelle), gepaart über die Varianten. 4 · 4 · 12 = 192 Läufe.
- **Kontrolle:** LB und kL2B, MNIST d100, Seed 42060 werden neu gerechnet und müssen bitgleich zu B1 sein (fp32, w8a4_best, w8a4a_best, Trainingsverlust). Zusammen 194 Läufe, etwa 2,5 h mit 2 Workern.
- Ein Ausfall (nicht endlicher Verlust) zählt als 90 % Fehler.

## Primärtest

**LB − LBh bei w8a4_best** (Testfehler in pp), je Zelle Wilcoxon zweiseitig exakt, Holm über die 4 Zellen. Gesamturteil wie in Lagma-5/B1: „besser“, wenn mindestens 2 Zellen besser und keine schlechter sind, „schlechter“ entsprechend umgekehrt, sonst „kein Effekt“ oder „gemischt“.

**H1:** Die mediane fp32-Differenz LB − LBh ist in jeder Zelle höchstens +0,5 pp.

**Erfolg (für die weiche Lagma-Hülle):** Primär „besser als LBh“ und H1 hält.

## Sekundär (jede Familie Holm über 4 Zellen, Gesamturteil wie oben)

| Nr. | Vergleich | Frage |
|---|---|---|
| S1 | LB − LBh, fp32 | Trainiert die weiche Hülle besser oder schlechter? |
| S2 | kL2B − kL2Bh, w8a4_best | Gilt der Primärbefund auch mit PreCap-Kern (BPreCap)? |
| S3 | kL2B − kL2Bh, fp32 | dito in fp32 |
| S4 | LB − LBh, w8a4a_best | Hält der Befund bei asymmetrischer Quantisierung? |
| S5 | LB − LBh, w8a3_best | Wird der Unterschied bei 3 Bit größer? |
| S6 | kL2Bh − LBh, fp32 | Trägt der PreCap-Kern bei harter Grenze etwas bei? |
| S7 | kL2B − LB, w8a4_best | Wiederholung B1 S4 auf neuen Seeds |
| S8 | kL2B − LB, fp32 | Wiederholung B1 S3 auf neuen Seeds |

**Beschreibend:**
- Mediane aller Metriken je Zelle und Form, Sättigung (weich: |Kern| > ρM, hart: |Kern| > M), Bereich, Gradientengefälle.
- **Bezug zu Blumenfeld et al. 2019:** 3-Bit- und 4-Bit-Kosten (w8a3_best − fp32, w8a4_best − fp32) bei d100 und d200 je Form.

## Deutungsregeln (vorab)

| Ergebnis | Deutung |
|---|---|
| Erfolg | Die weiche Lagma-Hülle hat einen eigenen Wert gegenüber der kritischen hard-tanh. BPreCap/LB mit φ bleiben die Empfehlung, und die Lagma-Form ist als Baustein gerechtfertigt |
| Primär „schlechter als LBh“ | Hartes Abschneiden ist bei 4 Bit besser. Die weiche Hülle bringt nichts, die Empfehlung wäre die bekannte kritische hard-tanh |
| Primär „kein Effekt“ | Die Form der Hülle ist egal. Der Gewinn kommt allein aus „kritisch + begrenzt“. Die Lagma-Form hat hier keinen messbaren Beitrag |
| Primär „besser“, H1 verletzt | Die weiche Hülle kauft 4 Bit mit fp32. Abwägung, kein klarer Vorzug |
| S2/S3 gleich gerichtet wie Primär/S1 | Der Befund hängt nicht vom Kern ab |
| S6 „kein Effekt“ | Auch bei harter Grenze bringt der PreCap-Kern in fp32 nichts (bestätigt B1) |
| S7 „schlechter“, S8 „kein Effekt“ | B1 wiederholt sich auf neuen Seeds |
| 3-Bit-Kosten bei d200 deutlich größer als bei d100 (beschreibend) | Passt zur Tiefengrenze von Blumenfeld et al., auch bei Quantisierung nach dem Training |
| Kontrolle nicht bitgleich | Abweichung wird gesucht und berichtet, die Tests gelten (Paarung bleibt) |

## Vorab-Erwartung

| Größe | Erwartung |
|---|---|
| Primär „besser als LBh“ | 0,3 |
| Primär „schlechter als LBh“ | 0,25 |
| Primär „kein Effekt“ oder „gemischt“ | 0,45 |
| H1 hält | 0,7 |
| Erfolg | 0,2 |
| S1 „kein Effekt“ | 0,5 |
| S6 „kein Effekt“ | 0,6 |
| S7 „schlechter als LB“ | 0,85 |
| S8 „kein Effekt“ | 0,7 |
| 3-Bit-Kosten d200 > d100 bei allen Formen | 0,6 |
| Kontrolle bitgleich | 0,95 |

## Neuheit und Literatur (ehrlich)

- **Bekannt:**
  - kritische hard-tanh (Pennington, Schoenholz & Ganguli 2017, https://arxiv.org/abs/1711.04735)
  - Tiefengrenze bei quantisierten Aktivierungen (Blumenfeld, Gilboa & Soudry 2019, https://arxiv.org/abs/1906.00771)
  - χ1 = 1 + ε mit ε < 1/L (Chen et al. 2020, https://arxiv.org/abs/2001.00254)
  - gelernte harte Clips (PACT), schichtweise Quantilgrenzen (SCULPT 2026)
- **Nicht gefunden:** der direkte Vergleich weich gegen hart bei gleicher Kritikalität, gemessen bei Quantisierung nach dem Training.
- B2 ist ein Baustein für die Entscheidung über BPreCap und Lagma, kein Neuheitsanspruch.

## Grenzen

- Nur MLP ohne Residuen, einfache Daten (MNIST, Fashion), zwei Tiefen, 12 Seeds.
- Lernrate nicht für die neuen Formen getunt.
- ρ = 0,8 und M = 3 (BPreCap) fest. Die weiche und die harte Grenze unterscheiden sich durch die gleiche Kritikalität leicht im M.
- Nur Quantisierung nach dem Training, pro Tensor.

## Freeze

Eingefroren werden `b2_run.py`, `b2_analyze.py`, `test_b2_mechanik.py`, `test_b2_analyse.py`, diese Datei, `lr_from_a.csv`, `lr_from_p3a3.csv`, `p7_hard_bound.json` und `src/precap/*.py`.

Vorab gemachte Tests (30.10. vormittags):

- **Selbsttest:** n_confirm 194. Alle Formen haben Mittel ≈ 0, Varianz ≈ 1 und χ1 auf Soll (Gauß-Hermite). MLX gleich numpy auf 1e-6.
- **Mechanik** (`test_b2_mechanik.py`): MECHANIK OK.
  - LBh und kL2Bh am Fixpunkt (Gitter): Mittel < 1e-11, Varianz 1,000000, χ1 1,020000 bzw. 1,010000. Beide begrenzt und gleich `mx.clip`.
  - LB und kL2B: Konstanten, Aktivierung und Gains gleich B1. `execute` ist auf synthetischen Daten bitgleich zu B1.
  - 3 Bit: 7 bzw. 8 Stufen, exakte 0. 4 Bit unverändert.
  - Lernraten wie kL2 in B1, Seeds neu, 194 Läufe.
- **Auswertung an künstlichen Daten** (`test_b2_analyse.py`): AUSWERTUNG OK.
  - „Erfolg“ und „weich teuer in fp32“ werden erkannt.
  - „Null“ ergibt 6 von 6 „kein Effekt“.
  - Eine falsche Kontrolle (1e-6) wird erkannt, ein Ausfall gezählt.
  - **Ein Aufruf mit „.“ aus dem Ergebnisordner findet B1**, die Lehre aus B1 ist damit umgesetzt.
