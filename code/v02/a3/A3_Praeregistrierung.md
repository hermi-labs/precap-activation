# A3 – Präregistrierung: kL2 im TAT-Rezept gegen TAT (privat)

Stand: 27.10.2026, 23:45, vor dem Start. Entworfen von Claude nach dem Ergebnis von A2, mit Aarons Freigabe für Folgeversuche in dieser Nacht.

## Frage

A2 ergab: kL2 ist in allen 8 Zellen schlechter als TAT. TAT lief dabei im eigenen Rezept (SUO-Init, Normierung je Beispiel), kL2 im Rezept aus A (Gauß-Init, keine Normierung). Kommt der Abstand von der Aktivierungsform oder vom Rezept?

## Eine einzige Änderung

kL2 bekommt genau das TAT-Rezept. Diese Variante heißt kL2r:

- skaliert-orthogonale Init (SUO wie in dks, wie TAT in A2), Bias 0
- Eingabe je Beispiel auf mittleres Quadrat 1 normiert (wie TAT in A2)

Aktivierung, Konstanten (λ, α, b wie in A), Netz, Optimierer, Daten, Budgets und Seeds bleiben wie in A2. Die LR wird neu getunt, weil eine andere Init eine andere beste LR haben kann. Raster, Tuning-Seeds und Auswahlregel sind dieselben wie für TAT in A2.

Code: `a3_run.py` ist aus `a2_run.py` abgeleitet. Geändert sind nur die Variantenliste, die Zuordnung von Init und Normierung zu kL2r, die LR-Quelle und die Reproduktionsliste.

## Varianten und Vergleichsdaten

| Name | Quelle |
|---|---|
| kL2r | neu in A3 |
| TAT | aus A2 (`studien/a2/workers/`), gleiche Seeds, gleicher Code-Pfad |
| kL2 (Gauß-Rezept) | kurzes Budget aus A, langes aus A2 (nur für S1) |

## Läufe

- Tuning: kL2r, LR-Raster 1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, Seeds 42000–42003, 1720 Schritte, Tiefe 100/200, MNIST/Fashion. Zusammen 96 Läufe. Gewählt wird die höchste mediane Val-Genauigkeit am letzten Schritt. Divergenz zählt als 0,1. Bei Gleichstand gewinnt die kleinere LR.
- Bestätigung:
  - 1720 Schritte mit Seeds 42010–42024 (15)
  - 6880 Schritte mit Seeds 42010–42019 (10)
  - zusammen 100 Läufe
- Reproduktion: TAT MNIST d100, 1720 Schritte, Seeds 42010/42011, neu gerechnet. Bestanden, wenn jede Abweichung zu A2 höchstens 0,10 pp beträgt. Sonst wird das vermerkt, und alle Primärtests gelten nur als vorläufig.
- Ein Ausfall (nicht endlicher Loss) zählt als Testfehler 90 %.
- Gesamt: 96 + 102 = 198 Läufe. Geschätzte Laufzeit etwa 2,5 h auf 2 Workern.

## Primärtest

kL2r gegen TAT im fp32-Testfehler, gepaart je Seed:

- 4 Zellen × 2 Budgets = 8 Tests
- Wilcoxon zweiseitig, Holm über alle 8, α = 0,05

Gesamturteil wie in A und A2:

| Urteil | Bedingung |
|---|---|
| besser als TAT | mindestens 2 Zellen besser, keine schlechter |
| schlechter als TAT | mindestens 2 Zellen schlechter, keine besser |
| kein Effekt | keine Zelle signifikant |
| gemischt | alles andere |

## Sekundär

- **S1:** kL2r gegen kL2 (Gauß-Rezept), 8 Tests, Holm. Frage: Wie viel bringt das Rezept allein?
- **Beschreibend:** Train-Loss, Aktivierungsbereich, Gradientengefälle bei Init, LR am Rasterrand.

## Vorab-Erwartung

kL2r gegen TAT:

| Urteil | Erwartung |
|---|---|
| besser als TAT | 0,15 |
| kein Effekt | 0,30 |
| schlechter als TAT | 0,40 |
| gemischt | 0,15 |

S1 (Rezept hilft kL2): besser in mindestens 2 Zellen 0,6.

Begründung: TAT hat bei Init ein fast flaches Gradientengefälle (log10 ≈ 0), kL2 eines von etwa 0,45. Das kommt von χ1 = 1 + 2/L und nicht vom Rezept. Das Rezept kann helfen, schließt diesen Abstand aber vermutlich nicht ganz.

## Entscheidungsregel und Bedeutung

- **kL2r „besser“ oder „kein Effekt“:** Der Abstand in A2 kam vom Rezept. Die beschränkte, glatte Form ist dann gleichwertig zu TAT oder besser. Das kann so in v0.2.
- **kL2r „schlechter“:** Auch mit gleichem Rezept ist TAT besser. Der Vorteil von kL2 liegt dann nur gegenüber SELU. v0.2 berichtet das ehrlich, und die Nische für PreCap ist die Beschränktheit (Quantisierung, Lagma).
- **„gemischt“:** Die Zellen werden einzeln berichtet, es gibt keine Gesamtaussage.

## Konstanten

Breite 256, Batch 128, Adam (0,9/0,999/1e-8), fp32, kein LayerNorm, Val 5000. Daten: `data/{mnist,fashion}`. kL2-Konstanten: d100 λ = 2,649586999, α = 1,461343209, b = 0,318766948; d200 λ = 4,170485099, α = 1,296823062, b = 0,212387949.

## Grenzen

- Adam statt K-FAC für alle (wie A2).
- Die LR für 6880 Schritte ist dieselbe wie für 1720 (wie A2).
- Nur MLPs, MNIST/Fashion, fp32. Quantisierung ist nicht Teil von A3.
- kL2-Konstanten wurden für Gauß-Eingaben kalibriert. Mit SUO und normierter Eingabe ist die Vorverteilung ebenfalls etwa N(0,1). Das wird im Selbsttest nicht geprüft, sondern über das Gradientengefälle beschreibend berichtet.

## Freeze

Eingefroren werden `a3_run.py`, `a3_analyze.py`, `lr_tat_from_a2.csv` und diese Datei (`meta/frozen_sha256.txt`). Die Auswertung wurde vorab an künstlichen Daten getestet: Ein eingebauter Effekt von −0,5 pp wurde erkannt, ein Nulleffekt nicht, und ein Ausfall wurde als 90 % gezählt. Ein Trockenlauf gegen die echten A/A2-Daten hat die Zuordnung geprüft: kL2 kurz 15, TAT kurz 15, kL2 lang 10, TAT lang 10, Reproduktion 0,00 pp.
