# A2 – Auswertung kL2 gegen TAT und SELU

*Veröffentlichte Fassung vom 29.10.2026: Der Bericht ist inhaltlich unverändert. Entfernt wurden nur der Vermerk „privat“ und Querverweise auf noch unveröffentlichte Studien. Präregistrierung und Skripte sind die eingefrorenen Originale.*

Stand: 27.10.2026, 23:15. Die Studie lief von etwa 19:15 bis 23:09 auf dem Mac M1 Pro (MLX). Die Auswertung folgt strikt der Präregistrierung (`A2_Praeregistrierung.md`).

## Ergebnis in Kürze

- **Urteil laut Präregistrierung: kL2 ist „schlechter als TAT“.** Alle 8 Primärzellen sind signifikant schlechter (Holm-p 0,0005–0,0098), in beiden Budgets, auf beiden Datensätzen und bei beiden Tiefen.
  - Bei 1720 Schritten liegt kL2 1,5–2,5 pp hinter TAT, in jeder Zelle mit 15 von 15 Seeds.
  - Bei 6880 Schritten sind es 0,4–1,9 pp, mit 9/10 bis 10/10 Seeds.
- **Vorab-Erwartung:** Für „schlechter als TAT“ hatte ich 0,20 angesetzt. Eingetreten ist also ein Ergebnis, das ich für eher unwahrscheinlich gehalten hatte.
- **Datenlage:** 96/96 Tuning- und 184/184 Bestätigungsläufe ok, keine Ausfälle. Die Hashes sind ok. Die Reproduktion gegen A liegt bei 0,000 pp in allen 4 Läufen, die Übernahme der kL2- und SELU-Werte aus A ist damit zulässig.
- **Sekundär:**
  - kL2 bleibt besser als SELU, auch bei 6880 Schritten (S1).
  - TAT ist in allen 8 Zellen besser als SELU (S2).
- **Bedeutung laut Präregistrierung:** Der Vorteil aus A ist im Wesentlichen die bekannte Tiefenabstimmung. Er wird ehrlich so berichtet und nicht als Neuheit verkauft.
- **Wichtige Einschränkung:** Verglichen wurden zwei vollständige Rezepte. TAT lief wie im TAT-Rezept mit skaliert-orthogonaler Init (SUO) und mit Normierung der Eingabe je Beispiel. kL2 lief wie in A mit Gauß-Init und ohne diese Normierung. Ob der Abstand von der Aktivierungsform kommt oder vom Rezept, kann A2 nicht trennen. Siehe Folgeversuch A3.

## Befehle und Prüfungen

```
shasum -a 256 -c meta/frozen_sha256.txt
  a2_run.py: OK / a2_analyze.py: OK / lr_from_a.csv: OK / A2_Praeregistrierung.md: OK
workers/w0.csv 92 Zeilen, 92 ok; workers/w1.csv 92/92; tune/workers/w0.csv 48/48; tune/workers/w1.csv 48/48
.venv/bin/python -I studien/a2/a2_analyze.py studien/a2 studien/a
  Urteil kL2 vs TAT: schlechter als TAT {1720: 'schlechter als TAT', 6880: 'schlechter als TAT'}
  | Sekundaer: {'S1': 'besser als SELU', 'S2_1720': 'besser als SELU', 'S2_6880': 'besser als SELU'}
  | Repro ok: True | Ausfaelle: 0 | LR am Rand: 1
```

Hinweis: In der Präregistrierung heißt die Hash-Datei `FROZEN_SHA256.txt`. Tatsächlich liegt sie unter `meta/frozen_sha256.txt` und enthält genau die vier dort genannten Dateien. Inhaltlich ist das keine Abweichung.

Sicherung der Rohdaten: `sicherung/a2_2026-10-27/` (Worker-CSV, Tuning, Logs, `lr_tuned.csv`, mit `SHA256.txt`).

## Primärtest: kL2 gegen TAT (fp32-Testfehler, Wilcoxon zweiseitig, Holm über 8)

| Budget | Zelle | Median-Diff. kL2 − TAT | Seeds kL2 besser/schlechter | Holm-p | Urteil |
|---|---|---|---|---|---|
| 1720 | MNIST 100 | +1,51 pp | 0/15 | 0,0005 | schlechter |
| 1720 | Fashion 100 | +2,07 pp | 0/15 | 0,0005 | schlechter |
| 1720 | MNIST 200 | +1,63 pp | 0/15 | 0,0005 | schlechter |
| 1720 | Fashion 200 | +2,45 pp | 0/15 | 0,0005 | schlechter |
| 6880 | MNIST 100 | +0,41 pp | 1/9 | 0,0098 | schlechter |
| 6880 | Fashion 100 | +1,01 pp | 1/9 | 0,0078 | schlechter |
| 6880 | MNIST 200 | +1,86 pp | 0/10 | 0,0078 | schlechter |
| 6880 | Fashion 200 | +1,75 pp | 0/10 | 0,0078 | schlechter |

## Sekundär

**S1: kL2 gegen SELU bei 6880 Schritten (Holm über 4).** d100 ist besser (MNIST −1,08 pp, Fashion −2,87 pp, je 10/0). Bei d200 ist das Ergebnis unklar (−4,56 bzw. −15,55 pp im Median, aber nur 8/2 Seeds, Holm-p 0,55). SELU trainiert bei d200 mit langem Budget in einzelnen Seeds doch durch. Vorab erwartet war „SELU holt bei d100 auf, bei d200 bleibt kL2 klar vorn“ (0,7). Eingetreten ist eher das Gegenteil: Bei d100 bleibt kL2 klar vorn, bei d200 streut SELU stark.

**S2: TAT gegen SELU (Holm über 8).** TAT ist in allen 8 Zellen besser (−1,6 bis −35,5 pp).

## Beschreibend (Mediane)

| Zelle | Budget | kL2 | TAT | SELU |
|---|---|---|---|---|
| MNIST 100 | 1720 | 5,04 % | 3,52 % | 6,46 % |
| MNIST 100 | 6880 | 3,21 % | 2,83 % | 4,34 % |
| Fashion 100 | 1720 | 15,62 % | 13,58 % | 16,64 % |
| Fashion 100 | 6880 | 13,08 % | 12,07 % | 15,93 % |
| MNIST 200 | 1720 | 5,44 % | 3,86 % | 39,53 % |
| MNIST 200 | 6880 | 4,71 % | 3,01 % | 9,73 % |
| Fashion 200 | 1720 | 16,53 % | 14,04 % | 32,19 % |
| Fashion 200 | 6880 | 14,73 % | 12,98 % | 33,09 % |

- **Lernkurve:** Der Abstand zu TAT schrumpft bei d100 mit längerem Training (MNIST +1,51 → +0,41 pp), bei d200 nicht (+1,63 → +1,86 pp). kL2 hat bei d200 MNIST einen Train-Loss von 0,149, TAT von 0,022. kL2 trainiert also langsamer und ist nicht stärker überangepasst.
- **Gradientengefälle bei Init (log10 erste/letzte Schicht):** kL2 0,45–0,53, TAT −0,09 bis +0,03, SELU 1,6–3,2. TAT hält den Gradientenfluss praktisch perfekt gleich. kL2 hat das vorhergesagte Gefälle (L−1)/2 · log10 χ1 ≈ 0,43.
- **Aktivierungsbereich (Median des Maximums):** kL2 7,0–11,2, TAT 4,9–16,8, SELU 5,2–8,8. TAT ist unbeschränkt und erreicht bei MNIST d100 im langen Lauf 16,8.
- **Tempo (ms/Schritt, Referenzpfad ohne Kernel):** kL2 ist 1,05–1,46-mal langsamer als TAT. Das ist nur untereinander vergleichbar.
- **LR am Rasterrand:** TAT bei Fashion d200 wählte 1e-5, den unteren Rand. TAT hätte mit kleinerer LR eventuell noch besser abgeschnitten. Das schwächt das Urteil nicht, es könnte den Abstand höchstens vergrößern. kL2 lag in A nirgends am Rand (gleiches Raster), die LR-Wahl war also für beide fair.

## Grenzen

- **Adam statt K-FAC:** TAT lief bewusst mit Adam und nicht mit K-FAC wie im Paper. Laut Paper ist TAT mit Adam schwächer. TAT hat also trotz dieses Nachteils gewonnen, das Urteil ist eher konservativ. Ein schwaches TAT-Ergebnis wäre vorsichtig zu deuten gewesen. Ein starkes TAT-Ergebnis ist umso belastbarer.
- **Rezept statt Einzeländerung:** Init (SUO gegen Gauß) und Eingabe-Normierung unterscheiden sich zwischen kL2 und TAT. A2 vergleicht Rezepte wie präregistriert, nicht reine Aktivierungsformen.
- **Umfang:** nur MLPs der Breite 256, MNIST/Fashion, fp32, zwei Budgets. Längere Läufe oder andere Architekturen können anders ausgehen.
- **Quantisierung** war in A2 nicht im Fokus. Ob kL2 durch die Beschränkung bei wenigen Bits gegenüber TAT etwas gewinnt (TAT ist unbeschränkt, Bereich bis 16,8), ist offen.

## Was daraus folgt

1. **Öffentlich darf kL2 nicht als Verbesserung gegenüber dem Stand der Technik bei tiefen MLPs dargestellt werden.** Belegbar ist: kL2 schlägt SELU bei Tiefe 100/200 klar. TAT, ein bekanntes Verfahren von 2022, ist in diesem Setup aber noch besser.
2. **Die naheliegende offene Frage ist A3:** Liegt der Abstand an der Aktivierung oder am Rezept? Dafür wird kL2 mit genau dem TAT-Rezept (SUO-Init, Normierung je Beispiel, LR neu getunt) gegen dieselben TAT-Läufe aus A2 gestellt. Das ist eine einzige Änderung, und sie entspricht im Kern der geplanten Phase 3a-4 (kritische orthogonale Init).
3. **Mögliche eigene Nische bleibt die Beschränktheit:** wenige Bits und Ausreißer. TAT ist unbeschränkt. Das spricht für die Richtung „A + Quantisierung“, muss aber erst gemessen werden.
4. **Für v0.2** heißt das: ehrlicher Bericht mit A, A2, Q1 und Q1b gemeinsam. Die Kernbotschaft kann nicht „besser als TAT“ sein.
