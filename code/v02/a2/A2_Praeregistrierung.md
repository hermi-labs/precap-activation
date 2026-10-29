# Studie A2 – Präregistrierung (privat)

Stand: 27.10.2026, vor dem ersten Lauf festgelegt. Danach werden nur noch Fehler korrigiert, und das wird hier offen vermerkt.

## Frage

Studie A hat gezeigt: Tiefen-PreCap kL2 (χ1 = 1 + 2/L) schlägt SELU bei Tiefe 100 und 200 deutlich. Damit ist aber noch nichts über den Stand der Technik gesagt. Die naheliegende Kritik lautet: „Das ist im Kern DKS/TAT, und TAT gibt es schon.“ A2 prüft deshalb zwei Dinge:

1. Ist kL2 bei gleichem Aufwand besser, gleich gut oder schlechter als TAT mit Leaky ReLU (Zhang, Botev, Martens 2022, [arXiv:2203.08120](https://arxiv.org/abs/2203.08120))?
2. Liegt der Vorsprung gegenüber SELU nur am kurzen Budget? Dazu kommt ein viermal längeres Budget.

## Varianten

| Name | Aktivierung | Init | LR |
|---|---|---|---|
| kL2 | PreCap, Konstanten wie in A (d100: λ=2.649586999, α=1.461343209, b=0.318766948; d200: λ=4.170485099, α=1.296823062, b=0.212387949) | Gauß, Gain wie in A | aus A übernommen |
| tat | Leaky ReLU nach TAT: φ(x) = s·(max(x,0) + a·min(x,0)), s = √(2/(1+a²)) | skaliert-orthogonal (SUO wie in dks), Bias 0; Eingabe je Beispiel auf mittleres Quadrat 1 normiert (wie im TAT-Rezept) | eigens getunt |
| selu | SELU | Gauß, Gain wie in A | aus A bzw. 3a-3 übernommen |

TAT-Parameter: Sie kommen direkt aus der offiziellen DeepMind-Bibliothek `dks` (`_get_activations_params(['leaky_relu'], method='TAT')` mit L-facher Komposition als Teilnetz-Funktion, Ziel η = 0,9):

- d100: a = 0,5704395324, s = 1,2284042441
- d200: a = 0,6795996428, s = 1,1696683780

Die eigene Kontrolle bestätigt das. Die 100- bzw. 200-fach verkettete C-Map ergibt bei c = 0 den Wert 0,9000000 (Abweichung < 1e-8). Der Skalenfaktor stimmt mit √(2/(1+a²)) überein. Die lokale C-Map bei c = 0 passt zur geschlossenen Formel.

Bewusste Abweichung vom TAT-Paper: Alle Varianten trainieren mit Adam statt K-FAC, damit der Optimierer gleich bleibt. Das Paper berichtet für Adam schwächere Ergebnisse als für K-FAC. Diese Abweichung wird im Bericht genannt.

## Aufbau

- MLP, Breite 256, Batch 128, Adam (0,9/0,999/1e-8), fp32, kein LayerNorm, keine Quantisierung (A2 hat nur fp32 im Fokus).
- Tiefen 100 und 200, Datensätze MNIST und Fashion-MNIST, also 4 Zellen.
- Budgets: 1720 Schritte (wie A) und 6880 Schritte (4×).
- Val-Split: die letzten 5000 Trainingsbilder, wie in A.

## LR-Wahl

- TAT: Raster 1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4; Seeds 42000–42003; 1720 Schritte; je Zelle 24 Läufe, zusammen 96. Gewählt wird die höchste mediane Val-Genauigkeit am letzten Schritt. Divergenz zählt als 0,1. Bei Gleichstand gewinnt die kleinere LR.
- kL2 und SELU: Die LR aus A wird unverändert übernommen (`lr_from_a.csv`). Für kL2 sind das 2e-5 in allen Zellen, für SELU 5e-5 bei d100 (aus 3a-3) und 2e-5 bei d200.
- Für 6880 Schritte gilt dieselbe LR wie für 1720. Das ist eine bekannte Schwäche: Ein längeres Budget hätte womöglich eine andere beste LR. Sie trifft aber alle drei Varianten gleich.

## Läufe

- 1720 Schritte, Seeds 42010–42024 (15):
  - TAT neu, 60 Läufe.
  - kL2 und SELU werden aus A übernommen (gleiche Seeds, gleicher Code-Pfad).
  - Reproduktionskontrolle: kL2 und SELU bei MNIST d100 mit Seeds 42010/42011 werden in A2 neu gerechnet, 4 Läufe. Sie gilt als bestanden, wenn jede Abweichung zum A-Wert höchstens 0,10 pp beträgt. Fällt sie durch, wird das im Bericht vermerkt, und die 1720er-Tests zu kL2 und SELU gelten nur als vorläufig.
- 6880 Schritte, Seeds 42010–42019 (10): alle 3 Varianten neu, 120 Läufe.
- Zusammen 96 Tuning- und 184 Bestätigungsläufe. Ein Ausfall (nicht endlicher Loss) zählt als Testfehler 90 %.

## Primärtest

Verglichen wird kL2 gegen TAT im fp32-Testfehler, gepaart je Seed:

- 4 Zellen × 2 Budgets = 8 Tests.
- Wilcoxon zweiseitig, Holm über alle 8, α = 0,05.

Jede Zelle bekommt ein Ergebnis: besser, schlechter oder unklar.

Gesamturteil (wie in A):

- „besser als TAT“: mindestens 2 Zellen besser und keine schlechter.
- „schlechter als TAT“: umgekehrt.
- „kein Effekt“: keine einzige Zelle signifikant.
- sonst „gemischt“.

Das Urteil wird zusätzlich je Budget berichtet, aber nur beschreibend.

## Sekundär (Holm je Familie)

- S1: kL2 gegen SELU bei 6880 Schritten, 4 Tests. Frage: Holt SELU mit längerem Training auf?
- S2: TAT gegen SELU, 8 Tests.
- Beschreibend:
  - Val-Fehler nach 1720/3440/6880 Schritten (Lernkurve)
  - Train-Loss
  - ms/Schritt
  - Gradientenverhältnis erste/letzte Schicht bei Init
  - Aktivierungsbereich
  - LR am Rasterrand

## Vorab-Erwartung

kL2 gegen TAT, Gesamturteil:

| Urteil | Erwartung |
|---|---|
| besser als TAT | 0,30 |
| kein Effekt | 0,35 |
| schlechter als TAT | 0,20 |
| gemischt | 0,15 |

Weitere Erwartungen:

- S1: SELU holt bei d100 einen Teil auf. Bei d200 bleibt kL2 klar vorn (0,7).
- Begründung: TAT ist ebenfalls auf die Tiefe abgestimmt und mit orthogonaler Init sehr stark. Der Unterschied liegt vor allem in der Form der Aktivierung, also glatt und beschränkt gegenüber stückweise linear.

## Was das Ergebnis bedeutet

- „besser als TAT“ oder „kein Effekt“: kL2 ist mindestens konkurrenzfähig zum Stand der Technik, dann ist v0.2 sinnvoll. Bei „kein Effekt“ wäre die Botschaft: gleichwertig, mit glatter, beschränkter Form.
- „schlechter als TAT“: Der Vorteil aus A ist im Wesentlichen bekannte Tiefenabstimmung. Dann wird ehrlich berichtet und nicht als Neuheit verkauft.

## Integrität

- Vor dem Start werden die sha256-Hashes von `a2_run.py`, `a2_analyze.py`, `lr_from_a.csv` und dieser Datei eingefroren (`FROZEN_SHA256.txt`).
- Die Auswertung prüft die Hashes.
- Alle Daten bleiben privat auf dem Mac (`studien/a2/`).
