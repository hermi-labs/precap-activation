# Idee A – Tiefen-PreCap: χ1 = 1 + k/L (Präregistrierung, privat)

Stand: 27.10.2026. Diese Präregistrierung wird vor dem Start eingefroren (SHA-256 in `meta/frozen_sha256.txt`). Danach ändern sich weder der Code noch die Regeln.

## Frage

Die Gradientenverstärkung über L Schichten wächst etwa wie χ1^((L−1)/2):

- **SELU/cs bei Tiefe 100:** etwa 35×
- **SELU/cs bei Tiefe 200:** etwa 970×

Setzt man χ1 = 1 + k/L, bleibt die Verstärkung unabhängig von der Tiefe ungefähr e^(k/2):

- **k = 1:** 1,65×
- **k = 2:** 2,7×

Die Frage ist: Trainieren tiefe, nicht normierte MLPs damit besser als SELU, vor allem bei Tiefe 100 und 200?

## Einordnung der Vorergebnisse

- **3a-3:** e03 (χ1 = 1,03, fest) war bei MNIST 100 besser als SELU.
- **C1:** Eine schwächere Stabilisierung des Varianz-Fixpunkts (dν'/dν nahe 1) schadet, und zwar umso mehr, je tiefer das Netz ist.
- **Zielkonflikt:** Bei festem s = 1,5 zieht ein kleineres χ1 automatisch dν'/dν Richtung 1 und vergrößert den Wertebereich (siehe Tabelle). A misst also genau diesen Zielkonflikt: weniger Gradientenwachstum gegen schwächere Stabilisierung.

## Varianten (eine Änderung gegenüber 3a-3: nur χ1 bzw. die Konstanten)

Die Form ist f(x) = λ·CapExp(s·tanh(b·x/s)), für x ≤ 0 zusätzlich mit α, und s = 1,5, c = 1. Die Konstanten wurden mit `a_constants.py` exakt gelöst (feines Gitter, E f = 0, E f² = 1).

| Variante | Tiefe | χ1 | λ | α | b | Wertebereich | dν'/dν | vorh. log10-Gefälle |
|---|---|---|---|---|---|---|---|---|
| cs (fest) | alle | 1,0716 | 1,1027 | 2,0538 | 0,7119 | [−1,76, 3,39] | 0,79 | 0,74 / 1,49 / 2,99 |
| e03 (fest) | alle | 1,030 | 1,9856 | 1,6081 | 0,4119 | [−2,48, 6,11] | 0,94 | 0,31 / 0,64 / 1,28 |
| kL1 | 50 | 1,020 | 2,6496 | 1,4613 | 0,3188 | [−3,01, 8,15] | 0,971 | 0,21 |
| kL1 | 100 | 1,010 | 4,1705 | 1,2968 | 0,2124 | [−4,20, 12,83] | 0,990 | 0,21 |
| kL1 | 200 | 1,005 | 6,2966 | 1,1978 | 0,1457 | [−5,86, 19,38] | 0,996 | 0,22 |
| kL2 | 50 | 1,040 | 1,6141 | 1,7397 | 0,4962 | [−2,18, 4,97] | 0,906 | 0,42 |
| kL2 | 100 | 1,020 | 2,6496 | 1,4613 | 0,3188 | [−3,01, 8,15] | 0,971 | 0,43 |
| kL2 | 200 | 1,010 | 4,1705 | 1,2968 | 0,2124 | [−4,20, 12,83] | 0,990 | 0,43 |
| SELU | alle | 1,0716 | – | – | – | [−1,76, ∞) | 0,78 | 0,74 / 1,49 / 2,99 |

Das vorhergesagte log10-Gefälle ist (L−1)/2·log10(χ1), angegeben für die Tiefen 50 / 100 / 200.

## Aufbau (wie 3a-3 / C1)

- **Netz:** MLP, Breite 256, L = 50 / 100 / 200 verdeckte Schichten, ohne Normierung und ohne Residuals. Init Gauss mit g = 1/√E[f²] wie in 3a-3, die kritische Init (3a-4) ist bewusst nicht dabei.
- **Training:** Adam, Batch 128, 1720 Schritte, MNIST und Fashion-MNIST, 5000 Validierungsbilder aus dem Training.
- **Rechenweg:** MLX auf dem Mac M1 Pro. Alle PreCap-Varianten laufen über dieselbe MLX-Referenzfunktion, ohne Kernel. SELU ist nativ.

## Stufe 1: LR-Tuning

- **Raster:** 1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4. Seeds 42000–42003.
- **Wahlregel:** höchste mediane val_acc am letzten Schritt. Ein divergierter Lauf zählt als 0,1. Bei Gleichstand gewinnt die kleinere LR.
- **Getunt werden:**
  - kL1 und kL2 in allen 6 Zellen (288 Läufe)
  - cs, e03 und SELU bei Tiefe 200 (144 Läufe)
- **Übernommen aus 3a-3** (`lr_from_p3a3.csv`): cs, e03 und SELU bei Tiefe 50 und 100.
- **Summe:** 432 Läufe.

## Stufe 2: Bestätigung

Bestätigt werden 5 Varianten (cs, e03, kL1, kL2, SELU) in 6 Zellen mit je 15 Seeds (42010–42024). Das sind 450 Läufe.

Gemessen wird:

- **Testfehler:** fp32 sowie Quantisierung wie in Q1b und C1
- **Wertebereich der Aktivierungen**
- **ms pro Schritt**
- **Gradientengefälle bei der Init:** log10(‖∇W₁‖/‖∇W_L‖) auf dem ersten Batch, vor dem ersten Schritt

## Primärtest

- **Vergleich:** kL1 und kL2 gegen SELU im fp32-Testfehler, in den Zellen Tiefe 100 und 200 × MNIST und Fashion. Das sind 2 × 4 = 8 Tests.
- **Statistik:** Wilcoxon-Vorzeichen-Rangtest, gepaart über Seeds, zweiseitig, Holm über alle 8 Tests, α = 0,05.
- **Ausfälle:** Ein divergierter Lauf zählt mit 90 % Fehler.
- **Urteil je Variante:**

  | Urteil | Bedingung |
  |---|---|
  | "besser als SELU" | mindestens 2 Zellen besser, keine schlechter |
  | "schlechter als SELU" | mindestens 2 Zellen schlechter, keine besser |
  | "kein Effekt" | keine signifikante Zelle |
  | "gemischt" | sonst |

- **Erfolg von Idee A:** mindestens eine Variante "besser als SELU".

## Sekundär (nur beschreibend bzw. mit eigenem Holm je Familie)

- **S1:** kL gegen cs, alle 6 Zellen (Holm über 12)
- **S2:** kL gegen e03, alle 6 Zellen (Holm über 12)
- **S3:** kL gegen SELU bei Tiefe 50 (Holm über 4)
- **S4:** cs und e03 gegen SELU, alle Zellen (Holm über 12). Bei Tiefe 50/100 ist das eine Wiederholung von 3a-3 mit denselben Seeds.
- **S5:** Quantisierungsverlust w8a4_best und w8a8, gegen SELU und gegen cs.
- **Gradiententheorie:** gemessenes gegen vorhergesagtes log10-Gefälle je Variante und Tiefe, nur beschreibend.
- **Grenzfall:** LR am Rand des Rasters wird gemeldet.

## Vorab-Erwartung

| Aussage | Wahrscheinlichkeit |
|---|---|
| kL2 "besser als SELU" | 0,35 |
| kL1 "besser als SELU" | 0,25 |
| mindestens eine kL-Variante "besser als SELU" (Erfolg A) | 0,4 |
| kL1 bei Tiefe 200 schlechter als kL2 (Wertebereich 19, dν'/dν 0,996) | 0,6 |
| gemessenes Gefälle folgt der Vorhersage (Rangfolge der Varianten stimmt) | 0,8 |

## Grenzen

- Nur MLPs ohne Normierung, MNIST und Fashion, 1720 Schritte.
- Die kL-Varianten haben bei Tiefe 100/200 einen deutlich größeren Wertebereich als cs. Der Vorteil "beschränkt" wird also teilweise aufgegeben, das betrifft vor allem die Quantisierung.
- Die kritische Init (3a-4) bleibt für eine eigene Studie.

## Dateien

- `a_run.py`, `a_analyze.py`, `a_constants.py/.json`, `lr_from_p3a3.csv`, `src/precap/`, `launch_a.sh`
- Ergebnisse: `tune/`, `lr_tuned.csv`, `workers/`, `analysis/a_result.json`
