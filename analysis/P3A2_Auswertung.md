# Phase 3a-2 – Auswertung (niedrige Lernraten, Tiefe 50 und 100)

Lauf: 26.10.2026, 11:18–15:42 auf dem M1 Pro. Präregistrierung: `docs/P3A2_Praeregistrierung.md`. Code, Rohdaten und automatische Auswertung liegen unter `code/phase3/p3a2_low_lr/`.

## Kurzfazit

- **Primärtest negativ:** SPreCap ist in allen 4 Zellen signifikant schlechter als SELU (Nachteil bestätigt, 0 von 4 Zellen besser).
- **PreCap0 lernt bei Tiefe 100,** wenn die LR klein genug ist (≤ 2e-5). Bei FashionMNIST erreicht es 18,2 % Testfehler, bei MNIST 8,3 %. Der Ausfall in 3a lag also an der LR, nicht an der Funktion an sich.
- **Gradientenprofil bestätigt die χ₁-Theorie:** SPreCap (χ₁ = 1,1305) ist bei Tiefe 100 zur Eingabe hin etwa 600-fach "aufgeblasen", SELU (χ₁ = 1,0716) etwa 35-fach. Das passt fast genau zur Mean-Field-Vorhersage und begründet 3a-3.
- **Beste Funktionen:** ELU (mittlerer Rang 1,25) und SELU (1,75), danach PreCap0 (3,0), ReLU (4,0) und SPreCap (5,0).
- **Keine Divergenz:** In keinem der 1092 Runs gab es nicht-endliche Werte. Bei so kleinen LRs zeigt sich der Stabilitätsvorteil beschränkter Funktionen also nicht.

Konfidenz in das Hauptergebnis: etwa 0,9. Der Effekt ist groß (14–15 von 15 Seeds schlechter, bei Tiefe 100 7–8 Prozentpunkte) und wurde unabhängig nachgerechnet.

## Prüfungen

- **Hashes:** `run_p3a2.py`, `analyze_p3a2.py` und `launch_p3a2.sh` sind auf dem Mac identisch mit dem eingefrorenen Stand. Auch der Launcher hat den Abgleich vor dem Start bestanden.
- **Integrität:**
  - 5460 Zeilen, 1092 von 1092 Runs (672 Tuning, 420 Bestätigung)
  - keine Duplikate, keine fehlenden und keine unerwarteten Runs
  - 0 nicht-endliche Werte
- **LR-Wahl:** Die Datei, die den Bestätigungslauf gesteuert hat, stimmt mit der Neuberechnung überein (`lr_selection_matches_file = true`). Ihr Hash ist gespeichert.
- **Rohdaten:** Die Rohdaten (`raw/p3a2_workers.tbz`, SHA-256 9b65425d…) wurden byte-genau übertragen. Ich habe die Analyse damit in der Sandbox erneut laufen lassen.
  - LR-Wahl, Sekundärtabelle, Tuning-Tabelle und `verdict.json` sind identisch.
  - Bei einigen p-Werten gibt es Unterschiede, weil die scipy-Versionen verschieden sind (Mac 1.13.1, Sandbox 1.18.1). Die neue Version nutzt bei Bindungen eine Näherung, die alte rechnet exakt.
  - Alle Urteile sind trotzdem identisch.
  - Ein eigener exakter Permutationstest liefert für den Primärtest genau die Mac-Werte (0,00201 / 6,1e−5 / 0,000122 / 6,1e−5).
  - Die Reproduktion liegt in `analysis_reproduktion_scipy118/`.

## Primärtest: SPreCap gegen SELU (Holm über 4)

| Zelle | SPreCap | SELU | Differenz (Median) | SPreCap besser | p (Holm) | Urteil |
|---|---|---|---|---|---|---|
| Fashion, 50 | 16,0 % | 14,8 % | +1,13 pp | 3 von 15 | 0,0020 | schlechter |
| Fashion, 100 | 24,2 % | 17,2 % | +7,10 pp | 0 von 15 | 0,00024 | schlechter |
| MNIST, 50 | 5,1 % | 4,5 % | +0,64 pp | 1 von 15 | 0,00024 | schlechter |
| MNIST, 100 | 13,9 % | 6,1 % | +8,03 pp | 0 von 15 | 0,00024 | schlechter |

## Medianer Testfehler (Bestätigung, 15 Seeds)

| Zelle | ELU | SELU | PreCap0 | ReLU | SPreCap | GELU (He) | SiLU (He) |
|---|---|---|---|---|---|---|---|
| Fashion, 50 | 14,1 | 14,8 | 14,9 | 15,5 | 16,0 | 17,2 | 17,3 |
| Fashion, 100 | 16,5 | 17,2 | 18,2 | 21,7 | 24,2 | 90,0 | 90,0 |
| MNIST, 50 | 3,9 | 4,5 | 4,9 | 5,1 | 5,1 | 5,8 | 6,8 |
| MNIST, 100 | 6,4 | 6,1 | 8,3 | 11,5 | 13,9 | 88,6 | 88,6 |

Vergleich mit 3a bei Tiefe 100: Mit den kleineren LRs verbessert sich SELU stark (Fashion von 37,1 auf 17,2 %, MNIST von 14,1 auf 6,1 %), SPreCap nur mäßig (28,1 auf 24,2 % bzw. 19,4 auf 13,9 %). Der scheinbare Vorteil von SPreCap bei Fashion 100 in 3a war also eine Folge des zu kleinen LR-Rasters.

## Sekundärtests (40, Holm)

- **PreCap0:**
  - besser als SiLU und GELU (je 4 von 4 Zellen)
  - besser als ReLU (2 von 4, bei Tiefe 100)
  - schlechter als ELU (4 von 4) und SELU (3 von 4)
- **SPreCap:**
  - besser als SiLU (4 von 4) und GELU (3 von 4)
  - schlechter als ELU (4 von 4)
  - gegen ReLU: 1 von 4 Zellen schlechter, sonst unklar
- **SiLU und GELU mit He-Init (g = √2)** bleiben bei Tiefe 100 auf Zufallsniveau. Die Gradienten in der ersten Schicht liegen beim Start bei 1e−10 bis 1e−14. Diese Vergleiche sind deshalb kein fairer Erfolg von PreCap: Sie zeigen nur, dass diese Init bei 100 Schichten ohne Normierung nicht funktioniert.

## Mechanismus: Gradientenprofil beim Start

Gemessen wurde $\log_{10}(\lVert g_1\rVert/\lVert g_L\rVert)$, Median über die Bestätigungs-Seeds. Die Vorhersage aus der Mean-Field-Theorie ist $\tfrac{L-1}{2}\log_{10}\chi_1$.

| Funktion | χ₁ | Tiefe 50: gemessen (Fashion / MNIST) | vorhergesagt | Tiefe 100: gemessen | vorhergesagt |
|---|---|---|---|---|---|
| SPreCap | 1,1305 | 1,33 / 1,36 | 1,31 | 2,77 / 2,81 | 2,64 |
| SELU | 1,0716 | 0,69 / 0,71 | 0,73 | 1,54 / 1,56 | 1,49 |
| PreCap0 | 1,064 | 0,46 / 0,46 | 0,66 | 1,06 / 1,08 | 1,33 |
| ReLU | 1 | −0,01 / −0,05 | 0 | −0,12 / −0,16 | 0 |

- **SPreCap, SELU und ReLU:** Die Vorhersage stimmt auf etwa ±0,15 Dekaden.
- **PreCap0:** Die Abweichung ist größer, vermutlich weil der Mittelwert ungleich 0 ist und der Fixpunkt q* sich verschiebt.
- **Vorregistrierte Erwartung widerlegt:** Ich hatte für PreCap0 ein "deutlich negativeres Profil" erwartet. Tatsächlich liegt es zwischen SELU und ReLU. Die Gradienten verschwinden also nicht.

## LR-Abhängigkeit bei Tiefe 100 (Tuning, mediane val_acc, explorativ)

| Funktion | 1e-5 | 2e-5 | 5e-5 | 1e-4 | 2e-4 | 5e-4 |
|---|---|---|---|---|---|---|
| SELU (Fashion) | 0,80 | 0,83 | 0,85 | 0,19 | 0,10 | 0,10 |
| PreCap0 (Fashion) | 0,80 | 0,83 | 0,10 | 0,14 | 0,10 | 0,10 |
| SPreCap (Fashion) | 0,61 | 0,70 | 0,76 | 0,70 | 0,19 | 0,13 |
| ELU (Fashion) | 0,83 | 0,85 | 0,85 | 0,52 | 0,70 | 0,58 |
| SELU (MNIST) | 0,89 | 0,93 | 0,95 | 0,14 | 0,11 | 0,10 |
| PreCap0 (MNIST) | 0,89 | 0,93 | 0,21 | 0,11 | 0,11 | 0,10 |
| SPreCap (MNIST) | 0,39 | 0,65 | 0,85 | 0,90 | 0,11 | 0,10 |

Beobachtungen (explorativ, nur 4 Seeds):
- **Der "Tod" ist ein Kollaps ohne Divergenz.** Er tritt bei allen selbstnormierenden Funktionen auf, sobald die LR zu groß ist: Das Netz gibt dann eine konstante Ausgabe aus, aber keine NaN-Werte. SELU kollabiert ab 1e-4, PreCap0 schon ab 5e-5.
- **SPreCap verträgt eine LR von 1e-4 noch gut** (0,70 bzw. 0,90). SELU und PreCap0 sind dort schon kollabiert. Bei kleinen LRs lernt SPreCap dagegen deutlich langsamer.
- **Mögliche Erklärung:** Das größere χ₁ wirkt wie eine größere effektive LR in den ersten Schichten, also ein "eingebauter Verstärker". Das ist eine Hypothese und nicht vorregistriert.

## Vorregistrierte Erwartungen – Abgleich

| Frage | Erwartung (Konfidenz) | Ergebnis |
|---|---|---|
| SPreCap gegen SELU (primär) | kein bestätigter Unterschied (0,55) | falsch, Nachteil bestätigt |
| SPreCap bei Tiefe 100 eher besser | ja (0,5) | falsch |
| PreCap0 lernt bei Tiefe 100 mit kleiner LR | nein (0,6) | falsch, lernt gut |
| Gradientenprofil PreCap0 deutlich negativer | ja (0,7) | falsch |
| Gewählte LRs nicht am Rand | ja (0,7) | richtig (nur SiLU/GELU am Rand) |
| SiLU/GELU mit He-Init lernen bei Tiefe 100 besser | ja (0,6) | falsch |

Nur 1 von 6 Erwartungen hat gestimmt. Meine Einschätzungen vor dem Lauf waren also schlecht kalibriert. Genau deshalb ist die Vorregistrierung wichtig.

## Folgerungen

1. **SPreCap in der jetzigen Form bewährt sich nicht.** Gegen SELU ist es klar unterlegen.
2. **Die naheliegende Ursache ist messbar:** χ₁ ist zu groß. 3a-3 testet genau diese Stellschraube mit `spreccap_cs` (χ₁ = SELU) und `spreccap_e03` (χ₁ = 1,03), dazu eine Variante mit lernbarem s. Das ist bereits vorregistriert, die Hashes sind eingefroren.
3. **PreCap0 ist rehabilitiert,** aber nicht besser als SELU oder ELU.
4. **Für das Zwischenziel v0.1** entscheidet die vorregistrierte Regel in 3a-3, ob SPreCap "bewährt" ist. Fällt sie negativ aus, wird der Stand trotzdem ehrlich abgesichert, dann aber ohne Kernels.
