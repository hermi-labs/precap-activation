# Phase 3a-3 – Auswertung (χ₁-angepasstes SPreCap, Tiefe 50 und 100)

Der Lauf fand am 26.10.2026 von 16:25 bis 21:20 auf dem M1 Pro statt. Die Präregistrierung steht in `docs/P3A3_Praeregistrierung.md`. Code, Rohdaten und automatische Auswertung liegen unter `code/phase3/p3a3_chi1/`.

## Kurzfazit

- **Primärtest: gleichauf mit SELU.** SPreCap-cs (χ₁ = 1,0716, gleich wie SELU) liegt in allen 4 Zellen höchstens 0,2 Prozentpunkte von SELU entfernt (−0,12 bis +0,20). In keiner Zelle ist es signifikant schlechter oder besser.
- **v0.1 nach der vorregistrierten Regel: bewährt.** Keine Zelle ist signifikant schlechter als SELU, und in 4 von 4 Zellen liegt der Unterschied bei ≤ +0,5 Prozentpunkten (nötig waren 3 von 4).
- **Der Nachteil von SPreCap aus 3a-2 lag an χ₁.** Mit derselben Formel und nur angepasstem χ₁ fällt der Abstand zu SELU bei Tiefe 100 von 7 Prozentpunkten auf 0. SPreCap-cs ist in 4 von 4 Zellen bestätigt besser als das alte SPreCap.
- **Dosis-Wirkung bestätigt: kleineres χ₁ ergibt kleineren Fehler.** Die Variante mit χ₁ = 1,03 (SPreCap-e03) liegt auf dem besten mittleren Rang (1,5), noch vor ELU (1,75). Bei MNIST mit Tiefe 100 ist sie bestätigt besser als SELU (6,4 % → 5,3 % Fehler, Holm p = 0,0064).
- **Robustheit bei hoher LR: kein Vorteil.** Bei LR 5e-4 und Tiefe 50 lernen SELU und ELU noch (60–72 % val acc), alle PreCap-Varianten sterben dort (10–15 %). Die tanh-Grenze macht die Funktion bei hohen Lernraten also nicht robuster, eher im Gegenteil.
- **Keine Divergenz:** 0 von 936 Runs mit nicht-endlichen Werten.

Konfidenz in das Hauptergebnis (gleichauf mit SELU): etwa 0,85. Konfidenz in den Vorteil von e03: etwa 0,6. Er ist nur in einer Zelle bestätigt, und e03 war ein Sekundärtest.

## Prüfungen

- **Hashes:** `run_p3a3.py`, `analyze_p3a3.py` und `launch_p3a3.sh` sind auf dem Mac identisch mit dem eingefrorenen Stand (745132b1…, 1b24d88a…, 6b77f963…).
- **Integrität:**
  - 4680 Zeilen, 936 von 936 Runs (576 Tuning, 360 Bestätigung)
  - keine Duplikate, keine fehlenden und keine unerwarteten Runs
  - 0 nicht-endliche Werte
- **LR-Wahl:** `lr_selection_matches_file = true`, SHA-256 der Steuerdatei 6b23e7f1…. Keine gewählte LR liegt am Rand des Gitters.
- **Reproduktion:** Das Gesamtarchiv (SHA-256 c8262689…) wurde byte-genau übertragen. Die eingefrorene Analyse lief damit in der Sandbox erneut.
  - `verdict.json`, LR-Wahl, Sekundär- und Tuning-Tabelle sind identisch.
  - In den übrigen Tabellen unterscheiden sich nur die informative Spalte `p_scipy_info` (scipy 1.13.1 gegenüber 1.18.1) und die letzte Nachkommastelle bei Gradientennormen.
  - Die maßgeblichen exakten p-Werte und alle Urteile sind identisch.
  - Die Reproduktion liegt in `analysis_reproduktion_scipy118/`, die Rohdaten in `raw/p3a3_workers.tbz` (SHA-256 95e6a1e9…).

## Primärtest: SPreCap-cs gegen SELU (Holm über 4)

| Zelle | SPreCap-cs | SELU | Differenz (pp) | Siege/Niederlagen | Holm p | Urteil |
|---|---|---|---|---|---|---|
| Fashion 50 | 14,92 % | 14,69 % | +0,20 | 4/11 | 0,33 | unklar |
| Fashion 100 | 16,98 % | 17,29 % | −0,12 | 9/6 | 1,0 | unklar |
| MNIST 50 | 4,42 % | 4,31 % | +0,12 | 7/8 | 1,0 | unklar |
| MNIST 100 | 6,27 % | 6,36 % | +0,02 | 7/8 | 1,0 | unklar |

Angegeben ist jeweils der Median des Testfehlers über 15 Seeds. Gerechnet wurde mit dem exakten gepaarten Vorzeichen-Permutationstest.

## Sekundärtests (Holm über 24)

| Vergleich | Fashion 50 | Fashion 100 | MNIST 50 | MNIST 100 |
|---|---|---|---|---|
| cs gegen altes SPreCap | **−1,04, besser** | **−7,23, besser** | **−0,64, besser** | **−6,74, besser** |
| lernbares s gegen cs | −0,02 | −0,06 | **−0,45, besser** | −0,42 (Holm 0,36) |
| cs gegen ELU | +0,20 | +0,36 | **+0,58, schlechter** | +0,53 |
| e03 gegen SELU | +0,16 | −0,74 (Holm 0,12) | −0,37 | **−1,33, besser** |
| e03 gegen cs | −0,28 | −1,01 | −0,44 | **−0,70, besser** |
| altes SPreCap gegen SELU | **+1,21, schlechter** | **+7,13, schlechter** | **+1,03, schlechter** | **+7,04, schlechter** |

Die Werte sind Differenzen der Mediane in Prozentpunkten. Fett markiert ist, was nach Holm signifikant ist.

## χ₁-Dosis-Wirkung (Median Testfehler)

| Zelle | SPreCap (χ₁ 1,1305) | SPreCap-cs (1,0716) | SPreCap-e03 (1,0300) |
|---|---|---|---|
| Fashion 50 | 16,15 % | 14,92 % | 14,57 % |
| Fashion 100 | 24,21 % | 16,98 % | 16,61 % |
| MNIST 50 | 5,13 % | 4,42 % | 4,05 % |
| MNIST 100 | 13,56 % | 6,27 % | 5,28 % |

In allen 4 Zellen sinkt der Fehler streng mit χ₁. Den größten Sprung bringt der Schritt von 1,13 auf 1,07.

Mittlerer Rang in der Bestätigung (1 = am besten):

| Funktion | Rang |
|---|---|
| e03 | 1,5 |
| ELU | 1,75 |
| lernbares s | 3,25 |
| cs | 4,25 |
| SELU | 4,25 |
| altes SPreCap | 6,0 |

## Init-Gradientenprofil: gemessen gegen vorhergesagt

Angegeben ist log10(g₁/g_L), also das Verhältnis der Gradienten von der ersten zur letzten Schicht beim Start. Die Vorhersage lautet (L−1)/2 · log10 χ₁.

| Funktion | Tiefe 50 gemessen | vorhergesagt | Tiefe 100 gemessen | vorhergesagt |
|---|---|---|---|---|
| altes SPreCap | 1,346 | 1,305 | 2,804 | 2,637 |
| SPreCap-cs | 0,703 | 0,736 | 1,540 | 1,487 |
| SELU | 0,699 | 0,736 | 1,558 | 1,487 |
| e03 | 0,342 | 0,315 | 0,572 | 0,635 |

Die Theorie trifft alle Werte auf etwa 0,17 Größenordnungen genau. SPreCap-cs und SELU sind beim Start praktisch nicht zu unterscheiden, so wie es konstruiert war.

## Lernbares s

Gelernt wurde je Schicht s = 1,5 · e^ρ. Über alle Schichten und Seeds:

| Zelle | Minimum | Median | Maximum |
|---|---|---|---|
| Fashion 50 | 1,00 | 1,34 | 1,96 |
| Fashion 100 | 0,97 | 1,28 | 1,94 |
| MNIST 50 | 0,96 | 1,26 | 2,12 |
| MNIST 100 | 0,94 | 1,29 | 1,93 |

Das Netz verkleinert s im Median leicht, von 1,5 auf etwa 1,3, und macht die Begrenzung damit etwas enger. Bestätigt besser als festes s ist das aber nur bei MNIST 50.

## Robustheit gegenüber der Lernrate (Tuning, Median val acc über 4 Seeds)

Das war die Kernfrage: Lernen die begrenzten Varianten bei hohen LRs weiter, wo SELU zusammenbricht?

| Zelle | LR | SELU | ELU | cs | lernbares s | e03 | altes SPreCap |
|---|---|---|---|---|---|---|---|
| Fashion 50 | 5e-4 | 0,66 | 0,72 | 0,11 | 0,10 | 0,15 | 0,14 |
| MNIST 50 | 5e-4 | 0,61 | 0,59 | 0,14 | 0,13 | 0,11 | 0,10 |
| Fashion 100 | 1e-4 | 0,14 | 0,65 | 0,19 | 0,19 | 0,13 | **0,68** |
| MNIST 100 | 1e-4 | 0,19 | 0,18 | 0,17 | 0,50 | 0,20 | **0,86** |

Was die Tabelle zeigt:

- **Bei Tiefe 50 und LR 5e-4** sind SELU und ELU deutlich robuster. Alle PreCap-Varianten fallen auf Zufallsniveau.
- **Bei Tiefe 100 und LR 1e-4** lernt nur das alte SPreCap (χ₁ = 1,13) noch gut. Das wiederholt die explorative Beobachtung aus 3a-2.
- **Es gibt also einen Zielkonflikt:** Ein höheres χ₁ verträgt bei großer Tiefe eine höhere LR, erreicht aber beim Optimum einen schlechteren Wert. Ein kleineres χ₁ ist beim Optimum besser, verträgt aber weniger LR.
- **Breite des LR-Fensters** (mittlere Zahl der LRs nahe am Optimum): e03 2,5, ELU 1,75, cs und lernbares s 1,5, SELU und altes SPreCap 1,25. Diese Kennzahl ist explorativ.

Dieser Teil ist explorativ, denn die Tuning-Daten haben nur 4 Seeds.

## Erwartungen (vorregistriert) und Ergebnis

| Erwartung | Wahrscheinlichkeit | Eingetreten |
|---|---|---|
| Gradientenprofil von cs bei 1,49 ± 0,15 | 0,8 | ja (1,54) |
| cs besser als altes SPreCap | 0,65 | ja (4/4) |
| cs bewährt: nein | 0,6 | nein, es hat sich bewährt |
| Nachteil von cs bestätigt: nein | 0,55 | ja |
| lernbares s nicht anders als cs | 0,6 | überwiegend (3/4 unklar, 1/4 besser) |
| ELU am besten | 0,6 | nein, knapp: e03 Rang 1,5 vor ELU 1,75 |
| Nachteil von altem SPreCap gegen SELU wiederholt | 0,8 | ja (4/4) |

4 von 7 Erwartungen sind klar eingetreten, eine weitere überwiegend. Die beiden Fehleinschätzungen fielen zugunsten von SPreCap aus.

## Einordnung

- **Was belegt ist:** SPreCap-cs ist bei Tiefe 50 und 100 ohne Normierung gleichauf mit SELU. Gleichzeitig ist es beschränkt, die Ausgabe liegt in [−1,76, 3,39], was SELU nicht ist. Die χ₁-Theorie sagt das Verhalten in der Tiefe quantitativ voraus.
- **Was angedeutet ist:** Ein kleineres χ₁ (e03) könnte SELU in großer Tiefe schlagen. Das ist nur in einer Zelle bestätigt und sollte in 3a-4 primär geprüft werden.
- **Was nicht belegt ist:** ein Robustheitsvorteil bei hohen Lernraten durch die Begrenzung. Bei Tiefe 50 ist es umgekehrt.
- **Einschränkungen:** nur MLPs mit Breite 256, nur MNIST und FashionMNIST, nur Adam, 1720 Schritte. Ob das auf CNNs und Transformer übertragbar ist, ist offen.

## Folgen für den Fahrplan

- Nach der vorregistrierten Regel ist v0.1 bewährt. Damit kann v0.1 mit Kernels gesichert werden.
- **Für 3a-4 schlage ich vor:**
  - e03 (χ₁ = 1,03) als primären Kandidaten gegen SELU und ELU aufzunehmen, auch mit kritischer Init
  - die LR-Robustheit primär zu messen
