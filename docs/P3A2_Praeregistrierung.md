# Phase 3a-2 – Präregistrierung: Nachlauf mit nach unten offenem Lernraten-Raster

Stand: 26.10.2026, festgelegt vor dem Start. Code-Hashes in `code/phase3/p3a2_low_lr/meta/frozen_sha256.txt`. Spätere Änderungen werden als Abweichung dokumentiert.

## Anlass

In Phase 3a haben alle beschränkten Kandidaten in 6 von 6 Zellen die kleinste Lernrate 1e-4 gewählt. Bei 50 und 100 Schichten fiel die Genauigkeit schon bei 3e-4 auf Zufallsniveau. Die Kandidaten waren also wahrscheinlich untertuned. Außerdem ließ die Init-Regel SiLU und GELU explodieren (siehe `analysis/P3A_Auswertung.md`). 3a-2 behebt beides. Die Ergebnisse von 3a bleiben unverändert gültig. 3a-2 ist ein neuer, eigenständiger Versuch mit neuen Seeds.

## Hauptfrage

Ist SPreCap bei 50 und 100 Schichten ohne Normalisierung besser als SELU, wenn beide fair getunt sind?

## Aufbau (fest)

| Punkt | Festlegung |
|---|---|
| Daten, Aufteilung, Netz | wie 3a (MNIST, FashionMNIST; letzte 5000 Trainingsbilder = Validierung; Breite 256; keine Norm, kein Residual) |
| Tiefen | 50, 100 (4 Zellen) |
| Aktivierungen | spreccap, selu, precap0, relu, elu, silu_he, gelu_he |
| Init | wie 3a (g = 1/sqrt(E[f(Z)²])), außer silu_he und gelu_he mit g = √2 (üblicher Kaiming-Faktor) |
| Training | Adam, Batch 128, **1720 Schritte** (ca. 4 Epochen), Checkpoints 0/430/860/1290/1720 |
| LR-Raster | **1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4** |
| Tuning | alle LRs × Seeds 41000–41003 (4 Seeds, 672 Runs) |
| LR-Wahl | wie 3a: höchste mediane val_acc bei Schritt 1720, divergiert = 0,1, Gleichstand → kleinere LR; Datei `meta/lr_selection.json` vor der Bestätigung |
| Bestätigung | gewählte LR × Seeds 41010–41024 (15 Seeds, 420 Runs) |
| Neu protokolliert | Gradientennorm der ersten, mittleren und letzten verdeckten Schicht bei Init (feste 256 Trainingsbilder, ohne Update) und an jedem Checkpoint |

## Bestätigende Analyse (fest)

- Testfehler bei Schritt 1720, gepaart über den Seed, zweiseitiger Wilcoxon-Test (zsplit).
- **Primär:** SPreCap gegen SELU in 4 Zellen, Holm über 4 Tests, α = 0,05. Ein Vorteil gilt als bestätigt, wenn SPreCap in mindestens 3 von 4 Zellen signifikant besser ist und in keiner schlechter. Ein Nachteil gilt spiegelbildlich.
- **Sekundär:** {spreccap, precap0} × {selu, relu, elu, silu_he, gelu_he} × 4 Zellen = 40 Tests, eigene Holm-Korrektur, gleiche Regel. Nur ergänzend.

## Deskriptiv (ohne Entscheidungsregel)

- LR am unteren (1e-5) oder oberen Rand (5e-4)
- Zahl der Zellen, in denen eine Funktion überhaupt lernt (beste mediane val_acc im Tuning > 50 %)
- Gradientenprofil bei Init: log10(‖∇W erste Schicht‖ / ‖∇W letzte Schicht‖), negativ = verschwindende Gradienten zum Eingang hin
- Divergenzraten, mittlerer Rang, Varianz der Aktivierungen

## Erwartungen vor dem Start

| Frage | Erwartung | Konfidenz |
|---|---|---|
| SPreCap gegen SELU (primär) | kein bestätigter Unterschied | 0,55 |
| SPreCap gegen SELU bei Tiefe 100 | SPreCap eher besser | 0,5 |
| PreCap0 lernt bei Tiefe 100 mit kleiner LR | nein, höchstens schwach | 0,6 |
| Gradientenprofil PreCap0 bei Tiefe 100 | deutlich negativer als bei SPreCap und SELU | 0,7 |
| Gewählte LRs liegen im Raster, nicht am Rand | ja, für die meisten Funktionen | 0,7 |
| SiLU/GELU mit He-Init lernen bei Tiefe 100 deutlich besser als in 3a | ja | 0,6 |

## Ablauf

`launch_p3a2.sh` wie bei 3a: caffeinate, Lock, Hash-Prüfung gegen den eingefrorenen Stand, Selbsttest, 7 Worker Tuning, Integrität, LR-Wahl, 7 Worker Bestätigung, Integrität, Analyse, `done.txt`. Geschätzte Laufzeit 3,5–4 Stunden.
