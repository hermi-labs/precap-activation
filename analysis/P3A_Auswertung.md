# Phase 3a – Auswertung: tiefe MLPs ohne Normalisierung

Stand: 26.10.2026, 11:10. Grundlage: `docs/P3A_Praeregistrierung.md` und die eingefrorene Analyse `analyze_p3a.py`. Code-Hashes vor und nach dem Lauf identisch.

## Kurzfassung

- **Primäres Ergebnis:** Keine der drei Hauptthesen ist bestätigt. SPreCap ist nicht robust besser als SELU, PreCap0 nicht robust besser als SiLU oder GELU. Die Ergebnisse hängen stark von der Tiefe ab.
- **Überraschung:** PreCap0 lernt bei 100 Schichten überhaupt nicht (Testfehler 90 %, also Zufall), obwohl das Signal vorwärts stabil bleibt und kein Run divergiert. Beschränkt heißt nicht trainierbar.
- **Lichtblick:** SPreCap ist bei 100 Schichten auf FashionMNIST die beste aller 9 Funktionen (28,1 % Fehler gegen 32,0 % bei ReLU und 37,1 % bei SELU) und auf MNIST die zweitbeste. Bei 50 Schichten ist es dagegen signifikant schlechter als SELU.
- **Wichtigste Einschränkung:** Die gewählte Lernrate liegt bei allen drei Kandidaten in 6 von 6 Zellen am unteren Rasterrand (1e-4). Bei 50 und 100 Schichten fällt die Genauigkeit schon bei 3e-4 auf Zufallsniveau. Das Optimum liegt also vermutlich unter dem Raster, die Kandidaten sind wahrscheinlich untertuned. Das gilt abgeschwächt auch für SELU und ELU (je 5 von 6).
- **Konfidenz:** "PreCap0 hat in tiefen MLPs ohne Norm keinen robusten Vorteil" etwa 0,75. "SPreCap ist bei extremer Tiefe mit SELU konkurrenzfähig" etwa 0,5. Klären kann das nur ein Nachlauf mit tieferem Lernraten-Raster.

## Integrität

| Prüfung | Ergebnis |
|---|---|
| Runs | 2430 / 2430 (Tuning 1620, Bestätigung 810) |
| Zeilen | 9537, keine Duplikate, keine Lücken |
| Nicht-endliche Runs | 106, alle im Tuning bei zu hohen LRs, gewertet als Zufallsniveau |
| Code-Hashes | `run_p3a.py`, `analyze_p3a.py`, `launch_p3a.sh` unverändert gegenüber `meta/frozen_sha256.txt` |
| LR-Wahl | in `meta/lr_selection.json` gespeichert vor der Bestätigung (sha256 85908ac0…), von der Analyse unabhängig neu berechnet und identisch |
| Holm-Korrektur | in der Sandbox unabhängig nachgerechnet, identisch |
| Rohdaten | auf dem Mac in `Phase3/p3a_deep_mlp/workers/w0–w6.csv` (2,7 MB), Hashes in `meta/workers_sha256.txt` |

## Primäre Tests (18 Tests, Holm, α = 0,05, n = 15 Seeds)

Testfehler als Median, Differenz in Prozentpunkten (negativ = Kandidat besser).

| Zelle | SPreCap / SELU | PreCap0 / SiLU | PreCap0 / GELU |
|---|---|---|---|
| MNIST 20 | 4,56 / 4,30 (+0,37, unklar) | 4,34 / 4,29 (−0,03, unklar) | 4,34 / 4,36 (+0,06, unklar) |
| MNIST 50 | 6,64 / 5,54 (+1,28, **schlechter**) | 6,24 / 17,41 (−11,2, **besser**) | 6,24 / 9,47 (−3,2, **besser**) |
| MNIST 100 | 19,4 / 14,1 (+4,8, unklar) | 89,9 / 85,2 (+4,6, **schlechter**) | 89,9 / 70,1 (+17,4, **schlechter**) |
| Fashion 20 | 15,1 / 14,9 (+0,18, unklar) | 14,6 / 15,5 (−0,61, **besser**) | 14,6 / 14,7 (−0,19, unklar) |
| Fashion 50 | 17,2 / 16,3 (+1,16, **schlechter**) | 17,0 / 23,7 (−6,1, **besser**) | 17,0 / 21,3 (−4,3, **besser**) |
| Fashion 100 | 28,1 / 37,1 (−11,7, **besser**) | 90,0 / 76,0 (+12,0, **schlechter**) | 90,0 / 59,2 (+30,8, **schlechter**) |
| **Urteil** | 1 besser, 2 schlechter, 3 unklar: **nicht bestätigt** | 3 besser, 2 schlechter: **nicht bestätigt** | 2 besser, 2 schlechter: **nicht bestätigt** |

Nötig wären mindestens 4 von 6 Zellen besser und keine schlechter.

## Medianer Testfehler aller Funktionen (Bestätigung, %)

| Zelle | ELU | GELU | PreCap0 | ReLU | SELU | SiLU | SPreCap | SPreCap-ε | tanh |
|---|---|---|---|---|---|---|---|---|---|
| MNIST 20 | 3,91 | 4,36 | 4,34 | 4,24 | 4,30 | 4,29 | 4,56 | **3,84** | 4,81 |
| MNIST 50 | **4,89** | 9,47 | 6,24 | 6,66 | 5,54 | 17,41 | 6,64 | 5,11 | 9,23 |
| MNIST 100 | 66,0 | 70,1 | 89,9 | 29,1 | **14,1** | 85,2 | 19,4 | 25,5 | 74,9 |
| Fashion 20 | **14,52** | 14,74 | 14,60 | 14,73 | 14,89 | 15,53 | 15,09 | 14,84 | 15,05 |
| Fashion 50 | **15,60** | 21,32 | 16,95 | 17,22 | 16,25 | 23,68 | 17,19 | 16,55 | 21,36 |
| Fashion 100 | 53,2 | 59,2 | 90,0 | 32,0 | 37,1 | 76,0 | **28,1** | 77,1 | 52,1 |

Mittlerer Rang über die 6 Zellen: ELU 2,5, SELU 3,3, SPreCap-ε 3,7, ReLU 4,0, SPreCap 4,8, PreCap0 5,7, GELU 6,3, tanh 7,0, SiLU 7,7.

## Sekundäre Familie (108 Tests, deskriptiv)

38 besser, 14 schlechter, 56 unklar. Das Kriterium (mindestens 4 von 6 besser, keine schlechter) erfüllen:
- SPreCap gegen GELU, SiLU und tanh
- SPreCap-ε gegen tanh

Gegen SELU, ELU und ReLU erfüllt es kein Kandidat. PreCap0 erfüllt es gegen keine Referenz.

Vorsicht bei SiLU und GELU: Die präregistrierte Init-Regel g = 1/sqrt(E[f(Z)²]) normiert das zweite Moment und nicht die Varianz. Bei SiLU und GELU explodiert dadurch das Signal schon bei der Initialisierung (SiLU, 100 Schichten: Varianz der letzten Schicht ≈ 8·10¹³, GELU ≈ 2·10⁶). Siege gegen SiLU und GELU in der Tiefe messen deshalb zum großen Teil diese Init-Regel, nicht die Funktion selbst. Das betrifft auch die primären Tests PreCap0/SiLU und PreCap0/GELU bei 50 Schichten.

## Mechanismus (Median der Bestätigungsläufe, 100 Schichten)

| Funktion | Var. letzte Schicht bei Init | Train-Loss bei 860 | Gradientennorm bei 860 |
|---|---|---|---|
| PreCap0 | 0,89 | 2,31 (Zufall) | 2,4 |
| SPreCap | 1,00 | 0,74 | 23,4 |
| SELU | 0,99 | 0,81 | 11,0 |
| ReLU | 0,25 | 0,86 | 14,3 |
| SiLU | 8·10¹³ | 33,1 | 53.420 |
| GELU | 2·10⁶ | 1,75 | 26,3 |

- PreCap0 hält das Signal vorwärts stabil (Varianz um 0,9, maximal 3,04, also an der Schranke), lernt aber nicht. Das passt zur geordneten Phase aus der Theorie tiefer Netze: Der mittlere quadrierte Jacobi-Faktor liegt unter 1, und Gradienten verschwinden über 100 Schichten exponentiell. Diese Deutung ist vorläufig, denn gemessen wurde nur die Gesamt-Gradientennorm, nicht der Verlauf pro Schicht.
- SPreCap ist auf den Fixpunkt (0, 1) geeicht und behält kräftige Gradienten. Das stützt die Phase-1-Rechnung (Gradientenverstärkung 10^2,70) qualitativ.

## Sekundäre Kennzahlen

| Kennzahl | ELU | GELU | PreCap0 | ReLU | SELU | SiLU | SPreCap | SPreCap-ε | tanh |
|---|---|---|---|---|---|---|---|---|---|
| Divergenzrate Tuning | 13,3 % | 10,6 % | **0 %** | 9,4 % | 12,2 % | 13,3 % | **0 %** | **0 %** | **0 %** |
| LR am unteren Rand (von 6) | 5 | 0 | 6 | 4 | 5 | 1 | 6 | 6 | 3 |
| LR am oberen Rand (von 6) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| LR-Fenster (± 0,5 pp) | 1,17 | 1,17 | 2,67* | 1,33 | 1,17 | 1,17 | 1,00 | 1,17 | 1,33 |

\* Artefakt: PreCap0 liegt bei 100 Schichten bei allen LRs auf Zufallsniveau. Dadurch zählen dort alle 6 LRs als "nahe am Optimum".

- Beschränkte Funktionen (PreCap0, SPreCap, SPreCap-ε, tanh) divergieren nie numerisch. In diesem engen Sinn ist die Stabilitätsthese bestätigt. Das Versagen verschiebt sich bei PreCap0 aber von "NaN" zu "lernt nicht".

## Abgleich mit den Erwartungen vor dem Start

| Erwartung | Eingetreten? |
|---|---|
| SPreCap ≈ SELU, kein bestätigter Unterschied (0,6) | ja, gemischt: schlechter bei 50, besser bei Fashion 100 |
| PreCap0 bei Tiefe 20 nicht besser als SiLU/GELU (0,6) | weitgehend ja (einmal knapp besser gegen SiLU) |
| PreCap0 bei Tiefe 100 eher besser (0,4) | nein, das Gegenteil: lernt gar nicht |
| Kein bestätigter Vorteil für PreCap0 (0,7) | ja |
| SPreCap-ε schwächer als SPreCap (0,6) | gemischt: besser bei 20 und 50, bei 100 deutlich schlechter (MNIST 25,5 gegen 19,4 %, Fashion 77,1 gegen 28,1 %) |
| tanh bei 50/100 deutlich schlechter (0,6) | ja |

## Einschränkungen

1. **Lernraten-Raster unten nicht offen.** Bei 3D-V1 war das Problem der obere Rand, hier ist es der untere. Bei 50 und 100 Schichten ist für die beschränkten Funktionen 1e-4 die einzige brauchbare LR (z. B. SPreCap Fashion 50: 83,5 % bei 1e-4, 10,4 % bei 3e-4). Die Urteile "SPreCap schlechter als SELU bei 50" und "PreCap0 lernt bei 100 nicht" können sich bei kleineren LRs ändern.
2. **Init-Regel** benachteiligt SiLU und GELU stark (siehe oben). Sie ist präregistriert, aber nicht die übliche Praxis.
3. **Kurzes Training** mit 860 Schritten (ca. 2 Epochen). Bei 100 Schichten ist noch kein Modell auskonvergiert.
4. Nur MLPs auf zwei einfachen Bilddatensätzen.

## Konsequenz für den Fahrplan

- PreCap0 ist als Aktivierung für tiefe Netze ohne Norm nicht empfehlenswert. Seine Stärke (Beschränktheit) verhindert NaN, aber nicht das Verschwinden der Gradienten.
- Interessant bleibt nur SPreCap bei extremer Tiefe. Vor Phase 3b schlage ich einen kleinen, eigens präregistrierten Nachlauf **3a-2** vor:
  - nur 50 und 100 Schichten
  - LR-Raster nach unten offen (1e-5, 2e-5, 5e-5, 1e-4, 2e-4)
  - Hauptvergleich SPreCap gegen SELU, dazu PreCap0, ReLU und ELU
  - Gradientennorm pro Schicht protokollieren
  - Init für SiLU/GELU als Varianz-Normierung (zusätzlicher Arm)
- Fällt 3a-2 für SPreCap positiv aus, folgen 3b (CIFAR-10) und 3c (Transformer mit LayerNorm). Sonst wird das Paper als präregistrierte Negativ-Studie geschrieben.
