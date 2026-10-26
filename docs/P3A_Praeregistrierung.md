# Phase 3a – Präregistrierung: tiefe MLPs ohne Normalisierung

Stand: 26.10.2026, festgelegt vor dem Start des Hauptlaufs. Die Hashes von Code und Analyse stehen in `code/phase3/p3a_deep_mlp/meta/frozen_sha256.txt`. Jede spätere Änderung wird als Abweichung dokumentiert.

## Fragestellung

Phase 1 hat gezeigt, dass PreCap beschränkt ist und dadurch die Signalvarianz in tiefen Netzen ohne Normalisierung begrenzt bleibt. SPreCap hat einen Fixpunkt (Mittelwert 0, Varianz 1) wie SELU. Phase 3a prüft, ob daraus in echten Daten ein messbarer Vorteil entsteht. Getestet werden tiefe MLPs (20, 50 und 100 Schichten, Breite 256) ohne Normalisierung und ohne Residualverbindungen auf MNIST und FashionMNIST. Das ergibt 6 Zellen.

Hauptfragen:

1. Ist SPreCap besser als SELU, also das passende selbstnormierende Gegenstück?
2. Ist PreCap0 (γ=0) besser als SiLU und GELU, also als die Standardfunktionen, denen PreCap nahe 0 ähnelt?

## Aufbau (fest)

| Punkt | Festlegung |
|---|---|
| Daten | MNIST, FashionMNIST (IDX-Originaldateien, SHA-256 in `data/sha256.txt`) |
| Aufteilung | Die letzten 5000 Trainingsbilder sind die Validierung. Die Testmenge (10.000) wird nur für die bestätigende Auswertung genutzt. Standardisierung mit Mittelwert und Standardabweichung des Trainingsteils. |
| Netz | 784 → [256]×Tiefe → 10, Tiefe 20/50/100, keine Normalisierung, kein Residual, kein Dropout |
| Init | W ~ N(0, g²/fan_in), Bias 0, g = 1/sqrt(E[f(Z)²]) für Z~N(0,1) je Aktivierung (Gauss-Hermite). Für ReLU ergibt das He, für SELU und SPreCap LeCun. Ausgabeschicht g=1. |
| Training | Adam, Batch 128, 860 Schritte (ca. 2 Epochen), Kreuzentropie, CPU mit je 1 Thread |
| Aktivierungen | relu, gelu, silu, elu, selu, tanh, precap0, spreccap (λ 0,80676869644, α 2,39695130922, b 1), spreccap_e03 (λ 1,985552225, α 1,608136395, b 0,4118896262) |
| LR-Raster | 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2 |
| Stufe 1: Tuning | alle 6 LRs × Seeds 40000–40004 (1620 Runs) |
| LR-Wahl | je (Dataset, Tiefe, Aktivierung) die LR mit der höchsten medianen Validierungsgenauigkeit bei Schritt 860 über die Tuning-Seeds. Divergierte Runs zählen mit 0,1. Bei Gleichstand gilt die kleinere LR. Die Wahl wird in `meta/lr_selection.json` gespeichert, bevor Stufe 2 startet. |
| Stufe 2: Bestätigung | gewählte LR × Seeds 40010–40024 (15 Seeds, 810 Runs) |
| Divergenz | Ein nicht-endlicher Loss, Gradient oder Eval-Wert beendet den Run; er zählt als Testfehler 0,9 (Zufallsniveau). |

## Bestätigende Analyse (fest)

- Kennzahl: Testfehler bei Schritt 860 der 15 Bestätigungs-Seeds, gepaart über den Seed. Gleicher Seed bedeutet gleiche Batch-Reihenfolge; die Init-Ziehung ist gleich, die Skalierung hängt von der Aktivierung ab.
- Test: zweiseitiger Wilcoxon-Vorzeichen-Rang-Test (zero_method="zsplit") je Zelle.
- Primäre Familie: 3 Paare (spreccap/selu, precap0/silu, precap0/gelu) × 6 Zellen = 18 Tests, Holm-Korrektur, α = 0,05.
- Ein Vorteil gilt als bestätigt, wenn der Kandidat in mindestens 4 von 6 Zellen signifikant besser ist (Median der Differenz < 0) und in keiner Zelle signifikant schlechter. Ein Nachteil gilt spiegelbildlich.
- Sekundäre Familie: alle 3 Kandidaten × 6 Referenzen × 6 Zellen = 108 Tests, eigene Holm-Korrektur, gleiche Regel. Diese Familie ist ergänzend und kann die Hauptaussage nicht ersetzen.
- Begründung für 15 Seeds: Mit 10 Seeds ist der kleinste exakte Wilcoxon-p-Wert 0,00195. Nach Holm über 18 oder 108 Tests könnte nie Signifikanz erreicht werden. Mit 15 Seeds liegt der kleinste p-Wert bei 6,1e-5. Das ist im Testlauf mit Kunstdaten aufgefallen und wurde vor dem Start korrigiert.

## Sekundär und deskriptiv (fest, ohne Entscheidungsregel)

- LR-Fenster: Zahl der LRs, deren mediane Validierungsgenauigkeit (Tuning) höchstens 0,5 Prozentpunkte unter dem eigenen Optimum liegt.
- Divergenzraten (Tuning und Bestätigung).
- Wie oft die gewählte LR am Rasterrand liegt (3e-2 oder 1e-4).
- Mittlerer Rang des medianen Testfehlers über die 6 Zellen.
- Varianz der Aktivierungen in erster, mittlerer und letzter Schicht über die Checkpoints (Mechanismus).

## Erwartungen vor dem Start

| Frage | Erwartung | Konfidenz |
|---|---|---|
| SPreCap gegen SELU | kein bestätigter Unterschied | 0,6 |
| PreCap0 gegen SiLU/GELU bei Tiefe 20 | kein Unterschied oder leicht schlechter | 0,6 |
| PreCap0 gegen SiLU/GELU bei Tiefe 100 | eher besser, weil SiLU/GELU ohne Normalisierung stärker leiden | 0,4 |
| Vorteil für PreCap0 bestätigt (≥ 4/6) | nein | 0,7 |
| spreccap_e03 | schwächer als spreccap, weil fast linear | 0,6 |
| tanh | bei Tiefe 50/100 deutlich schlechter | 0,6 |

## Ablauf und Integrität

`launch_p3a.sh`: caffeinate, Lock-Datei, 7 Worker für das Tuning, Integritätsprüfung (keine Lücken, keine Duplikate), LR-Wahl, 7 Worker für die Bestätigung, erneute Integritätsprüfung, `analyze_p3a.py`, danach `done.txt`. Die Wiederaufnahme nach einem Abbruch übernimmt fertige Runs, ohne sie neu zu rechnen.
