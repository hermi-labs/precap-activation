# 3D-V1 – Ausführliche Auswertung

Stand: 26.10.2026, 06:40 Uhr. Lauf vom 25.10.2026 20:38 bis 26.10.2026 02:18 (M1 Pro, 7 Worker).
Präregistrierung: `code/3D/v1_transfer3d/docs/V1_3D_Praeregistrierung.md` (vor dem Start eingefroren).

## 1. Kurzfassung

- **Hauptfrage (präregistriert):** Wenn jede Aktivierung ihre eigene Lernrate bekommt, zeigt weder PreCap noch CapExp in 3D einen bestätigten Vorteil gegenüber SiLU oder SiLU×2. Das Kriterium (mindestens 6 von 10 Zellen signifikant besser) wird in keiner der 8 Kombinationen erreicht.
- **Ohne Normalisierung** sind PreCap und CapExp im Median schlechter. Ihr Test-MSE ist rund 1,3- bis 1,4-mal so hoch wie bei SiLU bzw. SiLU×2. Signifikant schlechter sind sie bei 6 von 40 Tests, alle bei der Aufgabe gyroid.
- **Mit LayerNorm** liegen alle vier Aktivierungen etwa gleichauf (Median-Verhältnis 0,93 bis 1,07). Genau ein Test ist signifikant besser: CapExp gegen SiLU bei torus_sdf, Tiefe 8, mit MSE-Verhältnis 0,34.
- **Lernraten-Robustheit (explorativ):** Der Hinweis von gestern Abend aus dem Zwischenstand hat sich nicht verallgemeinert. PreCap hat kein breiteres gutes Lernraten-Fenster als SiLU. Mit LayerNorm ist SiLU sogar am robustesten.
- **Stabilität:** 0 von 7200 Läufen sind numerisch entgleist, auch CapExp bei LR 1e-2 nicht.

Konfidenz in die Gesamtaussage „kein Vorteil in diesem Aufbau“: etwa 0,85.

## 2. Integrität

| Prüfung | Ergebnis |
|---|---|
| Läufe | 7200 / 7200 |
| Checkpoint-Zeilen | 57.600 / 57.600, alle Schlüssel eindeutig |
| Nicht-endliche Läufe | 0 |
| SHA-256 `run_3d_v1.py`, `analyze_3d_v1.py`, Präregistrierung | identisch mit `meta/frozen_sha256.txt` vom Start |
| Auswertung | automatisch um 02:18 mit dem eingefrorenen Skript |

## 3. Präregistrierte Auswertung

Hier sind 80 gepaarte Wilcoxon-Tests auf 20 Bestätigungs-Seeds ausgewertet, mit Holm-Korrektur und α = 0,05. Das Verhältnis ist der Median von MSE(Kandidat) / MSE(Referenz). Werte unter 1 bedeuten, dass der Kandidat besser ist.

### Kriterium je Kombination

| Kandidat | Norm | Referenz | besser | schlechter | unklar | Median-Verhältnis | Vorteil bestätigt |
|---|---|---|---|---|---|---|---|
| PreCap | none | SiLU | 0 | 2 | 8 | 1,44 | nein |
| PreCap | none | SiLU×2 | 0 | 2 | 8 | 1,41 | nein |
| PreCap | LN | SiLU | 0 | 0 | 10 | 1,03 | nein |
| PreCap | LN | SiLU×2 | 0 | 0 | 10 | 0,97 | nein |
| CapExp | none | SiLU | 0 | 1 | 9 | 1,31 | nein |
| CapExp | none | SiLU×2 | 0 | 1 | 9 | 1,30 | nein |
| CapExp | LN | SiLU | 1 | 0 | 9 | 0,93 | nein |
| CapExp | LN | SiLU×2 | 0 | 0 | 10 | 1,07 | nein |

### Signifikante Zellen

| Aufgabe | Tiefe | Norm | Kandidat | Referenz | Verhältnis | 90-%-KI | Holm-p | Urteil |
|---|---|---|---|---|---|---|---|---|
| gyroid | 8 | none | PreCap | SiLU | 1,92 | 1,69–2,13 | 0,0002 | schlechter |
| gyroid | 8 | none | PreCap | SiLU×2 | 1,85 | 1,70–1,98 | 0,0005 | schlechter |
| gyroid | 16 | none | PreCap | SiLU | 1,77 | 1,54–2,26 | 0,0007 | schlechter |
| gyroid | 16 | none | PreCap | SiLU×2 | 1,42 | 1,31–1,61 | 0,0007 | schlechter |
| gyroid | 16 | none | CapExp | SiLU | 1,79 | 1,46–1,94 | 0,0047 | schlechter |
| gyroid | 16 | none | CapExp | SiLU×2 | 1,61 | 1,31–1,68 | 0,036 | schlechter |
| torus_sdf | 8 | LN | CapExp | SiLU | 0,34 | 0,27–0,44 | 0,0028 | besser |

Die Erwartung vor dem Lauf war „kein klarer Vorteil von PreCap gegen SiLU×2“ mit etwa 0,7. Sie ist eingetroffen. Die Erwartung „CapExp besser als PreCap“ mit etwa 0,35 ist teilweise eingetroffen: Mit LayerNorm hat CapExp den besten mittleren Rang (siehe unten), signifikant ist das aber nicht.

## 4. Explorativ: Lernraten-Robustheit

Diese Auswertung war nicht im Voraus festgelegt. Sie nutzt alle 30 Seeds (Median des log10-Test-MSE je Zelle und Lernrate) und ist deshalb nur ein Hinweis, kein Beweis. Skript: `code/3D/v1_transfer3d/analysis/explorativ_lr_robustheit.py`.

### Median über die 10 Zellen (Aufgabe × Tiefe)

Die Spreizung ist der Abstand zwischen bester und schlechtester Lernrate. Der Regret ist der Abstand zum eigenen Optimum, jeweils in log10-MSE. Ein Regret von 0,3 bedeutet etwa doppelten Fehler.

| Norm | Aktivierung | Bestwert | Spreizung | Regret 1e-3 | Regret 3e-3 | Regret 1e-2 |
|---|---|---|---|---|---|---|
| none | PreCap | −3,89 | 0,99 | 0,99 | 0,00 | 0,02 |
| none | CapExp | −3,88 | 0,98 | 0,98 | 0,00 | 0,07 |
| none | SiLU | −4,05 | 1,41 | 1,41 | 0,36 | 0,00 |
| none | SiLU×2 | −4,02 | 1,26 | 1,26 | 0,03 | 0,00 |
| LN | PreCap | −3,91 | 0,18 | 0,04 | 0,00 | 0,18 |
| LN | CapExp | −3,96 | 0,21 | 0,05 | 0,01 | 0,21 |
| LN | SiLU | −3,85 | 0,08 | 0,00 | 0,01 | 0,04 |
| LN | SiLU×2 | −3,88 | 0,12 | 0,03 | 0,01 | 0,11 |

Mittlere Anzahl Lernraten, die höchstens 10 % schlechter als das eigene Optimum sind (von 3):

| Norm | PreCap | CapExp | SiLU | SiLU×2 |
|---|---|---|---|---|
| none | 1,3 | 1,2 | 1,0 | 1,1 |
| LN | 1,4 | 1,3 | 1,9 | 1,3 |

Mittlerer Rang des Bestwerts (1 = beste der vier):

| Norm | PreCap | CapExp | SiLU | SiLU×2 |
|---|---|---|---|---|
| none | 3,8 | 3,1 | 1,6 | 1,5 |
| LN | 2,7 | 1,8 | 2,8 | 2,7 |

Gewählte Lernraten auf den Tuning-Seeds:

| Norm | Aktivierung | 1e-3 | 3e-3 | 1e-2 |
|---|---|---|---|---|
| none | PreCap | 0 | 6 | 4 |
| none | CapExp | 2 | 6 | 2 |
| none | SiLU | 0 | 1 | 9 |
| none | SiLU×2 | 0 | 7 | 3 |
| LN | PreCap | 2 | 8 | 0 |
| LN | CapExp | 5 | 4 | 1 |
| LN | SiLU | 7 | 1 | 2 |
| LN | SiLU×2 | 3 | 6 | 1 |

### Deutung

- Ohne Normalisierung ist PreCap etwas weniger empfindlich, wenn die Lernrate zu klein ist: Spreizung 0,99 gegen 1,41. Der Grund ist einfach: PreCap hat sein Optimum schon bei 3e-3, SiLU erst bei 1e-2. Im Bereich der jeweils guten Lernraten ist der Unterschied klein (1,3 gegen 1,0 gute Lernraten). Und bei der jeweils besten Lernrate ist SiLU deutlich besser.
- SiLU ohne Normalisierung wählt in 9 von 10 Zellen den oberen Rasterrand 1e-2. Mit einem größeren Raster würde SiLU vermutlich noch besser abschneiden. Der Abstand zu PreCap ist hier also eher unter- als überschätzt.
- Der vorläufige Hinweis von gestern (smooth_bowl3d, Tiefe 8: Spreizung 0,19 gegen 0,72) war ein Einzelfall und hält über alle Aufgaben nicht.

### Explorativ: Wechselwirkung mit LayerNorm

LayerNorm verbessert PreCap und CapExp relativ zu SiLU deutlich. Die mediane Änderung der log10-Differenz durch LayerNorm liegt bei −0,22 (PreCap gegen SiLU, 10 von 10 Zellen negativ) bzw. −0,20 (CapExp gegen SiLU, 9 von 10). Mit LayerNorm liegen deutlich mehr Voraktivierungen über 0,5: im Mittel 46 % gegenüber 16 % ohne Norm. Das passt zu der Vermutung, dass PreCap und CapExp ihre Form erst bei größeren Eingaben ausspielen. Ein kausaler Beleg ist das nicht. Der Rangkorrelationstest zwischen diesem Anteil und dem Abstand PreCap gegen SiLU×2 ist nicht signifikant (ρ = −0,28, p = 0,22, n = 20).

Auffällig ist gyroid, die Aufgabe mit den größten Voraktivierungen (p99 ≈ 2,7–3,2 ohne Norm). Dort verliert PreCap ohne Normalisierung am stärksten. Das passt zur Sättigung des tanh bei etwa 3,08.

## 5. Was daraus folgt

1. Ein allgemeiner Genauigkeitsvorteil von PreCap ist in den synthetischen Aufgaben (1D, 2D, 3D) mit fairer Lernrate nicht nachweisbar. Die frühere Wahrscheinlichkeit von etwa 0,25 für einen allgemeinen Vorteil senke ich auf etwa 0,1.
2. Die Robustheits-These (Erwartung vorher etwa 0,4) ist geschwächt. Ohne Norm gibt es einen schwachen Vorteil nur bei zu kleiner Lernrate, mit Norm keinen. Neue Schätzung: etwa 0,2.
3. Was bleibt: Mit LayerNorm sind PreCap und CapExp konkurrenzfähig und nie instabil. CapExp hat mit LayerNorm den besten mittleren Rang (1,8), und bei torus_sdf gibt es eine einzelne starke Zelle. Das ist ein Ansatzpunkt, aber kein Ergebnis.
4. Für Phase 3 des Fahrplans heißt das:
   - Das Lernraten-Raster muss nach oben offen sein (bis mindestens 3e-2), sonst wird SiLU benachteiligt.
   - LayerNorm bzw. Pre-LN-Netze sind der interessantere Fall. Gerade ein Mini-Transformer ist dafür der passende Test.
   - Die Varianten ohne tanh-Sättigung (CapExp, SPreCap mit größerem Sättigungswert) sollten mitlaufen.

## 6. Einschränkungen

- Synthetische 3D-Regressionsaufgaben, Residual-MLP mit Breite 64, nur 3000 Adam-Schritte. Echte Netze können sich anders verhalten.
- Nur 3 Lernraten. SiLU stößt ohne Norm an den Rand.
- Die Auswertung der Lernraten-Robustheit ist explorativ und nutzt auch die Tuning-Seeds.
- GELU wurde aus Zeitgründen weggelassen. In V18 verhielt es sich wie SiLU.

## 7. Dateien

- `code/3D/v1_transfer3d/analysis/V1_3D_Auswertung.md`: automatische Gesamttabelle aller 80 Tests
- `code/3D/v1_transfer3d/analysis/v1_confirmatory.csv`, `v1_verdict.json`, `v1_lr_selection.csv`, `v1_mechanism.csv`, `v1_layernorm_interaction_explorative.csv`
- `code/3D/v1_transfer3d/analysis/v1_explorativ_lr_robustheit.csv`, `explorativ_lr_robustheit.py`, `explorativ_lr_robustheit.txt`
- `code/3D/v1_transfer3d/logs/integrity.json`, `analysis.log`
- Die Rohdaten (`workers/*.csv`, etwa 28 MB) liegen auf dem Mac unter `3D/v1_transfer3d/workers/`.
