# C7 – Präregistrierung (Entwurf): Unabhängige Wiederholung von V2a2 mit PyTorch/CUDA auf anderer Hardware

Stand: 22.11.2026, vormittags. **Entwurf, noch nicht eingefroren.** PreCap, BPreCap, die Hülle und das Kalibrierungsrezept sind Aarons Ideen. Das Design habe ich mit Claude entworfen, nach dem Vorbild von C6 (v0.3.1). Start nur auf Aarons PC (RTX 4070) und nur mit seiner Freigabe.

## Warum

- Alle CNN-Studien für v0.4 liefen auf einem Rechner mit einem Framework (Apple M1 Pro, MLX 0.32.4). Ein Leser kann fragen, ob die Befunde an MLX oder an Apple-Kernels hängen.
- C7 wiederholt den Hauptvergleich von v0.4 (V2a2: BPreCap gegen SELU und TAT bei Tiefe 50, 4 Bit) mit einer **eigenständigen PyTorch-Umsetzung** auf **NVIDIA-Hardware**.
- „Wiederholung“ heißt: Daten, Startgewichte, Batch-Reihenfolge und Rechenvorschrift sind identisch, die Kernels und damit die Rundung sind andere. Bitgleiche Ergebnisse sind nicht zu erwarten.

## Die eine Änderung

Framework und Hardware: PyTorch (float32, ohne TF32, deterministische Algorithmen) auf CUDA statt MLX auf Apple-GPU. Sonst ist alles wie in V2a2:
- Netz L = 50, 32 Kanäle, Delta-orthogonal, ohne Normierung und Residuen;
- CIFAR-10 mit denselben Normierungskonstanten (float32-Werte der Studien, fest eingetragen);
- AdamW wie MLX 0.32: ohne Bias-Korrektur, entkoppelter WD, Kosinus auf 0 über 3440 Schritte, das k-te Update nutzt schedule(k − 1);
- 8 Messpunkte, frühes Stoppen am kleinsten Val-Fehler;
- PTQ: w8 je Ausgangskanal; a8 symmetrisch; a4 asymmetrisch mit max, p999, p9999, p99999 und mse4, Wahl der besten auf 1000 Trainingsbildern; für TAT zusätzlich CLE.

**Hyperparameter aus dem V2a2-Tuning, nicht neu getunt:** BPreCap LR 3e-5, WD 0; SELU 3e-5, 0; TAT 1e-4, 0.

**Seeds:** 42350–42361 wie V2a2. Damit sind Startgewichte und Batch-Reihenfolge je Seed dieselben wie in V2a2. **Soll:** 36 Läufe.

## Vorab-Prüfung (Mac)

- `test/mlx_referenz_c7.py` (MLX) und `test/test_c7_aequivalenz.py` (PyTorch, CPU):
  - Daten und Startgewichte bitgleich;
  - Formen;
  - Logits und Gradienten bei der Init, beide gegen float64;
  - Optimierer an synthetischen Gradienten;
  - Quantisierer bitgleich;
  - Kalibrierung bei der Init;
  - CLE;
  - 172 Trainingsschritte.
- Ergebnis in `test/aequivalenz_ergebnis_<gerät>.json`, Ausgaben in `test/aequivalenz_ausgabe_*.txt`.
- **Zwei Änderungen vor dem Freeze, nach dem ersten Testlauf auf dem Mac (22.11.):**
  - SELU nutzt jetzt die gerundeten MLX-Konstanten (λ = 1,0507, α = 1,67326) statt der exakten aus `F.selu`. Erst damit rechnet C7 dieselbe Funktion wie V2a2. Lauf 1 zeigte eine Formabweichung von 1,3·10⁻⁵.
  - Kriterium für den Gradienten bei der Init: vorher „Maximum je Schicht ≤ 3 × MLX“, jetzt „Median über die Schichten ≤ 3 × MLX und Maximum < 10⁻²“. Grund: Bei SELU liegt der größte Fehler in Schicht 0 (3 Eingangskanäle). Dort summiert die CPU-Faltung von PyTorch ungenauer als MLX (Maximum 1,4·10⁻³ gegen 1,4·10⁻⁴, Median 2,3·10⁻⁴ gegen 0,9·10⁻⁴). Das ist Rundung, kein Rechenfehler.
- Danach Selbsttest und Rauchtest auf dem PC.

## Primärtests (wie V2a2, Holm über 2)

- **P1:** BPreCap − SELU, Testfehler w8a4a_best am Punkt des frühen Stoppens, Wilcoxon gepaart über 12 Seeds, zweiseitig.
- **P2:** BPreCap − TAT, ebenso.
- **Urteil:**
  - „besser“: Holm-p < 0,05 und Median < 0;
  - „schlechter“: Holm-p < 0,05 und Median > 0;
  - sonst „unklar“.
- Ein Ausfall zählt als 90 %.

## Sekundär (beschreibend, kein Test)

- Je Form und Messgröße: Median CUDA gegen Median MLX (V2a2), Differenz in pp, für fp32, w8a8, w8a4a_max, w8a4a_best und den 4-Bit-Preis.
- Je Seed: Abstand CUDA − MLX.
- TAT nach CLE gegen MLX.

## Entscheidungsregel

| Ausgang | Deutung für v0.4 |
|---|---|
| P1 und P2 „besser“ | Der Hauptbefund hängt nicht an MLX oder Apple-Hardware. v0.4 berichtet C7 als Wiederholung (wie C6 in v0.3.1). |
| P2 „besser“, P1 „unklar“ | Der Abstand zu TAT ist robust, der kleinere zu SELU nicht. v0.4 schwächt die SELU-Aussage ab. |
| ein Vergleich „schlechter“ | Der Befund hängt an der Plattform. v0.4 berichtet das offen und schränkt die Kernaussage ein. |

## Vorab-Wahrscheinlichkeiten

- P2 „besser“: 0,95 (in V2a2 −13,7 pp, 12/0)
- P1 „besser“: 0,75 (in V2a2 −2,55 pp, 12/0; der Abstand ist kleiner)
- Mediane je Form innerhalb von 1 pp zu MLX: 0,7
- alle 36 Läufe ok: 0,95

## Grenzen

- Gleiche Seeds und Hyperparameter wie V2a2. C7 prüft also die Plattform, nicht neue Stichproben.
- Kein Tuning auf CUDA.
- Nur V2a2. Die tanh-Vergleiche und die Rauschmessungen werden nicht wiederholt.

## Aufwand

36 Läufe auf der RTX 4070. Geschätzt 3–6 min je Lauf, also 2–4 h mit einem Prozess (Schätzung, nicht gemessen; der Rauchtest misst es).
