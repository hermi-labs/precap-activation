# C6 – Präregistrierung: Unabhängige Wiederholung mit PyTorch/CUDA auf anderer Hardware (CIFAR-10, d100, privat)

Stand: 01.11.2026, 11:30. Vor dem Start, **noch nicht eingefroren**. PreCap, die Tiefenfamilie und Lagma sind Aarons Ideen. Das Design habe ich mit Claude entworfen. Fahrplan v0.3, Punkt C6 (Aaron, 31.10.). **Start nur nach Aarons Freigabe auf seinem PC** (RTX 4070), Zugang per SSH gibt Aaron. Nichts veröffentlichen.

## Warum

- **Plattform:** Alle B-Studien liefen auf einem Rechner mit einem Framework (Apple M1 Pro, MLX). Ein Leser kann fragen, ob die Befunde an MLX oder an Apple-Kernels hängen.
- **Ziel:** C6 wiederholt die Kernvergleiche mit einer **eigenständigen PyTorch-Umsetzung** auf **NVIDIA-Hardware** (CUDA).
- **Was „Wiederholung“ hier heißt:** Daten, Startgewichte und Rechenvorschrift sind identisch, die Hardware-Kernels und damit die Rundung sind andere. Bitgleiche Ergebnisse sind nicht zu erwarten.

## Was vorab geprüft ist (Mac, CPU, 01.11. vormittags)

`test/test_c6_aequivalenz.py` gegen die MLX-Referenz aus dem eingefrorenen `studien/b10/b10_run.py`: **AEQUIVALENZ OK, 42 Prüfungen.**

- **Daten und Startgewichte bitgleich.**
  - Die Normierung nutzt die exakten float32-Werte der Studien. Grund: NumPy 2.0.2 summiert float32 ungenauer als 2.5.4, mit dem die Studien liefen.
- **Logits und Kalibrierung bei der Init:** relativ 4·10⁻⁸ bis 1,4·10⁻⁴.
- **Gradienten bei der Init:** PyTorch-float32 und MLX-float32 liegen gleich weit von einer float64-Rechnung entfernt (BPreCap und TAT etwa 10⁻³, SELU etwa 2·10⁻², LB etwa 10⁻⁵). Die Unterschiede sind also Rundung über 100 Schichten, keine andere Vorschrift.
- **AdamW an identischen Gradienten:** gleich bis auf 1,5·10⁻⁸ (wenige float32-Stellen).
- **LR-Plan:** Das k-te Update nutzt schedule(k−1), wie MLX. Das wurde in MLX mit SGD direkt nachgemessen.
- **Nach 172 Schritten:** Testfehler innerhalb von 2 pp der MLX-Läufe.
- **Rauchtest** (3 Läufe, 40 Schritte, mit Zwischenmessung): ok.

## Design

- **Netz, Daten, Rezept, Regime, Messung wie B10:**
  - MLP mit Breite 256 und Tiefe 100, ohne Normierung und Abkürzungen;
  - orthogonale Init × Gain, Normierung je Beispiel;
  - AdamW mit Kosinus auf 0 über 6880 Schritte;
  - Messpunkte alle 860 Schritte, frühes Stoppen nach fp32-Val;
  - Quantisierung nach dem Training: w8 je Kanal, a4 symmetrisch und asymmetrisch mit der besten von vier Kalibrierungen, a3 symmetrisch.
- **Formen und Hyperparameter aus den MLX-Studien übernommen, nicht neu getunt:**

| Kürzel | Form | LR / WD | Quelle |
|---|---|---|---|
| bp | BPreCap (kL2B_tr) | 1e-4 / 0,3 | B10 |
| lb | LB_tr | 1e-4 / 0,3 | B10 |
| ta | TAT | 1e-4 / 0,3 | B10b (TAT a) |
| t0 | TAT | 2e-4 / 0 | B10 (Wahl des fp32-Tunings) |
| se | SELU_tr | 2e-4 / 0,3 | B12 |

- **Seeds:** 42168–42179, dieselben wie in B10, B10b und B12. Startgewichte und Batch-Reihenfolge sind damit identisch zu den MLX-Läufen.
- **Läufe:** 5 × 12 = 60.
- **Genauigkeit:** float32, TF32 ausgeschaltet, deterministische Algorithmen.
- **Hardware:** RTX 4070, 2 Prozesse.

## Primärtest

**BPreCap − TAT a bei w8a4a_best am Stopp-Punkt.** Das ist die Wiederholung des B10b-Primärtests.
- Zweiseitiger exakter Wilcoxon-Test, gepaart über Seeds, α = 0,05.
- Eigene exakte Rechnung: Nullen verworfen, Mittelränge.
- **Erfolg (Befund wiederholt):** p < 0,05 und Median < 0.
- Ein Ausfall zählt als 90 %.

## Sekundär (Holm über 5)

| Nr. | Vergleich (Stopp-Punkt) | wiederholt |
|---|---|---|
| S1 | BPreCap − TAT a, fp32 | B10b S2 (dort unklar) |
| S2 | BPreCap − TAT (2e-4, ohne WD), w8a4a | B10-Primär |
| S3 | BPreCap − SELU, w8a4a | B12 P3 (dort unklar) |
| S4 | BPreCap − LB, fp32 | Kern-Befund B9/B10 |
| S5 | BPreCap − SELU, fp32 | B12 S3 |

**Beschreibend:**
- Mediane je Form und 4-Bit-Preis.
- Bereich.
- Stopp-Punkte.
- **Plattformvergleich:** Mediane CUDA gegen MLX je Form und die Differenz je Seed in fp32. Die MLX-Roh-CSVs liegen in `mlx_vergleich/`.

## Deutungsregeln (vorab)

| Ergebnis | Deutung |
|---|---|
| Erfolg und S2 „besser“ | Der 4-Bit-Vorteil gegen TAT hängt nicht an MLX oder Apple-Hardware |
| Primär „unklar“ | Auf CUDA nicht wiederholt. In v0.3 offen berichten, die Aussage gilt dann nur für MLX/Apple |
| Primär „schlechter“ | Widerspruch zur MLX-Reihe. Ursache suchen, v0.3 erst danach |
| S3 und S5 wie B12 (S3 unklar, S5 besser) | Das SELU-Bild wiederholt sich |
| Mediane CUDA und MLX je Form innerhalb von 1 pp (fp32) | Die Plattformen stimmen im Mittel überein |

## Vorab-Erwartung

| Größe | Erwartung |
|---|---|
| Primär „besser“ | 0,85 |
| S2 „besser“ | 0,9 |
| S3 „unklar“ | 0,6 |
| Mediane je Form innerhalb von 1 pp | 0,7 |
| Äquivalenztest auf CUDA ok | 0,85 |

## Grenzen

- Keine neue Hyperparameter-Suche. Gemessen wird die Übertragbarkeit der MLX-Wahl.
- Nur CIFAR-10, d100.
- Andere Rundung, deshalb sind Trajektorien je Seed nicht vergleichbar, nur die Verteilungen.
- Der Äquivalenztest auf CUDA läuft erst auf dem PC (im Startskript). Schlägt er fehl, startet C6 nicht.

## Ablauf

- **Start:** `python c6_launch.py --device cuda --workers 2` auf dem PC, Anleitung in `ANLEITUNG_PC.md`.
- **Das Startskript führt aus:**
  - Prüfung der Daten (SHA-256);
  - Freeze (`meta/frozen_sha256.txt`);
  - Selbsttest;
  - Äquivalenztest auf CUDA;
  - Auswertungstest;
  - Rauchtest;
  - Start der Worker.
- **Auswertung:** `python c6_analyze.py` auf dem PC oder nach dem Zurückkopieren auf den Mac.
