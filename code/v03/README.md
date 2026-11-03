# Studien zu v0.3 und v0.3.1

16 präregistrierte Studien (B1–B15, B10b) und eine beschreibende Diagnose (D1) zum Preprint v0.3 ([`paper/PreCap_paper_v0.3.pdf`](../../paper/PreCap_paper_v0.3.pdf)). Mit v0.3.1 ([`paper/PreCap_paper_v0.3.1.pdf`](../../paper/PreCap_paper_v0.3.1.pdf)) kommen vier präregistrierte Studien (V1, V1c, V1d, V3) und vier beschreibende Messungen (D2, V1b, D3, D3b) dazu. Sie liefen mit Apple MLX auf einem MacBook Pro M1 Pro. Dazu kommt eine unabhängige Wiederholung mit PyTorch/CUDA auf einer RTX 4070 (C6). Jede Studie ändert genau eine Sache gegenüber ihrer Vorgängerin.

| Ordner | Frage | Bericht |
|---|---|---|
| `b1/` | BPreCap gegen eine geregelte Lagma-Hülle (`kL2_LD100`), gegen kL2 aus v0.2, LB und SPreCap-cs; MNIST/Fashion, Tiefe 100/200 | `b1/B1_Auswertung.md` |
| `b2/` | weiche Lagma-Hülle gegen harte Grenze | `b2/B2_Auswertung.md` |
| `b3/` | Trägt der PreCap-Kern auf CIFAR-10 etwas bei? | `b3/B3_Auswertung.md` |
| `b4/` | Ist es die weiche Form oder die engere Grenze? | `b4/B4_Auswertung.md` |
| `b5/` | Hält der B3-Befund bei Tiefe 200 nach Nachtuning der Lernrate? | `b5/B5_Auswertung.md` |
| `b6/` | Kommt der TAT-Vorsprung vom Rezept oder von der Form? | `b6/B6_Auswertung.md` |
| `b7/` | Hält der Vorteil von BPreCap über LB bei vierfachem Budget? | `b7/B7_Auswertung.md` |
| `b8/` | Hält der B6-Befund bei Tiefe 200? | `b8/B8_Auswertung.md` |
| `b9/` | BPreCap gegen TAT im gleichen Rezept, langes Training mit frühem Stoppen | `b9/B9_Auswertung.md` |
| `b10/` | wie B9 mit AdamW, Kosinus-LR und Gewichtsabfall | `b10/B10_Auswertung.md` |
| `b10b/` | Gegenprobe: TAT in den Einstellungen mit Gewichtsabfall | `b10b/B10b_Auswertung.md` |
| `b11/` | CIFAR-100 | `b11/B11_Auswertung.md` |
| `b12/` | Basislinien ReLU, GELU, SELU im gleichen Rezept | `b12/B12_Auswertung.md` |
| `b13/` | weich gegen hart im TAT-Rezept | `b13/B13_Auswertung.md` |
| `b14/` | feste weiche Hülle gegen lernbaren harten Clip (PACT-Stil) | `b14/B14_Auswertung.md` |
| `b15/` | BPreCap gegen SELU bei Tiefe 200 | `b15/B15_Auswertung.md` |
| `d1/` | Diagnose: Warum zahlt TAT bei 4 Bit mehr? (beschreibend) | `d1/D1_Auswertung.md` |
| `c6/` | Wiederholung mit PyTorch/CUDA auf anderer Hardware | `c6/C6_Auswertung.md` |
| **v0.3.1** | | |
| `d2/` | Diagnose: Warum bricht BPreCap bei LR-Spitze 2·10⁻⁴ ein? (beschreibend) | `d2/D2_Auswertung.md` |
| `v1/` | wie B10 mit 5 % linearem LR-Anlauf für alle Formen (Tiefe 100) | `v1/V1_Auswertung.md` |
| `v1b/` | Wo liegt die LR-Grenze mit Anlauf? (beschreibend) | `v1b/V1b_Auswertung.md` |
| `v1c/` | wie B15 mit LR-Anlauf, BPreCap gegen SELU bei Tiefe 200 | `v1c/V1c_Auswertung.md` |
| `v1d/` | Länge des Anlaufs (0/1/2,5/5 %) und BPreCaps 4-Bit-Preis bei Tiefe 200 | `v1d/V1d_Auswertung.md` |
| `d3/` | Ausreißer und Fehlerfortpflanzung je Schicht bei Tiefe 200, int4 und int8 (beschreibend) | `d3/D3_Auswertung.md` |
| `d3b/` | Anlauf und Nutzung der Hülle bei Tiefe 200 (beschreibend, 2 Seeds, nicht im Preprint ausgewertet; V1d verweist darauf) | `d3b/D3b_Auswertung.md` |
| `v3/` | QAT als Obergrenze: BPreCap mit PTQ gegen LSQ- und PACT-QAT (Tiefe 100) | `v3/V3_Auswertung.md` |

## Inhalt je Ordner

- `<NAME>_Praeregistrierung.md`: Sie wurde vor dem Start eingefroren und ist unverändert. Einige Präregistrierungen erwähnen weitere, noch unveröffentlichte Studien. Sie bleiben trotzdem unverändert, damit die Hashes stimmen.
- `meta/frozen_sha256.txt`, `meta/frozen_at.txt`: Hashes und Zeitpunkt des Einfrierens. Prüfen mit `(cd code/v03/<name> && shasum -a 256 -c meta/frozen_sha256.txt)`. Die Zeilen zu den CIFAR-Dateien zeigen auf den Datenordner des Mac. Bei C6 liegt die Hash-Datei in `c6/pc_ergebnisse/meta/`.
- `*_run.py`, `*_analyze.py`, `launch_*.sh`: eingefrorene Skripte. `src/` ist, wo vorhanden, die eingefrorene Kopie des Pakets, mit der die Studie lief.
- `workers/`, `tune/workers/`: Rohergebnisse je Lauf (CSV). `analysis/`: Ausgabe der Auswertung. `logs/`: Protokolle der Worker.
- `<NAME>_Auswertung.md`: Bericht. In der veröffentlichten Fassung sind nur der Vermerk „privat“ im Titel und in B1 ein Satz mit Verweis auf eine unveröffentlichte Studie entfernt. Die Ergebnisse sind unverändert.

## Weitere Formen im Code (Lagma-Regler)

Die Lauf-Skripte stammen aus einer früheren, noch unveröffentlichten Studienreihe zur Lagma-Hülle (Lagma-1 bis 5). Sie enthalten deshalb weitere Formen, die in v0.3 nicht vorkommen: Hüllen, deren Grenze im Training über eine Ziel-Eingriffsrate geregelt wird (`LR10`, `LR03`, `LA10`, `LT`, `LTA`, `LD100`, `LD125`). B1 nutzt im Primärtest `kL2_LD100` als Vergleich. Das Preprint berichtet B1 nur über die Sekundärtests. Das gilt auch für die Lauf-Skripte von V1, V1b, V1c und V1d (v0.3.1), die auf B10 bzw. B15 aufbauen. Die Skripte sind unverändert, damit die Hashes stimmen.

## Nachrechnen

```bash
export PRECAP_DATA=/pfad/zu/data          # Ordner mit mnist/, fashion/ und cifar10/ bzw. cifar100/ (Binärfassung)
python code/v03/b10/b10_run.py --selfcheck    # Selbsttest, MLX nötig
```

- Die Auswertungsskripte erwarten zum Teil die Ordner von Vorgängerstudien als weitere Argumente. Welche das sind, steht oben im jeweiligen Bericht. Dort stehen die alten Pfade `studien/<name>`, im Repo heißen sie `code/v03/<name>`.
- **Auswertungen v0.3.1** (vom Repo-Wurzelordner, Python mit scipy):
  - `python code/v03/v1/v1_analyze.py code/v03/v1 code/v03/b10`
  - `python code/v03/v1b/v1b_analyze.py code/v03/v1b code/v03/v1`
  - `python code/v03/v1c/v1c_analyze.py code/v03/v1c code/v03/b15`
  - `python code/v03/v1d/v1d_analyze.py code/v03/v1d code/v03/v1c code/v03/b15`
  - `python code/v03/v3/v3_analyze.py code/v03/v3`
  - Vor der Veröffentlichung (03.11.2026) so aus diesem Ordner neu gerechnet: alle fünf Ergebnisdateien byte-gleich zu den Originalen.
- **D2 und D3** prüfen in ihren Hashes auch `../b10/b10_run.py`, `../b13/b13_run.py` bzw. `../b15/b15_run.py`. Diese Dateien sind die veröffentlichten aus v0.3, deshalb stimmen die Hashes nur in diesem Ordner.
- Die CIFAR-Daten liegen nicht im Repo, nur ihre Prüfsummen. Quelle: https://www.cs.toronto.edu/~kriz/cifar.html
- **Neulauf vor der Veröffentlichung (01.11.2026):** Ein Lauf aus B10 (BPreCap, Seed 42173) wurde in einer sauberen Kopie an einem fremden Ort neu gestartet. Er war in allen 93 Spalten bitgleich zum Original, ausgenommen Laufzeiten und Zeitstempel.
- **Abbildungen 1–4 des Preprints:** `python paper/make_fig1_v03.py` und `python paper/make_figures_v03.py`. Die Übersicht `paper/figures_v03/v03_overview.png` entsteht mit `python paper/make_overview_v03.py`, ebenfalls pixelgleich geprüft. Sie lesen nur das Paket bzw. die Roh-CSVs hier und ergeben pixelgleich die Bilder in `paper/figures_v03/` (geprüft am 02.11.2026).

## C6 (PyTorch/CUDA)

- `c6_run.py` ist eine eigenständige PyTorch-Umsetzung. `test/test_c6_aequivalenz.py` vergleicht sie vor dem Start mit der MLX-Fassung, die Ergebnisse stehen in `pc_ergebnisse/meta/`.
- **Nicht im Repo** sind die beiden Referenzdateien `test/mlx_ref.npz` (39 MB) und `test/mlx_ref2.npz` (13 MB). Sie lassen sich mit `test/mlx_referenz.py` und `test/mlx_referenz2.py` auf einem Mac mit MLX neu erzeugen.
  - SHA-256 der Originale: `3ebe29ee3e4d141b18c4e7cf989d0435749b964ab7575ca8ae24ed3d288ebbd0` bzw. `f2ed417c36c1310df6e89e880702ccf0b3e678565e2df5efa890754a24627702`, wie in `pc_ergebnisse/meta/frozen_sha256.txt`.
- `pc_paket.sh` baut das Paket für den PC.

## English note

Code, preregistrations (German), raw per-run CSVs, analysis outputs and reports for the v0.3 and v0.3.1 preprints (v0.3.1 adds D2, V1, V1b, V1c, V1d, D3, D3b and V3). All scripts are hash-frozen and unchanged. They also contain additional, unpublished hull variants with a regulated bound (`LR*`, `LA10`, `LT*`, `LD*`), which are not part of the v0.3 claims. CIFAR data is not included; set `PRECAP_DATA` to your data folder.
