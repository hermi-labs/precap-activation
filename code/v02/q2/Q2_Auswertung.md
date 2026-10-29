# Q2 – Auswertung: SPreCap-cs gegen TAT bei 4-Bit-Aktivierungen

*Veröffentlichte Fassung vom 29.10.2026: Der Bericht ist inhaltlich unverändert. Entfernt wurden nur der Vermerk „privat“ und Querverweise auf noch unveröffentlichte Studien. Präregistrierung und Skripte sind die eingefrorenen Originale.*

Stand: 28.10.2026, 02:05. Ausgewertet strikt nach `Q2_Praeregistrierung.md`. Lauf: 28.10., 01:34 bis 01:56. Vergleichsdaten cs und SELU aus Q1b; Q1 und Q1b sind im v0.2-Entwurf gemeinsam berichtet.

## Integrität

```
$ (cd studien/q2 && shasum -a 256 -c meta/frozen_sha256.txt)
q2_run.py: OK
q2_analyze.py: OK
lr_from_p3a3.csv: OK
Q2_Praeregistrierung.md: OK
src/precap/__init__.py: OK
src/precap/_presets.py: OK
src/precap/mlx.py: OK
src/precap/torch.py: OK
```

- Tuning 96/96 ok, Bestätigung 62/62 ok, 0 Ausfälle. Keine TAT-LR am Rasterrand.
- Reproduktion cs MNIST d100 (Seeds 42010/42011) gegen Q1b: fp32 und w8a4_best je 0,00 pp. Bestanden.
- TAT-fp32 bei MNIST d100 (3,52 %) stimmt mit A2 überein (3,52 %), die Implementierung ist also konsistent.
- Rohdaten gesichert in `sicherung/q2_2026-10-28/` (mit `SHA256.txt`).
- Auswertung: `.venv/bin/python -I studien/q2/q2_analyze.py studien/q2 studien/q1b` → `analysis/q2_result.json`.

## Primärtest: cs gegen TAT, Testfehler bei w8a4_best, Holm über 4

Differenz = cs − TAT, negativ heißt cs besser.

| Zelle | n | Median-Diff | W/L | Holm-p | Urteil |
|---|---|---|---|---|---|
| MNIST d50 | 15 | −1,37 pp | 13/2 | 0,0003 | besser |
| Fashion d50 | 15 | −7,71 pp | 15/0 | 0,0002 | besser |
| MNIST d100 | 15 | −6,53 pp | 15/0 | 0,0002 | besser |
| Fashion d100 | 15 | −6,87 pp | 15/0 | 0,0002 | besser |

**Gesamturteil: „besser als TAT“** (4/4 Zellen). Vorab-Erwartung dafür war 0,30.

## Sekundär

| Familie | MNIST d50 | Fashion d50 | MNIST d100 | Fashion d100 | Urteil |
|---|---|---|---|---|---|
| S1 cs − TAT, fp32 | +0,81 | +1,15 | +2,68 | +3,93 | cs schlechter (4/4) |
| S2 cs − TAT, Verlust durch Quantisierung | −1,88 | −8,97 | −9,32 | −11,09 | cs besser (4/4) |
| S3 TAT − SELU, w8a4_best | +2,49 | +6,67 | +3,42 | +6,08 | TAT schlechter als SELU (4/4) |

Alle Holm-p ≤ 0,005.

## Beschreibend (Mediane, Testfehler in %)

| Zelle | Akt. | fp32 | w8a8 | w8a4 max | w8a4_best | Bereich max | gewählte Kalibrierung |
|---|---|---|---|---|---|---|---|
| MNIST d50 | cs | 4,60 | 4,62 | 7,83 | 6,83 | 3,3 | mse4 13, p999 2 |
| | TAT | 4,09 | 4,12 | 42,24 | 8,10 | 18,6 | mse4 15 |
| | SELU | 4,29 | 4,30 | 13,49 | 5,43 | 7,6 | mse4 12, p999 3 |
| Fashion d50 | cs | 14,81 | 14,75 | 15,77 | 15,51 | 3,4 | gemischt |
| | TAT | 13,65 | 13,56 | 51,62 | 22,97 | 7,0 | mse4 15 |
| | SELU | 14,98 | 14,98 | 22,57 | 16,92 | 7,7 | mse4 9, p999 6 |
| MNIST d100 | cs | 6,21 | 6,22 | 10,66 | 9,07 | 3,3 | mse4 12 |
| | TAT | 3,52 | 3,54 | 39,81 | 15,42 | 8,3 | mse4 15 |
| | SELU | 6,46 | 6,52 | 29,93 | 11,29 | 6,2 | mse4 15 |
| Fashion d100 | cs | 17,25 | 17,19 | 21,98 | 20,77 | 3,3 | mse4 14 |
| | TAT | 13,58 | 13,57 | 58,26 | 27,87 | 7,5 | mse4 15 |
| | SELU | 16,64 | 16,74 | 37,26 | 22,24 | 6,3 | mse4 15 |

- Bei 8 Bit verliert keine Variante nennenswert. Der Unterschied entsteht erst bei 4 Bit.
- TAT verliert bei 4 Bit 4 bis 14 pp, cs 1 bis 3,5 pp.
- Der Bereich von cs bleibt bei 3,3 bis 3,4, unabhängig von Tiefe und Datensatz. Das ist die Beschränktheit.

## Bedeutung

- **Die Beschränktheit ist in diesem Setup eine belegte Nische.** Bei 4-Bit-Aktivierungen ist cs in allen 4 Zellen klar besser als TAT, obwohl TAT in fp32 in allen 4 Zellen klar besser ist. Es ist ein Zielkonflikt, kein Sieg auf ganzer Linie.
- Bei d100 ist der Vorsprung bei 4 Bit (6,5 bis 6,9 pp) größer als der Rückstand in fp32 (2,7 bis 3,9 pp).
- TAT ist bei 4 Bit sogar schlechter als SELU. Sein Vorteil in fp32 kehrt sich bei wenigen Bits um.

## Wichtigste Einschränkung (ehrlich, vor jeder Veröffentlichung klären)

Die Quantisierung ist **pro Tensor symmetrisch** (`qa` in `q2_run.py`, wie in Q1b). Leaky ReLU ist stark asymmetrisch: Die negative Seite ist um den Faktor 0,57 (d100) bzw. 0,43 (d50) gestaucht. Bei symmetrischem Raster verschwendet TAT damit einen Teil der 16 Stufen. Asymmetrische Quantisierung mit Nullpunkt ist in der Praxis üblich und würde TAT vermutlich helfen. SELU ist ebenfalls asymmetrisch. Der Vergleich ist also für dieses Schema fair (alle bekommen dasselbe Schema und dieselbe Kalibrierwahl), aber er ist nicht schema-unabhängig.

**Vorschlag:** Folgestudie Q3 mit asymmetrischer Quantisierung pro Tensor (min/max mit Nullpunkt, gleiche Kalibrierwahl). Erst wenn cs dort auch vorne liegt, sollte „robuster bei 4 Bit als TAT“ öffentlich stehen. Sonst lautet die Aussage: „robuster bei symmetrischer 4-Bit-Quantisierung“.

Weitere Grenzen: nur Post-Training-Quantisierung, simuliert; Adam statt K-FAC; MLPs, Breite 256, 1720 Schritte, MNIST/Fashion.

## Vertrauen

0,9, dass das Ergebnis für dieses Schema stimmt (4/4, 13–15 von 15 Seeds, Reproduktion exakt). 0,5, dass es unter asymmetrischer Quantisierung in gleicher Stärke hält.
