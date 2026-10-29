# Q3 – Auswertung: asymmetrische 4-Bit-Quantisierung, cs gegen TAT

*Veröffentlichte Fassung vom 29.10.2026: Der Bericht ist inhaltlich unverändert. Entfernt wurden nur der Vermerk „privat“ und Querverweise auf noch unveröffentlichte Studien. Präregistrierung und Skripte sind die eingefrorenen Originale.*

Stand: 28.10.2026, 05:50. Ausgewertet strikt nach `Q3_Praeregistrierung.md`. Lauf: 28.10., 05:04 bis 05:38.

## Integrität

```
$ (cd studien/q3 && shasum -a 256 -c meta/frozen_sha256.txt)
q3_run.py: OK
q3_analyze.py: OK
lr_from_p3a3.csv: OK
lr_tat_from_q2.csv: OK
Q3_Praeregistrierung.md: OK
src/precap/__init__.py: OK
src/precap/_presets.py: OK
src/precap/mlx.py: OK
src/precap/torch.py: OK
```

- 180/180 ok, 0 Ausfälle.
- Reproduktion: alle 180 Läufe gefunden, größte Abweichung von fp32 und symmetrischem w8a4_best gegen Q1b/Q2 **0,00 pp**. Bestanden.
- Gesichert in `sicherung/q3_2026-10-28/`.
- Auswertung: `.venv/bin/python -I studien/q3/q3_analyze.py studien/q3 studien/q2 studien/q1b`.

## Primärtest: cs gegen TAT bei w8a4a_best (asymmetrisch), Holm über 4

| Zelle | Median-Diff | W/L | Holm-p | Urteil |
|---|---|---|---|---|
| MNIST d50 | +0,34 pp | 7/8 | 0,93 | unklar |
| Fashion d50 | −2,16 pp | 14/1 | 0,0005 | besser |
| MNIST d100 | −1,08 pp | 13/2 | 0,0012 | besser |
| Fashion d100 | −1,96 pp | 15/0 | 0,0002 | besser |

**Gesamturteil: „besser als TAT“** (3 besser, keine schlechter). Vorab-Erwartung 0,45.

## Sekundär

| Familie | MNIST d50 | Fashion d50 | MNIST d100 | Fashion d100 | Urteil |
|---|---|---|---|---|---|
| S1 cs − SELU, asym | +0,49 schlechter | −0,49 besser | −2,14 besser | −1,20 besser | gemischt |
| S2 TAT asym − sym | −2,37 | −5,84 | −6,47 | −6,88 | asym hilft TAT (4/4) |
| S3 cs asym − sym | −1,07 | −0,09 unklar | −1,36 | −1,53 | asym hilft cs (3/4) |
| S4 cs − TAT, sym (Wiederholung Q2) | −1,37 | −7,71 | −6,53 | −6,87 | identisch zu Q2 |

## Beschreibend (Mediane, Testfehler in %)

| Zelle | Akt. | fp32 | w8a4 sym best | w8a4 asym best | w8a4 asym max | untere Grenze |
|---|---|---|---|---|---|---|
| MNIST d50 | cs | 4,60 | 6,83 | 5,67 | 5,88 | −1,75 |
| | TAT | 4,09 | 8,10 | 5,76 | 18,06 | −7,06 |
| | SELU | 4,29 | 5,43 | 5,10 | 5,77 | −1,76 |
| Fashion d50 | cs | 14,81 | 15,51 | 15,41 | 15,68 | −1,76 |
| | TAT | 13,65 | 22,97 | 17,73 | 28,62 | −2,81 |
| | SELU | 14,98 | 16,92 | 16,01 | 16,79 | −1,76 |
| MNIST d100 | cs | 6,21 | 9,07 | 7,65 | 7,86 | −1,75 |
| | TAT | 3,52 | 15,42 | 8,94 | 26,05 | −4,33 |
| | SELU | 6,46 | 11,29 | 9,84 | 11,77 | −1,75 |
| Fashion d100 | cs | 17,25 | 20,77 | 19,24 | 20,21 | −1,75 |
| | TAT | 13,58 | 27,87 | 21,13 | 40,15 | −3,86 |
| | SELU | 16,64 | 22,24 | 20,61 | 22,30 | −1,75 |

- Asymmetrie halbiert ungefähr den 4-Bit-Verlust von TAT. Der Vorsprung von cs schrumpft von 6,5–7,7 auf 1,1–2,2 pp, bleibt aber in 3 von 4 Zellen signifikant.
- TAT wählt immer mse4 (15/15). Bei Max-Kalibrierung bleibt TAT weit zurück (18–40 %), cs kaum (5,9–20,2 %): cs braucht keine sorgfältige Kalibrierung.

## Bedeutung

- Der 4-Bit-Vorteil von cs gegen TAT hängt **nicht** am symmetrischen Schema. Er ist asymmetrisch aber deutlich kleiner (rund 1–2 pp statt 6–8 pp) und bei MNIST d50 nicht vorhanden.
- Formulierbar für v0.2: „Bei 4-Bit-Aktivierungen (Post-Training, pro Tensor, symmetrisch und asymmetrisch) ist cs bei Tiefe 100 robuster als TAT; in fp32 ist TAT besser.“
- Gegen SELU ist das Bild asymmetrisch gemischt (d100 besser, MNIST d50 schlechter). Für Q1/Q1b gilt die Aussage „cs robuster als SELU“ also auch asymmetrisch bei d100.

## Grenzen

Nur Post-Training-Quantisierung, simuliert, pro Tensor (kein pro Kanal, kein QAT). Adam statt K-FAC. MLPs, Breite 256, 1720 Schritte, MNIST/Fashion.

## Vertrauen

0,85 für „cs bei d100 auch asymmetrisch robuster als TAT“. Einschränkungen: kleiner Abstand (1–2 pp), pro-Kanal-Quantisierung nicht geprüft.
