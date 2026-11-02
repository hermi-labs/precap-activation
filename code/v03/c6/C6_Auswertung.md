# C6 – Auswertung: Unabhängige Wiederholung mit PyTorch/CUDA auf anderer Hardware (CIFAR-10, d100)

Stand: 01.11.2026, abends. Die Auswertung folgt strikt `C6_Praeregistrierung.md`. PreCap, die Tiefenfamilie und Lagma sind Aarons Ideen. Lauf und Auswertung habe ich mit Claude gemacht, den Lauf hat Aaron auf seinem PC gestartet.

## Kurz

- **Erfolg.** Auf Aarons PC mit einer eigenständigen PyTorch-Umsetzung ist BPreCap bei fairen 4 Bit besser als TAT mit Gewichtsabfall: **−5,04 pp, 12/0, p = 0,0005** (MLX, B10b: −4,48 pp).
  - Hardware: RTX 4070, CUDA 12.6, torch 2.14.1, Windows.
- **Alle Kernbefunde wiederholen sich, auch in der Größe:**
  - gegen TAT ohne WD −11,44 pp (MLX −11,67);
  - PreCap-Kern gegen linearen Kern in fp32 −1,91 pp (MLX −1,93);
  - gegen SELU in fp32 −1,13 pp (MLX −0,95).
- **Plattformen stimmen im Mittel überein:** Die Mediane je Form liegen höchstens 0,5 pp auseinander. Die Differenz je Seed in fp32 liegt zwischen −1,0 und +1,2 pp. Auch die 4-Bit-Preise sind praktisch gleich (BPreCap +3,1 gegen +3,0; TAT ohne WD +14,3 gegen +14,6).
- **Ein Unterschied in der Statistik, nicht in der Größe:** BPreCap gegen SELU bei 4 Bit (S3) ist auf CUDA „besser“ (−0,48 pp, 10/2, Holm 0,0049). Auf MLX war es „unklar“ (−0,47 pp, 9/3, p = 0,077). Der Effekt ist gleich groß und klein, nur einmal knapp signifikant. Für v0.3 bleibt die Aussage: „bei d100 gegen SELU bei 4 Bit kein wesentlicher Unterschied“.
- **Folge für v0.3:** Die Befunde hängen nicht an MLX oder Apple-Hardware.

## Integrität

| Prüfung | Ergebnis |
|---|---|
| Läufe | 60 von 60 ok, 0 Ausfälle |
| Code und Daten auf dem PC | alle 15 eingefrorenen Dateien (Code, Tests, MLX-Referenz, CIFAR-10) per SHA-256 **identisch** zu den Dateien auf dem Mac |
| Tests auf dem PC vor dem Start | Selbsttest, **Äquivalenztest gegen MLX auf CUDA**, Auswertungstest und Rauchtest ok (zweimal: Testlauf 17:12, Start 17:15) |
| Äquivalenz auf CUDA | Logits bei der Init relativ 3·10⁻⁶ bis 1,4·10⁻⁴. Gradienten so nah an float64 wie MLX (z. B. BPreCap erste Schicht: CUDA−f64 4,7·10⁻⁴, MLX−f64 2,6·10⁻³). AdamW gleich bis 1,5·10⁻⁸ |
| Laufzeit | 17:17–21:21 (2 Worker). Je Lauf etwa 2,5 min (SELU) bis 18 min (BPreCap), siehe unten |
| Umgebung | Python 3.12.10, torch 2.14.1+cu126, CUDA 12.6, RTX 4070, numpy 2.5.4, float32 ohne TF32, deterministische Algorithmen |
| Übertragung | `scp` vom PC in `studien/c6/pc_ergebnisse/` |
| Sicherung | `sicherung/c6_2026-11-01/`, 19 von 19 Dateien OK |

Befehl: `.venv/bin/python -I c6_analyze.py ~/…/studien/c6/pc_ergebnisse ~/…/studien/c6/mlx_vergleich`, Ausgabe in `pc_ergebnisse/analyse_ausgabe.txt`.

**Offen genannt:** Im Startprotokoll steht ein erster Abbruch (17:11). Grund war die cmd-Variable mit einem Leerzeichen am Ende (`set PRECAP_DATA=… &`), dadurch fand das Skript die Daten nicht. Das Skript hat dabei nichts eingefroren und nichts gestartet. Danach lief alles mit `set "PRECAP_DATA=…"`.

## Primärtest

| Test | Median | S/N | p | Urteil |
|---|---|---|---|---|
| BPreCap − TAT a (1e-4, WD 0,3), w8a4a, ES | **−5,04 pp** | 12/0 | 0,0005 | **besser, Erfolg** |

## Sekundärtests (Holm über 5)

| Nr. | Vergleich (ES) | Median | S/N | p (Holm) | Urteil | MLX-Studie zum Vergleich |
|---|---|---|---|---|---|---|
| S1 | BPreCap − TAT a, fp32 | −0,37 | 10/2 | 0,0049 | besser | B10b S2: −0,52, 9/3, unklar |
| S2 | BPreCap − TAT (2e-4, ohne WD), w8a4a | −11,44 | 12/0 | 0,0024 | besser | B10: −11,67, 12/0 |
| S3 | BPreCap − SELU, w8a4a | −0,48 | 10/2 | 0,0049 | besser | B12 P3: −0,47, 9/3, unklar |
| S4 | BPreCap − LB, fp32 | −1,91 | 12/0 | 0,0024 | besser | B10: −1,93, 12/0 |
| S5 | BPreCap − SELU, fp32 | −1,13 | 12/0 | 0,0024 | besser | B12: −0,95, 12/0 |

## Plattformvergleich (beschreibend, Mediane am Stopp-Punkt, Testfehler in %)

| Form | fp32 CUDA | fp32 MLX | w8a4a CUDA | w8a4a MLX | 4-Bit-Preis CUDA | 4-Bit-Preis MLX | Bereich CUDA |
|---|---|---|---|---|---|---|---|
| BPreCap | 46,56 | 46,52 | 49,68 | 49,62 | +3,11 | +3,02 | 3,0 |
| LB | 48,37 | 48,47 | 50,83 | 50,78 | +2,42 | +2,31 | 2,22 |
| TAT a (WD 0,3) | 46,86 | 47,06 | 54,65 | 54,17 | +7,72 | +7,27 | 3,23 |
| TAT (ohne WD) | 46,87 | 46,58 | 61,23 | 61,22 | +14,32 | +14,61 | 27,6 |
| SELU | 47,77 | 47,51 | 50,23 | 50,07 | +2,54 | +2,53 | 3,86 |

- **Differenz CUDA − MLX in fp32 je Seed:** BPreCap −0,77 bis +0,47, LB −0,96 bis +1,15, TAT a −0,95 bis +1,07, TAT ohne WD −0,88 bis +0,75, SELU −0,50 bis +0,97 pp.
  - Die Trajektorien laufen durch die andere Rundung auseinander, im Mittel stimmen sie überein.
  - Die Vorab-Erwartung „Mediane innerhalb 1 pp“ (0,7) ist erfüllt: höchstens 0,48 pp.
- **Laufzeit je Lauf auf der RTX 4070 (Mittel):**

| Form | Dauer |
|---|---|
| SELU | 154 s |
| TAT | 321 s |
| LB | 538 s |
| BPreCap | 1098 s |

  - Das Netz ist schmal und tief, die Karte ist durch viele kleine Kernel-Aufrufe begrenzt.
  - BPreCap hat die meisten elementweisen Operationen je Schicht (exp, tanh, Hülle). Ein fusionierter Kernel würde hier viel bringen. Das Paket hat einen für PreCap, aber noch keinen für BPreCap.

## Deutung nach den Vorab-Regeln

- **Erfolg und S2 „besser“:** Der 4-Bit-Vorteil gegen TAT hängt nicht an MLX oder Apple-Hardware.
- **S3 und S5:** Die Regel „S3 unklar, S5 besser, wie B12“ trifft nur zur Hälfte zu. S5 ist wie erwartet „besser“, S3 ist knapp „besser“ statt „unklar“, bei gleicher Größe (−0,48 gegen −0,47 pp). Das SELU-Bild wiederholt sich in der Größe. Der kleine 4-Bit-Abstand zu SELU ist auf einer Plattform signifikant, auf der anderen nicht.
- **Plattformen stimmen im Mittel überein.**

## Abgleich mit den Vorab-Erwartungen

| Größe | Erwartung | Ergebnis |
|---|---|---|
| Primär „besser“ | 0,85 | ja |
| S2 „besser“ | 0,9 | ja |
| S3 „unklar“ | 0,6 | nein, knapp „besser“ |
| Mediane je Form innerhalb von 1 pp | 0,7 | ja (höchstens 0,48 pp) |
| Äquivalenztest auf CUDA ok | 0,85 | ja |

## Grenzen

- Keine neue Hyperparameter-Suche. Gemessen ist die Übertragbarkeit der MLX-Wahl.
- Nur CIFAR-10, d100. B11 (CIFAR-100) und B15 (d200) sind nicht auf CUDA wiederholt.
- Dieselben Daten, dieselben Startgewichte und dieselbe Batch-Reihenfolge. Unabhängig sind Framework, Kernels, Rundung und Hardware, nicht die Versuchsplanung.

**Vertrauen:** 0,9, dass die Kernbefunde gegen TAT nicht an der Plattform hängen.
