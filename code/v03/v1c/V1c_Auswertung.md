# V1c – Auswertung: LR-Anlauf bei Tiefe 200, BPreCap gegen SELU (CIFAR-10, d200, TAT-Rezept)

Stand: 03.11.2026, 10:25, mit Claude. Ausgewertet strikt nach `V1c_Praeregistrierung.md` (Freeze 07:02). Läufe 07:10–10:17 auf 2 Workern (M1 Pro, Netzteil). PreCap, BPreCap und die Tiefenfamilie sind Aarons Ideen.

## Kurzfassung

- **Nach Präregistrierung: Erfolg.** Auch wenn beide Formen mit Anlauf trainieren und bis 3e-4 getunt werden, ist BPreCap bei d200 bei 4 Bit besser als SELU: **−8,15 pp, 12/0, p = 0,0005** (v0.3 ohne Anlauf: −8,95 pp).
- **Deutungsregel, Zeile 2:** Der Vorteil hält, aber der Anlauf bringt BPreCap bei d200 keine größere LR. Gewählt bleibt 5e-5 / 0,3 wie in B15. Bei 1e-4 bricht BPreCap mit Anlauf aber nicht mehr ein (46 % statt 85–86 % Val-Fehler).
- **SELU profitiert in fp32 vom Anlauf** (−3,5 pp, 12/0), alle 12 Läufe trainieren (B15: 5 von 12 kaum). Bei 4 Bit bleibt SELU gleich (S5 unklar, +0,11 pp). Der fp32-Abstand schrumpft deshalb von −11,97 auf −8,83 pp, der 4-Bit-Abstand kaum.
- **Neuer Befund (S3, schlechter):** Bei BPreCap senkt der Anlauf den fp32-Fehler leicht (−0,59 pp), erhöht aber den 4-Bit-Fehler (+1,64 pp, 0/12). Der 4-Bit-Preis steigt von +7,3 auf +9,5 pp. Dasselbe Muster gab es schwächer in V1 bei d100 (Preis +3,0 → +4,2, S3 unklar).

## Integrität

| Prüfung | Ergebnis |
|---|---|
| Hashes (`shasum -a 256 -c meta/frozen_sha256.txt`) | 15 von 15 OK |
| Zeilen | Tuning 48/48 ok, Bestätigung 24/24 ok, Kontrolle 2/2 ok, 0 Ausfälle |
| Kontrolle ohne Anlauf (Seed 42192) | kL2B_tr und selu_tr bitgleich zu B15, alle Kennzahlen an allen Messpunkten |

Befehl: `.venv/bin/python -I studien/v1c/v1c_analyze.py studien/v1c studien/b15`, Ergebnis in `analysis/v1c_result.json`.

## Tuning (Val-Fehler % am Ende, Seeds 42000 / 42001)

| LR \ WD | BPreCap 0 | BPreCap 0,3 | BPreCap 3,0 | SELU 0 | SELU 0,3 | SELU 3,0 |
|---|---|---|---|---|---|---|
| 5e-5 | 47,0 / 48,0 | **44,6 / 44,9** | 57,3 / 56,7 | 70,0 / 68,5 | 66,8 / 66,4 | 61,9 / 61,7 |
| 1e-4 | 50,4 / 49,7 | 46,2 / 46,1 | 58,8 / 59,8 | 65,8 / 65,6 | 62,7 / 62,5 | 90,5 / 90,5 |
| 2e-4 | 84,6 / 84,1 | 82,1 / 78,0 | 77,2 / 76,7 | 62,6 / 62,6 | 55,9 / 55,9 | 90,5 / 90,5 |
| 3e-4 | 81,8 / 86,2 | 81,7 / 90,5 | 77,8 / 90,4 | 90,2 / 86,1 | **56,2 / 53,9** | 90,5 / 90,5 |

- **Wahl:** BPreCap 5e-5 / 0,3 (unterer Rand des Rasters), SELU 3e-4 / 0,3 (oberer Rand).
- **Einbruch nach der Vorab-Schwelle** (beide Seeds über 80 %): BPreCap bei 2e-4/0 und 3e-4/0. Die übrigen Zellen bei 2e-4 und 3e-4 liegen bei 77–90 % und sind praktisch ebenfalls gestört, verfehlen aber die Schwelle in mindestens einem Seed.
- **BPreCap bei 1e-4:** kein Einbruch in allen drei WD (B15 ohne Anlauf: 85–86 % in allen WD). Der Anlauf verschiebt die Grenze bei d200 also von 1e-4 auf 2e-4, wie bei d100 von 2e-4 auf 3e-4 (V1b).

## Primärtest

| Vergleich | Median | W/L | p | Urteil |
|---|---|---|---|---|
| kL2B_tr − selu_tr, w8a4a, frühes Stoppen | **−8,15 pp** | 12/0 | 0,0005 | besser (Erfolg) |

Einzeldifferenzen: −9,09 · −7,48 · −8,44 · −8,93 · −7,44 · −6,93 · −6,93 · −11,78 · −8,10 · −8,65 · −8,19 · −7,46.

## Sekundär (Holm über 6)

| Nr. | Vergleich | Median | W/L | Holm-p | Urteil |
|---|---|---|---|---|---|
| S1 | kL2B_tr − selu_tr, fp32, ES | −8,83 | 12/0 | 0,003 | besser |
| S2 | kL2B_tr V1c − B15, fp32, ES | −0,59 | 12/0 | 0,003 | besser |
| S3 | kL2B_tr V1c − B15, w8a4a, ES | **+1,64** | 0/12 | 0,003 | **schlechter** |
| S4 | selu_tr V1c − B15, fp32, ES | −3,50 | 12/0 | 0,003 | besser |
| S5 | selu_tr V1c − B15, w8a4a, ES | +0,11 | 6/6 | 0,380 | unklar |
| S6 | kL2B_tr − selu_tr, w8a4a, Ende (6880) | −7,83 | 12/0 | 0,003 | besser |

**Zu S2 und S3:** BPreCap hat in V1c und B15 dieselbe LR/WD. Der Vergleich misst hier praktisch den Anlauf allein. Bei SELU (S4, S5) misst er das Paket aus Anlauf und neuer Wahl (3e-4 statt 2e-4).

## Beschreibend (Bestätigung, Mediane über 12 Seeds, frühes Stoppen)

| Form | LR / WD | Stopp-Punkte | fp32 | w8a4 | w8a4a | w8a3 | 4-Bit-Preis | kaum trainiert |
|---|---|---|---|---|---|---|---|---|
| BPreCap | 5e-5 / 0,3 | 11× 6880, 1× 6020 | 45,94 | 57,62 | 55,34 | 83,53 | +9,53 | 0 (B15: 0) |
| SELU | 3e-4 / 0,3 | 11× 6880, 1× 6020 | 54,84 | 64,96 | 63,42 | 84,63 | +8,39 | 0 (B15: 5) |

- **Bereich (Max der Kalibrierung):** BPreCap bleibt an allen Messpunkten bei 3,0 (die Hülle). Bei SELU fällt er von 3,8 auf 1,1.
- **Verlauf BPreCap fp32 / w8a4a:** 55,9/63,5 (860) → 50,0/55,6 (3440) → 46,1/55,1 (6020) → 45,9/55,5 (6880). Der 4-Bit-Fehler ist ab etwa Schritt 4300 flach, während fp32 weiter fällt.

## Vorab-Erwartungen gegen Ergebnis

| Größe | Erwartung | Ergebnis |
|---|---|---|
| BPreCap bricht mit Anlauf bei 1e-4 nicht ein | 0,7 | ja |
| Wahl für BPreCap ≥ 1e-4 | 0,45 | nein (5e-5) |
| Primär besser | 0,7 | ja |
| S4: Anlauf verbessert SELU in fp32 | 0,5 | ja (−3,5 pp) |
| SELU trainiert in allen 12 Läufen | 0,35 | ja |
| Kontrolle bitgleich | 0,95 | ja |

## Deutung

1. **Die v0.3-Aussage zu d200 hält auch mit Anlauf.** −8,15 statt −8,95 pp bei 4 Bit, beide 12/0. Der Vorsprung in fp32 schrumpft deutlicher (−8,8 statt −12,0 pp), weil SELU vom Anlauf und der höheren LR stärker profitiert als BPreCap.
2. **Die LR-Grenze von BPreCap bei d200 steigt mit Anlauf um eine Stufe** (Einbruch erst ab 2e-4), die beste LR bleibt aber 5e-5. Die Formulierung „diverges already at peak LR 10⁻⁴“ in v0.3 gilt nur ohne Anlauf und sollte in v0.3.1 eingeschränkt werden.
3. **Der Anlauf macht BPreCap bei 4 Bit etwas schlechter** (S3). Mit V1 zusammen ist das jetzt ein wiederholtes Muster: fp32 besser, 4-Bit-Preis höher. Eine Erklärung ist offen. Eine Vermutung: Ohne Anlauf bremsen die frühen großen Schritte die Gewichte so, dass mehr Werte im linearen Kern bleiben. Prüfbar mit der D3-Messung (frac_soft, Anhäufung) an V1c-Netzen, beschreibend und billig.

## Grenzen

- **Ränder des Rasters:** BPreCap liegt am unteren Rand (5e-5). In B15 war 2e-5 im Raster und lag schlechter, ohne Anlauf. Mit Anlauf ist 2e-5 nicht geprüft. SELU liegt am oberen Rand (3e-4). Bei 3e-4 brechen WD 0 und 3,0 schon ein, eine noch höhere LR dürfte also kaum helfen, geprüft ist das nicht.
- Nur eine Anlauflänge (5 %), nur CIFAR-10, d200, MLPs, TAT-Rezept.
- S2–S5 messen bei SELU Anlauf plus neue Wahl, nicht den Anlauf allein.
- SELU mit der Gauß-artigen Init des TAT-Rezepts, nicht mit LeCun-Init.

Sicherung: `../../sicherung/v1c_2026-11-03/` mit `SHA256.txt`.
