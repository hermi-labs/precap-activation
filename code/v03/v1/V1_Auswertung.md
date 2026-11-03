# V1 – Auswertung: LR-Anlauf statt kleiner LR (CIFAR-10, d100)

Stand: 03.11.2026, nachts. Auswertung strikt nach `V1_Praeregistrierung.md` (eingefroren 02.11., 22:21). PreCap und BPreCap sind Aarons Ideen. Lauf und Auswertung mit Claude, in Aarons Abwesenheit.

## Kurz

- **Nach Präregistrierung: Erfolg.** Auch wenn alle Formen mit Anlauf trainieren, ist BPreCap bei 4 Bit (w8a4a, frühes Stoppen) besser als TAT: **−8,82 pp, 12/0, p = 0,0005.**
- **Der Einbruch bei 2e-4 ist mit Anlauf weg,** in beiden Tuning-Seeds und bei jedem WD. Bei WD 0,3 liegt BPreCap bei 45,9/45,6 % Val-Fehler, in B10 waren es 84,0/81,7 %.
- **Bei 5e-4 bricht BPreCap weiter ein,** mit 76–87 % bei jedem WD. TAT wird bei 5e-4 ebenfalls deutlich schlechter (58–78 %). Die Grenze rückt also nach oben, sie verschwindet nicht.
- **Das Tuning wählt für BPreCap wieder 1e-4 / WD 0,3,** mit 44,7 % Val-Fehler gegen 45,75 % bei 2e-4. Damit greift die Deutungsregel „Erfolg, Wahl bleibt 1e-4“: Der Vorteil hält, und der Anlauf verhindert den Einbruch. Eine größere LR bringt BPreCap aber nichts.
- **Der Anlauf hilft TAT bei 4 Bit mehr als BPreCap:**
  - TAT: −2,12 pp (12/0);
  - BPreCap: +0,31 pp (unklar).
  
  Der Vorsprung schrumpft deshalb von −11,67 pp (B10) auf −8,82 pp.
- **In fp32 ist BPreCap mit Anlauf jetzt vorn:** −0,75 pp gegen TAT (11/1). In B10 waren beide gleichauf.
- **Kontrolle bitgleich,** 0 Ausfälle, alle Hashes OK.

## Integrität

| Prüfung | Ergebnis |
|---|---|
| Läufe | 54 Tuning + 38 Bestätigung = 92 von 92 ok |
| Kontrolle (Kosinus ohne Anlauf, Seed 42168, kL2B_tr 1e-4/0,3 und tat 2e-4/0) | **bitgleich** zur B10-Bestätigung, alle Kennzahlen an allen Messpunkten |
| Hash-Prüfung | 15 von 15 OK (Skripte, Tests, Präregistrierung, `src/precap`, CIFAR-10-Dateien) |
| Vorab-Tests | Selbsttest, Mechanik, Auswertung an künstlichen Daten und Rauchtest ok |
| Laufzeit | 02.11., 22:25 bis 03.11., 00:47, 2 Worker, M1 Pro am Netzteil |

Befehl: `.venv/bin/python -I studien/v1/v1_analyze.py studien/v1 studien/b10`. Die Ausgabe steht in `logs/analyse_ausgabe.txt`, das Ergebnis in `analysis/v1_result.json`.

## Tuning (Val-Fehler fp32 am Ende, Seeds 42000/42001)

| Form | LR | WD 0 | WD 0,3 | WD 3,0 |
|---|---|---|---|---|
| kL2B_tr | 1e-4 | 47,2 / 47,1 | **44,8 / 44,6** | 54,8 / 53,4 |
| | 2e-4 | 48,7 / 48,8 | 45,9 / 45,6 | 57,0 / 56,7 |
| | 5e-4 | 86,7 / 84,6 | 80,9 / 81,7 | 76,3 / 76,3 |
| LB_tr | 1e-4 | 49,4 / 48,8 | **46,6 / 46,3** | 54,9 / 54,9 |
| | 2e-4 | 86,9 / 83,3 | 48,7 / 48,4 | 56,8 / 58,6 |
| | 5e-4 | 83,8 / 85,6 | 85,9 / 79,6 | 76,9 / 78,6 |
| TAT | 1e-4 | 47,1 / 48,0 | 46,8 / 47,0 | 47,8 / 48,2 |
| | 2e-4 | **44,9 / 45,7** | 45,6 / 45,9 | 64,6 / 66,5 |
| | 5e-4 | 71,7 / 58,5 | 73,8 / 74,5 | 75,3 / 78,3 |

**Zum Vergleich B10 ohne Anlauf, 2e-4:**

| Form | WD 0 | WD 0,3 | WD 3,0 |
|---|---|---|---|
| kL2B_tr | 85,9 / 85,1 | 84,0 / 81,7 | 56,2 / 58,6 |
| LB_tr | 85,5 / 86,3 | 86,4 / 81,7 | 63,6 / 73,8 |
| TAT | 45,7 / 46,5 | 70,2 / 58,0 | 74,6 / 71,8 |

**Wahl** (fett, nach der Regel aus B10):
- kL2B_tr: 1e-4 / 0,3;
- LB_tr: 1e-4 / 0,3;
- TAT: 2e-4 / 0, wie in B10.

**Lage am Rand:** BPreCap und LB liegen mit 1e-4 am **unteren** Rand des neuen Rasters. Das 5e-5 aus B10 fehlt, dort lag in B10 aber keine Wahl.

**Weitere Beobachtungen:**
- LB bricht mit Anlauf bei 2e-4 nur noch ohne WD ein (86,9/83,3 %). BPreCap bricht bei 2e-4 bei keinem WD mehr ein.
- TAT profitiert mit Anlauf bei WD 0,3: 45,6/45,9 % gegen 70,2/58,0 % in B10.

## Primärtest

| Vergleich | Median | S/N | p (exakt, zweiseitig) | Urteil |
|---|---|---|---|---|
| kL2B_tr − TAT, w8a4a, frühes Stoppen | **−8,82 pp** | 12/0 | 0,0005 | **besser → Erfolg** |

Die Differenzen je Seed liegen zwischen −6,78 und −9,98 pp.

## Sekundär (Holm über 8)

| Nr. | Vergleich | Median | S/N | Holm-p | Urteil |
|---|---|---|---|---|---|
| S1 | kL2B_tr − TAT, fp32, ES | −0,75 | 11/1 | 0,010 | besser |
| S2 | kL2B_tr V1 − B10, fp32, ES | −0,85 | 11/1 | 0,004 | besser |
| S3 | kL2B_tr V1 − B10, w8a4a, ES | +0,31 | 4/8 | 0,052 | unklar |
| S4 | TAT V1 − B10, fp32, ES | −0,39 | 10/2 | 0,019 | besser |
| S5 | TAT V1 − B10, w8a4a, ES | **−2,12** | 12/0 | 0,004 | besser |
| S6 | LB_tr − TAT, w8a4a, ES | −8,22 | 12/0 | 0,004 | besser |
| S7 | kL2B_tr − LB_tr, fp32, ES | −1,89 | 12/0 | 0,004 | besser |
| S8 | kL2B_tr − TAT, w8a4a, Ende 6880 | −9,55 | 12/0 | 0,004 | besser |

**Zu S2 bis S5:** Diese Vergleiche messen das ganze Paket aus Anlauf und neuem Tuning gegen B10, gepaart über den Seed. Bei BPreCap und TAT ist die gewählte LR/WD dieselbe wie in B10. Hier misst der Vergleich deshalb praktisch den Anlauf allein.

## Mediane am Stopp-Punkt (Testfehler in %)

| Form | LR / WD | Stopp-Punkte | fp32 | w8a4 | w8a4a | w8a3 | 4-Bit-Preis (w8a4a − fp32) | Bereich |
|---|---|---|---|---|---|---|---|---|
| kL2B_tr | 1e-4 / 0,3 | 6020–6880 | **45,7** | 51,2 | **50,0** | 68,6 | +4,2 | 3,0 |
| LB_tr | 1e-4 / 0,3 | 6020–6880 | 47,5 | **50,5** | 50,5 | **61,6** | **+3,2** | 2,22 |
| TAT | 2e-4 / 0 | 6020–6880 | 46,3 | 63,2 | 58,8 | 84,8 | +12,6 | 22,5 |

**Zum Vergleich B10:**

| Form | fp32 | w8a4a | 4-Bit-Preis |
|---|---|---|---|
| kL2B_tr | 46,5 | 49,6 | +3,0 |
| TAT | 46,6 | 61,2 | +14,6 |

- **BPreCap:** Mit Anlauf wird fp32 besser, w8a4a bleibt fast gleich. Der 4-Bit-Preis steigt deshalb von +3,0 auf +4,2.
- **TAT:** Der Bereich bleibt groß (22–28), der Preis sinkt von +14,6 auf +12,6.

**Verlauf (Mediane fp32 / w8a4a / Bereich):**

| Schritt | kL2B_tr | LB_tr | TAT |
|---|---|---|---|
| 860 | 55,5 / 58,6 / 3,0 | 57,6 / 59,5 / 2,22 | 60,3 / 63,7 / 12,8 |
| 2580 | 50,5 / 53,3 / 3,0 | 52,9 / 54,4 / 2,22 | 54,5 / 60,2 / 25,6 |
| 4300 | 47,9 / 50,8 / 3,0 | 49,3 / 51,2 / 2,22 | 48,5 / 57,1 / 28,0 |
| 6880 | 45,7 / 50,0 / 3,0 | 47,5 / 50,5 / 2,22 | 46,2 / 59,2 / 21,9 |

## Deutung nach den Vorab-Regeln

**Zutreffende Regel:** „Erfolg, Wahl für BPreCap bleibt 1e-4.“ Der 4-Bit-Vorteil gegen TAT hält auch, wenn TAT den gleichen Anlauf bekommt. Der Anlauf verhindert den Einbruch bei 2e-4. Eine größere LR bringt BPreCap im Tuning aber nichts, 2e-4 ist um etwa 1 pp schlechter.

**Ehrliche Einordnung dazu:**
1. **Die v0.3-Grenze war schärfer formuliert, als sie ist.** Sie lautet „bei 2·10⁻⁴ divergiert BPreCap“. Ohne Anlauf stimmt das, mit 5 % Anlauf nicht mehr. Bei 5e-4 bricht BPreCap auch mit Anlauf ein. TAT verliert dort ebenfalls stark (58–78 %). Ab 5e-4 ist es also kein Sonderproblem von BPreCap mehr, aber BPreCap bricht dort härter ein.
2. **Der Vorsprung bei 4 Bit ist kleiner als in B10:** −8,8 statt −11,7 pp. TAT gewinnt durch den Anlauf 2,1 pp bei 4 Bit, BPreCap nichts.
3. **Der Einwand aus B10 bleibt:** Die Größe des Vorsprungs hängt an der gewählten TAT-Einstellung. Mit Anlauf liegt TAT bei 2e-4/WD 0,3 im Tuning fast gleichauf mit der Wahl (45,75 gegen 45,3 %). Wie groß der 4-Bit-Abstand dort wäre, misst V1 nicht. B9 (TAT mit WD) zeigte −3,8 pp. Die Richtung ist belastbar, die Größe hängt am Rezept.
4. **In fp32 ist BPreCap mit Anlauf knapp vorn** (−0,75 pp), vorher gleichauf.

## Abgleich mit den Vorab-Erwartungen

| Größe | Erwartung | Ergebnis |
|---|---|---|
| kein Einbruch bei 2e-4 mit Anlauf | 0,8 | ja, bei jedem WD |
| Einbruch bei 5e-4 | 0,5 | ja, bei jedem WD |
| Primär „besser“ | 0,65 | ja |
| S2 Anlauf verbessert BPreCap in fp32 | 0,3 | ja (−0,85 pp) |
| S4 Anlauf verbessert TAT in fp32 | 0,4 | ja (−0,39 pp) |
| S5 Anlauf senkt TATs 4-Bit-Fehler | 0,4 | ja (−2,12 pp) |
| Kontrolle bitgleich | 0,95 | ja |

## Grenzen

- Eine Anlauflänge (5 %), nicht gesucht. Nur CIFAR-10, d100, TAT-Rezept, MLPs, 2 Tuning-Seeds.
- BPreCap und LB wählen den unteren Rand 1e-4. Ob 5e-5 mit Anlauf besser wäre, ist offen. In B10 war es schlechter.
- Zwischen 2e-4 und 5e-4 gibt es keinen Messpunkt. Wo genau die Grenze mit Anlauf liegt, ist offen.
- Die Größe des 4-Bit-Vorteils hängt an der TAT-Einstellung (siehe oben, Punkt 3).

## Folge für v0.4 (Vorschlag)

**Neue Formulierung der LR-Grenze (Entwurf):**

> „Ohne LR-Anlauf divergiert BPreCap bei 2·10⁻⁴. Mit einem linearen Anlauf über 5 % der Schritte trainiert es dort normal, bei 5·10⁻⁴ nicht mehr (TAT verliert dort ebenfalls). Der 4-Bit-Vorteil gegen TAT hält mit Anlauf für beide (−8,8 pp, 12/0), ist aber kleiner als ohne (−11,7 pp).“

**Folgestudie V1b (erledigt, 03.11., beschreibend):** Grenze mit Anlauf bei 3e-4, 10 % Anlauf ändert nichts, 5e-5 ist schlechter als 1e-4. Die LR-Aussage für v0.4 steht dort in neuer Fassung: `studien/v1b/V1b_Auswertung.md`.

**Vertrauen:**
- 0,9, dass der Anlauf den Einbruch bei 2e-4 behebt (2 Tuning-Seeds bei jedem WD, dazu D2b);
- 0,9, dass der 4-Bit-Vorteil gegen TAT mit Anlauf in diesem Rezept hält (12/0);
- 0,5 für die Größe des Vorteils, sie hängt an der TAT-Einstellung.
