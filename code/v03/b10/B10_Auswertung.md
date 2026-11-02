# B10 – Auswertung: Hält B9 mit praxisüblichem Trainingsregime? (AdamW, Kosinus-LR, Gewichtsabfall, CIFAR-10, d100)

Stand: 31.10.2026, 19:35. Die Auswertung folgt strikt `B10_Praeregistrierung.md`. PreCap, die Tiefenfamilie und Lagma sind Aarons Ideen. Lauf und Auswertung habe ich mit Claude gemacht.

## Kurz

- **Erfolg, und deutlicher als in B9.** Mit AdamW, Kosinus-LR und für jede Form getuntem Gewichtsabfall ist BPreCap bei fairen 4 Bit **11,67 pp besser als TAT** (12/0, p = 0,0005). In fp32 sind beide gleichauf (46,5 % gegen 46,6 %, unklar).
- **Grund wieder der Bereich, diesmal extrem:**
  - Das Tuning wählt für TAT LR 2e-4 ohne Gewichtsabfall. Damit wächst sein Bereich auf bis zu 36 (am Stopp-Punkt 27,9), und 4 Bit kostet TAT +14,6 pp.
  - Bei BPreCap bleibt der Bereich bei 3,0, der 4-Bit-Preis liegt bei +3,0 pp.
- **Das Regime hilft BPreCap:** fp32 am Stopp-Punkt 46,5 % (B9: 49,6 %), w4a 49,6 % (B9: 53,7 %).
- **Wichtige Einschränkung:** Wie groß der Vorsprung ist, hängt daran, welche Einstellung das Tuning für TAT wählt. Das Tuning folgt der präregistrierten Regel (beste fp32-Validierung). TAT lag dabei am oberen LR-Rand, und eine zweitbeste Einstellung (1e-4, WD 0,3) war im Tuning fast gleich gut. Ob TAT dort bei 4 Bit viel besser wäre, misst B10 nicht. Die Richtung des Effekts ist belastbar, die Größe −11,7 pp ist es so nicht. Siehe „Kritische Prüfung“.

## Integrität

| Prüfung | Ergebnis |
|---|---|
| Läufe | Tuning 54 von 54 ok, Bestätigung 38 von 38 ok |
| Hash-Prüfung | 15 von 15 OK |
| Kontrolle | kL2B_tr und TAT, Seed 42156, konstante LR 5e-5, WD 0 über AdamW: alle 7 Kennzahlen an allen 8 Messpunkten bitgleich zu B9 |
| Schlaf | `pmset` meldet zwischen 16:49 und 19:31 keinen Schlaf. Die größte Lücke im Log (384 s) entspricht der Dauer eines Laufs |
| Laufzeit | 16:49–19:30 |
| Sicherung | `sicherung/b10_2026-10-31/`, 15 von 15 Dateien OK |

Befehl: `.venv/bin/python -I b10_analyze.py ~/…/studien/b10 ~/…/studien/b9`, Ergebnis in `analysis/b10_result.json`.

## Tuning (mediane Val-Genauigkeit in % am Ende, 2 Seeds)

| Form | LR-Spitze | WD 0 | WD 0,3 | WD 3,0 |
|---|---|---|---|---|
| kL2B_tr | 5e-5 | 50,5 | 53,9 | 47,2 |
| | 1e-4 | 52,3 | **54,5** | 47,3 |
| | 2e-4 | 14,5 | 17,1 | 42,6 |
| LB_tr | 5e-5 | 48,5 | 51,4 | 45,6 |
| | 1e-4 | 49,7 | **52,0** | 46,1 |
| | 2e-4 | 14,1 | 15,9 | 31,3 |
| TAT | 5e-5 | 49,8 | 49,8 | 51,2 |
| | 1e-4 | 52,2 | 52,7 | 50,5 |
| | 2e-4 | **53,9** | 35,9 | 26,8 |

**Wahl:**
- kL2B_tr: 1e-4 / 0,3
- LB_tr: 1e-4 / 0,3
- TAT: 2e-4 / 0, LR am oberen Rand

Bei 2e-4 divergieren BPreCap und LB ohne starken WD. TAT verträgt 2e-4 nur ohne WD. Mit WD 0,3 ist der Wert dort auffällig schlecht (35,9). Das ist vermutlich instabil, bei nur 2 Seeds.

## Primär und sekundär (d100, 12 Seeds, ES = frühes Stoppen)

| Test | Vergleich | Median | Siege/Niederlagen | p bzw. Holm | Urteil |
|---|---|---|---|---|---|
| **Primär** | kL2B_tr − TAT, w4a, ES | **−11,67 pp** | 12/0 | 0,0005 | **besser** |
| S1 | kL2B_tr − TAT, fp32, ES | −0,21 | 8/4 | 0,47 | unklar |
| S2 | kL2B_tr − TAT, w4, ES | −14,76 | 12/0 | 0,004 | besser |
| S3 | LB_tr − TAT, w4a, ES | −10,52 | 12/0 | 0,004 | besser |
| S4 | kL2B_tr − LB_tr, fp32, ES | −1,93 | 12/0 | 0,004 | besser |
| S5 | kL2B_tr − LB_tr, w4a, ES | −1,36 | 10/2 | 0,007 | besser |
| S6 | kL2B_tr − TAT, w4a, Ende | −12,10 | 12/0 | 0,004 | besser |
| S7 | kL2B_tr − TAT, fp32, Ende | −0,10 | 8/4 | 0,47 | unklar |
| S8 | LB_tr − TAT, fp32, ES | +1,81 | 0/12 | 0,004 | schlechter |

## Mediane am Stopp-Punkt (Testfehler in %)

| Form | LR / WD | Stopp-Punkte | fp32 | w8a4 | w8a4a | w8a3 | 4-Bit-Preis | Bereich am Stopp |
|---|---|---|---|---|---|---|---|---|
| kL2B_tr | 1e-4 / 0,3 | 5160–6880 | **46,5** | **50,7** | **49,6** | 65,0 | +3,0 | 3,0 |
| LB_tr | 1e-4 / 0,3 | 5160–6880 | 48,5 | **50,6** | 50,8 | **58,7** | **+2,3** | 2,22 |
| TAT | 2e-4 / 0 | 6020–6880 | **46,6** | 65,6 | 61,2 | 85,8 | +14,6 | 27,9 |

## Verlauf (Mediane: fp32 / w4a / Bereich)

| Schritt | kL2B_tr | LB_tr | TAT |
|---|---|---|---|
| 860 | 56,6 / 58,3 / 3,0 | 59,1 / 60,2 / 2,22 | 62,2 / 66,0 / 16,8 |
| 1720 | 53,3 / 54,8 | 55,2 / 56,2 | 58,9 / 64,0 / 25,9 |
| 3440 | 49,5 / 51,3 | 51,0 / 52,4 | 52,7 / 61,4 / 36,2 |
| 4300 | 48,1 / 50,1 | 49,8 / 51,1 | 49,2 / 60,4 / 36,3 |
| 5160 | 47,0 / 49,5 | 48,9 / 51,0 | 47,6 / 59,8 / 31,5 |
| 6020 | 46,5 / 49,7 | 48,4 / 51,0 | 46,8 / 61,2 / 28,2 |
| 6880 | 46,6 / 49,6 | 48,4 / 50,7 | 46,7 / 62,1 / 27,4 |

**So lese ich den Verlauf:**
- TATs Bereich steigt schon in den ersten 860 Schritten auf 16,8 und erreicht bei 4300 Schritten 36.
- Mit dem LR-Abfall sinkt er zwar wieder auf 27, aber nie auf das Niveau von B9 (6–8).
- Bei BPreCap und LB werden fp32 und 4 Bit gemeinsam besser, der Bereich bleibt fest.

## Kritische Prüfung

1. **Hängt die Größe am Tuning von TAT?** Ja, wahrscheinlich.
   - TAT bekam 2e-4 ohne WD, weil diese Einstellung im Tuning die beste fp32-Validierung hatte (53,9 gegen 52,7 bei 1e-4/WD 0,3; 2 Seeds).
   - Mit hoher LR und ohne WD wachsen die Gewichte, und damit wächst der Bereich.
   - In B9 (5e-5, Adam) lag der Bereich von TAT bei 6–8, der Vorsprung bei −3,8 pp.
   - **Folge:** Der Vorsprung liegt je nach TAT-Einstellung irgendwo zwischen −3,8 pp (B9) und −11,7 pp (B10). Die Richtung ist in beiden Studien gleich (je 12/0). Die Größe sollte in v0.3 nicht als „−11,7 pp“ ohne diesen Kontext stehen.
2. **Ist die Regel unfair gegen TAT?** Sie ist für alle Formen gleich und entspricht der Praxis: Man tunt in fp32 und quantisiert danach. Ein Gegner wird aber sagen: „Tune TAT for 4-bit, or pick a config with weight decay.“ Vorschlag: **B10b**, siehe unten.
3. **Orakel-Gegenprobe** (nicht präregistriert, nur beschreibend): Je Lauf der beste Test-w4a über alle 8 Messpunkte. kL2B_tr 49,4, TAT 59,5, Abstand −10,1 pp, 12/0. Der Vorsprung hängt also nicht an der Stopp-Regel.
4. **Kontrolle:** bitgleich zu B9. Der neue Code-Pfad (AdamW, Planer) ist sauber.

## Deutung nach den Vorab-Regeln

- **Erfolg:** Der 4-Bit-Vorteil hält auch mit praxisüblichem Regime. Er hängt nicht an konstanter LR oder fehlendem WD.
- **Regel „Bereich von TAT wächst nicht mehr“:** trifft nicht zu, im Gegenteil. Der Mechanismus aus B9 ist mit diesem Regime stärker statt schwächer.
- **Wahl am Rand:** TATs LR liegt am oberen Rand (2e-4). Eine noch höhere LR könnte TAT in fp32 helfen, bei 4 Bit vermutlich weiter schaden. Grenze genannt. Für B11 wurde das Raster deshalb um 5e-4 erweitert.
- **S4/S5:** Der PreCap-Kern ist auch hier in fp32 (−1,9) und bei w4a (−1,4) besser als der lineare Kern.

## Abgleich mit der Vorab-Erwartung

| Größe | Erwartung | Ergebnis |
|---|---|---|
| Primär „besser“ | 0,55 | eingetreten, viel deutlicher als erwartet |
| Primär „unklar“ | 0,35 | nein |
| Primär „schlechter“ | 0,10 | nein |
| S1 „unklar“ | 0,6 | ja |
| S2 „besser“ | 0,8 | ja |
| Bereich von TAT am Stopp-Punkt < 7,5 | 0,6 | **nein, 27,9** |
| mindestens eine Wahl am Rand | 0,5 | ja (TAT, LR) |
| Kontrolle bitgleich | 0,95 | ja |

Meine Erwartung, dass WD und LR-Abfall den Bereich von TAT bremsen, war falsch. Das Tuning hat für TAT gar keinen WD gewählt.

## Antwort auf die Frage

Ja. Mit praxisüblichem Regime hält der Befund: BPreCap ist in fp32 gleichauf mit TAT und bei 4 Bit klar besser (12/0). Der Vorsprung ist hier größer als in B9, weil die von TAT bevorzugte Einstellung seinen Bereich stark wachsen lässt.

**Vertrauen:** 0,85 für die Richtung. 0,5 für die Größe, die von der TAT-Einstellung abhängt.

## Vorschlag B10b (nicht gestartet)

- **Frage:** Bleibt BPreCap bei 4 Bit besser, wenn TAT mit seiner für 4 Bit günstigsten vernünftigen Einstellung läuft?
- **Aufbau:** TAT mit 1e-4/WD 0,3 und 5e-5/WD 3,0 (die zweit- und drittbesten Tuning-Werte mit Gewichtsabfall), je 12 Seeds 42168–42179. Vergleich gegen die vorhandenen kL2B_tr-Läufe aus B10.
- **Dauer:** etwa 1 h.
- **Wichtig:** Damit wählen wir die Einstellung für TAT gezielt zu seinen Gunsten. Das macht den Vergleich härter für BPreCap und gehört deshalb in v0.3.

## Grenzen

- 2 Tuning-Seeds. TATs LR am Rand.
- Gewichtsabfall wirkt auf alle Parameter. Kosinus ohne Anlauf.
- Nur d100, CIFAR-10, MLP, Quantisierung nach dem Training.
