# B12 – Auswertung: Basislinien ReLU, GELU, SELU im gleichen Rezept und Regime (CIFAR-10, d100)

Stand: 01.11.2026, 02:05. Die Auswertung folgt strikt `B12_Praeregistrierung.md`. PreCap, die Tiefenfamilie und Lagma sind Aarons Ideen. Lauf und Auswertung habe ich mit Claude gemacht.

## Kurz

- **Kein Erfolg nach der Vorab-Regel** (alle drei „besser“). Zwei von drei Primärtests sind „besser“, einer ist „unklar“:
  - gegen **ReLU**: −39,2 pp (12/0)
  - gegen **GELU**: −40,3 pp (12/0)
  - gegen **SELU**: −0,47 pp (9/3, Holm-p = 0,077), **unklar**
- **ReLU und GELU trainieren in diesem tiefen Netz schlecht oder gar nicht:**
  - ReLU: 62,5 % Testfehler in fp32.
  - GELU: 90 %, also Zufallsniveau. Der Trainingsverlust bleibt bei ln 10. Im Tuning lernte GELU teilweise (Median 26 % Genauigkeit), in der Bestätigung bei keinem Seed.
  - Das passt zur Theorie (`math/p12_tiefe_kollaps.py`): Feste Formen kollabieren bzw. werden chaotisch über 100 Schichten, auch mit orthogonaler Init.
- **SELU ist im gleichen Rezept ein starker Gegner:**
  - fp32: 47,5 % gegen 46,5 % bei BPreCap (BPreCap −0,95 pp, 12/0, besser).
  - Bei 4 Bit gleichauf.
  - Sein 4-Bit-Preis ist sogar etwas **kleiner** als der von BPreCap: +2,5 gegen +3,0 pp.
- **Folge für v0.3:** Der kleine 4-Bit-Preis ist **keine Besonderheit von BPreCap.** Er unterscheidet BPreCap von TAT (Leaky ReLU) und ReLU, nicht von SELU. Was BPreCap gegenüber SELU in diesem Aufbau bringt, ist nur ein kleiner fp32-Vorteil (−0,95 pp). Die Kernaussage muss das enthalten.

## Integrität

| Prüfung | Ergebnis |
|---|---|
| Läufe | Tuning 72 von 72 gelaufen, davon 70 ok und 2 divergiert (SELU, LR 5e-4, WD 0 und 0,3, Seed 42000; zählen nach der Regel als 10 %). Bestätigung 37 von 37 ok, 0 Ausfälle |
| Hash-Prüfung | 16 von 16 OK |
| Kontrolle | kL2B_tr, Seed 42168, B10-Wahl: bitgleich zu B10 |
| Schlaf | `pmset -g log`: kein Sleep- oder Wake-Eintrag zwischen 23:40 und 01:58. Am Netzteil |
| Laufzeit | 23:42–01:58 |
| Sicherung | `sicherung/b12_2026-11-01/`, 16 von 16 Dateien OK |
| Unabhängige Nachrechnung | P1, P2, P3, S3 exakt gleich (`analysis_check/nachrechnen.py`) |

Befehl: `.venv/bin/python -I b12_analyze.py ~/…/studien/b12 ~/…/studien/b10`, Ausgabe in `logs/analyse_ausgabe.txt`.

## Tuning (mediane Val-Genauigkeit in % am Ende, 2 Seeds)

| Form | LR-Spitze | WD 0 | WD 0,3 | WD 3,0 |
|---|---|---|---|---|
| ReLU_tr | 5e-5 | 39,0 | 38,7 | 34,1 |
| | 1e-4 | 33,3 | 34,3 | 38,4 |
| | 2e-4 | 35,4 | 34,1 | **39,2** |
| | 5e-4 | 9,5 | 9,5 | 9,5 |
| GELU_tr | 5e-5 | 20,5 | 20,4 | 9,5 |
| | 1e-4 | 24,2 | **26,1** | 9,5 |
| | 2e-4 | 25,3 | 19,2 | 9,5 |
| | 5e-4 | 20,8 | 9,5 | 9,5 |
| SELU_tr | 5e-5 | 43,2 | 44,2 | 51,9 |
| | 1e-4 | 48,3 | 49,8 | 50,1 |
| | 2e-4 | 51,0 | **52,7** | 26,3 |
| | 5e-4 | 12,3 | 12,8 | 9,5 |

- **Wahl:** ReLU 2e-4/3,0 (WD am Rand), GELU 1e-4/0,3, SELU 2e-4/0,3. Keine LR am Rand. Die Vorab-Erwartung „mindestens eine LR am oberen Rand“ (0,35) trifft nicht zu.
- Die Tuning-Werte von ReLU sind über das ganze Raster flach (33–39 %). Ein besseres Rezept für ReLU in 100 Schichten ohne Normierung kennen wir nicht.

## Primärtests (Holm über 3)

| Nr. | Vergleich (w8a4a, ES) | Median | S/N | p (Holm) | Urteil |
|---|---|---|---|---|---|
| P1 | kL2B_tr − ReLU_tr | −39,22 | 12/0 | 0,0015 | besser |
| P2 | kL2B_tr − GELU_tr | −40,34 | 12/0 | 0,0015 | besser |
| P3 | kL2B_tr − SELU_tr | −0,47 | 9/3 | 0,077 | **unklar** |

**Gesamt: kein Erfolg** (Regel: alle drei besser).

## Sekundärtests (Holm über 6)

| Nr. | Vergleich (ES) | Median | S/N | p (Holm) | Urteil |
|---|---|---|---|---|---|
| S1 | − ReLU_tr, fp32 | −15,52 | 12/0 | 0,0029 | besser |
| S2 | − GELU_tr, fp32 | −43,38 | 12/0 | 0,0029 | besser |
| S3 | − SELU_tr, fp32 | −0,95 | 12/0 | 0,0029 | besser |
| S4 | − ReLU_tr, w8a4 | −35,20 | 12/0 | 0,0029 | besser |
| S5 | − GELU_tr, w8a4 | −39,35 | 12/0 | 0,0029 | besser |
| S6 | − SELU_tr, w8a4 | −0,75 | 10/2 | 0,0093 | besser |

## Beschreibend (Mediane am Stopp-Punkt, Testfehler in %)

| Form | LR / WD | fp32 | w8a4 | w8a4a | w8a3 | 4-Bit-Preis | Bereich |
|---|---|---|---|---|---|---|---|
| BPreCap (B10) | 1e-4 / 0,3 | 46,5 | 50,7 | 49,6 | 65,0 | +3,0 | 3,0 |
| SELU_tr | 2e-4 / 0,3 | 47,5 | 51,2 | 50,1 | 62,9 | +2,5 | 3,85 |
| ReLU_tr | 2e-4 / 3,0 | 62,5 | 85,8 | 88,9 | 84,9 | +25,1 | 21,8 |
| GELU_tr | 1e-4 / 0,3 | 90,0 | 90,0 | 90,0 | 90,0 | 0 (Zufall) | 0,01 |

- **SELU bei 3 Bit** ist sogar besser als BPreCap (62,9 gegen 65,0 %, beschreibend).
- **SELU ist unten beschränkt** (−1,76), oben nicht. Sein Bereich am Stopp-Punkt bleibt mit 3,85 klein.
- **Zusammen mit B10b** (TAT mit kleinem Bereich zahlt trotzdem +7 pp) deutet das darauf hin, dass der hohe 4-Bit-Preis eine Eigenschaft **stückweise linearer, positiv homogener Formen** (ReLU, Leaky ReLU) in diesem Aufbau ist, nicht ein Fehlen der oberen Schranke. Das ist eine Vermutung, nicht geprüft. D1 wird um SELU erweitert, um sie zu prüfen (siehe unten).

## Deutung nach den Vorab-Regeln

- **Eine Form bei w4a „unklar“:** wird so berichtet. Die Aussage „bei 4 Bit besser“ gilt in v0.3 **nur gegen ReLU, GELU und TAT, nicht gegen SELU**.
- **fp32 „besser“ für BPreCap gegen alle drei:** Gewöhnliche Formen trainieren in diesem tiefen Netz schlechter. Gegen SELU ist der Abstand klein (−0,95 pp).

## Abgleich mit den Vorab-Erwartungen

| Größe | Erwartung | Ergebnis |
|---|---|---|
| P1 ReLU „besser“ | 0,8 | ja |
| P2 GELU „besser“ | 0,8 | ja |
| P3 SELU „besser“ | 0,7 | nein, unklar |
| alle drei „besser“ (Erfolg) | 0,55 | nein |
| fp32: mindestens eine Form „besser“ für BPreCap | 0,6 | ja, alle drei |
| LR am oberen Rand | 0,35 | nein |
| Kontrolle bitgleich | 0,95 | ja |

## Grenzen

- Nur d100. SELU ist bei d200 in v0.2 (Gauß-Init, kurzes Budget) gescheitert. Ob es im Rezept bei d200 hält, ist offen und für v0.3 die wichtigste Frage zu SELU.
- Nur 2 Tuning-Seeds, ein Datensatz.
- GELU wurde nicht tiefenangepasst. Ein Rezept, mit dem GELU in 100 Schichten ohne Normierung trainiert, haben wir nicht gesucht.
- Die BPreCap-Läufe stammen aus B10 und waren beim Entwurf bekannt.

**Vertrauen:** 0,9 für „gegen ReLU und GELU besser“, 0,8 für „gegen SELU bei 4 Bit kein nachweisbarer Unterschied bei d100“.

## Vorschlag (nicht präregistriert, für Aaron)

- **B15: SELU gegen BPreCap im Rezept bei d200** (und optional CIFAR-100). Wenn SELU bei d200 versagt, trägt die Tiefenkalibrierung von BPreCap. Wenn nicht, ist BPreCap gegenüber SELU in diesem Aufbau kaum besser, und v0.3 muss das so sagen.
