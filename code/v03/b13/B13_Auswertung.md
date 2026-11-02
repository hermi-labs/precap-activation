# B13 – Auswertung: Weich gegen hart im TAT-Rezept (CIFAR-10, d100)

Stand: 01.11.2026, 03:55. Die Auswertung folgt strikt `B13_Praeregistrierung.md`. PreCap, die Tiefenfamilie und Lagma sind Aarons Ideen. Lauf und Auswertung habe ich mit Claude gemacht.

## Kurz

- **Primär unklar.** Mit PreCap-Kern ist die weiche Hülle bei 4 Bit im Rezept nicht besser als der harte Clip an derselben Grenze: +0,04 pp, 6/6, p = 0,34.
- **Mit linearem Kern ist die weiche Hülle deutlich besser,** bei gleicher fp32-Güte:
  - w8a4a: −4,69 pp (12/0)
  - w8a4: −4,84 pp (12/0)
  - w8a3: −16,96 pp (12/0)
  - fp32: unklar, +0,11
- **Nach der Vorab-Regel** („Primär unklar“) bleibt K1 („weich besser als hart bei gleicher Grenze“) für den PreCap-Kern auf Gauß-Init beschränkt (B4). Für den linearen Kern gilt K1 im Rezept deutlich.
- **Deutung (beschreibend, nicht präregistriert):**
  - Der PreCap-Kern sättigt schon selbst weich (tanh im Inneren). Die Form der Hülle ist dort fast gleichgültig, der harte Clip greift selten: 1,5 % gesättigte Werte gegen 3,4 % bei der weichen Hülle.
  - Beim linearen Kern ist die Hülle die einzige Nichtlinearität. Der harte Clip macht die Form **stückweise linear**, und dann zahlt sie bei 4 Bit +7,7 pp. Das liegt fast genau beim Wert von TAT mit Gewichtsabfall (+7,3, B10b), die weiche Form zahlt +2,3.
  - Das stützt die Vermutung aus B10b und B12: **Stückweise lineare Formen zahlen bei 4 Bit mehr, glatt sättigende Formen wenig**, unabhängig vom Bereich.

## Integrität

| Prüfung | Ergebnis |
|---|---|
| Läufe | Tuning 48 von 48 ok, Bestätigung 25 von 25 ok, 0 Ausfälle |
| Hash-Prüfung | 16 von 16 OK |
| Kontrolle | kL2B_tr, Seed 42168, B10-Wahl: bitgleich zu B10 |
| Schlaf | `pmset -g log`: kein Sleep- oder Wake-Eintrag zwischen 01:58 und 03:46. Am Netzteil |
| Laufzeit | 01:58–03:45 |
| Sicherung | `sicherung/b13_2026-11-01/`, 16 von 16 Dateien OK |
| Unabhängige Nachrechnung | P1, S1, S5, S7: Differenzen, Mediane, S/N gleich. p für P1 0,330 statt 0,339 (gleiche Beträge, wie bei B10), Urteil gleich |

Befehl: `.venv/bin/python -I b13_analyze.py ~/…/studien/b13 ~/…/studien/b10`, Ausgabe in `logs/analyse_ausgabe.txt`.

## Tuning (mediane Val-Genauigkeit in % am Ende, 2 Seeds)

| Form | LR-Spitze | WD 0 | WD 0,3 | WD 3,0 |
|---|---|---|---|---|
| kL2Bh_tr | 5e-5 | 50,6 | 54,1 | 47,2 |
| | 1e-4 | 52,3 | **54,6** | 46,9 |
| | 2e-4 | 15,3 | 16,2 | 43,9 |
| | 5e-4 | 12,8 | 9,5 | 23,2 |
| LBh_tr | 5e-5 | 49,8 | **52,2** | 44,0 |
| | 1e-4 | 49,6 | 51,3 | 45,9 |
| | 2e-4 | 13,8 | 15,7 | 24,8 |
| | 5e-4 | 15,9 | 17,8 | 16,3 |

- **Wahl:** kL2Bh_tr 1e-4/0,3 (wie die weiche Form), LBh_tr 5e-5/0,3 (LR am unteren Rand, knapp vor 1e-4 mit 51,3).
- Auch die harten Formen brechen ab LR 2e-4 ein. Die LR-Empfindlichkeit liegt also nicht an der weichen Hülle.

## Primärtest

| Test | Median | S/N | p | Urteil |
|---|---|---|---|---|
| kL2B_tr − kL2Bh_tr, w8a4a, ES | +0,04 | 6/6 | 0,34 | **unklar** |

## Sekundärtests (Holm über 7)

| Nr. | Vergleich (ES) | Median | S/N | p (Holm) | Urteil |
|---|---|---|---|---|---|
| S1 | LB_tr − LBh_tr, w8a4a | −4,69 | 12/0 | 0,0034 | besser |
| S2 | kL2B_tr − kL2Bh_tr, w8a4 | −0,04 | 6/6 | 1,0 | unklar |
| S3 | LB_tr − LBh_tr, w8a4 | −4,84 | 12/0 | 0,0034 | besser |
| S4 | kL2B_tr − kL2Bh_tr, fp32 | −0,40 | 10/2 | 0,26 | unklar |
| S5 | LB_tr − LBh_tr, fp32 | +0,11 | 4/8 | 1,0 | unklar |
| S6 | kL2B_tr − kL2Bh_tr, w8a3 | +0,16 | 5/7 | 1,0 | unklar |
| S7 | LB_tr − LBh_tr, w8a3 | −16,96 | 12/0 | 0,0034 | besser |

## Beschreibend (Mediane am Stopp-Punkt, Testfehler in %)

| Form | LR / WD | fp32 | w8a4 | w8a4a | w8a3 | 4-Bit-Preis | Sättigung |
|---|---|---|---|---|---|---|---|
| BPreCap weich (B10) | 1e-4 / 0,3 | 46,5 | 50,7 | 49,6 | 65,0 | +3,0 | 3,4 % |
| BPreCap hart | 1e-4 / 0,3 | 46,9 | 50,5 | 49,5 | 64,6 | +2,7 | 1,5 % |
| LB weich (B10) | 1e-4 / 0,3 | 48,5 | – | 50,8 | – | +2,3 | 12,8 % |
| LB hart | 5e-5 / 0,3 | 48,3 | 55,8 | 55,9 | 75,8 | +7,7 | 3,5 % |

Sättigung: Anteil der Kernwerte über ρ·M (weich) bzw. über M (hart) auf den Kalibrierbildern am Ende, Median über die Seeds.

## Deutung nach den Vorab-Regeln

- **Primär „unklar“:** Im Rezept ist für BPreCap kein Vorteil der weichen Hülle gegenüber dem harten Clip nachweisbar. K1 bleibt in v0.3 für den PreCap-Kern auf Gauß-Init beschränkt (B4).
- **S1, S3, S7 „besser“:** Für den linearen Kern gilt K1 im Rezept deutlich, bei 4 und 3 Bit.
- **S4 nicht „schlechter“:** Die weiche Form kostet keine fp32-Güte.

## Abgleich mit den Vorab-Erwartungen

| Größe | Erwartung | Ergebnis |
|---|---|---|
| Primär „besser“ | 0,45 | nein |
| Primär „unklar“ | 0,45 | ja |
| S1 „besser“ | 0,45 | ja |
| S4 „unklar“ | 0,6 | ja |
| Kontrolle bitgleich | 0,95 | ja |

## Grenzen

- Nur 2 Tuning-Seeds. LBh_tr am unteren LR-Rand, LB_tr (B10) bei 1e-4. Im Tuning lagen beide Stufen für LBh fast gleich (52,2 gegen 51,3 %).
- Die weichen Läufe stammen aus B10 und waren beim Entwurf bekannt.
- Die Deutung „stückweise linear zahlt mehr“ ist nachträglich und beschreibend. D1 prüft sie teilweise, ein eigener Test fehlt.

**Vertrauen:** 0,85 für „beim linearen Kern ist weich klar besser“. 0,8 für „beim PreCap-Kern kein Unterschied im Rezept“. 0,5 für die Deutung „stückweise linear zahlt mehr“.
