# B14 – Auswertung: Feste weiche Hülle gegen lernbaren harten Clip (PACT-Stil) im TAT-Rezept (CIFAR-10, d100)

Stand: 01.11.2026, 06:05. Die Auswertung folgt strikt `B14_Praeregistrierung.md`. PreCap, die Tiefenfamilie und Lagma sind Aarons Ideen. Lauf und Auswertung habe ich mit Claude gemacht.

## Kurz

- **Primär unklar.** BPreCap (fest, weich) gegen denselben Kern mit lernbarem hartem Clip: +0,04 pp bei w8a4a, 6/6, p = 1,0.
- **Linearer Kern:** die feste weiche Hülle ist klar besser als der lernbare harte Clip, w8a4a −4,70 pp, w8a4 −5,59 pp (je 12/0), fp32 gleich.
- **Das Lernen der Grenze ändert nichts gegenüber dem festen harten Clip aus B13** (S6 +0,21, S7 −0,16, beide unklar).
- **Entscheidender Befund:** Die Grenze wurde **nicht durch den Gradienten gelernt**. Ihre Bewegung erklärt sich vollständig durch den entkoppelten Gewichtsabfall von AdamW:
  - BPreCap-Kern: von 3,0 auf 2,711. Reiner WD-Abfall 3 · exp(−0,3 · 3440 · 1e-4) = 2,706.
  - linearer Kern: von 2,299 auf 2,180. Reiner WD-Abfall 2,183.
  - Der Gradient trägt also höchstens etwa ±0,005 bei. Ohne WD bleibt α beim Startwert (3,002–3,006 im Tuning).
- **Folge:** B14 beantwortet die Frage „why not PACT?“ **nicht**. Gemessen wurde ein harter Clip, dessen Grenze durch den WD leicht schrumpft. Diese Grenze war in der Präregistrierung vorab genannt („Beweglichkeit der Grenze unter Adam“). Sie ist hier voll eingetreten, sogar stärker als erwartet.
- Für v0.3 heißt das: „Ein gelernter Clip im PACT-Stil ist mit Adam/AdamW in diesem Aufbau praktisch ein fester Clip und bringt nichts. Ein echter PACT-Vergleich (eigene LR für α oder SGD, Quantisierung während des Trainings) fehlt.“

## Integrität

| Prüfung | Ergebnis |
|---|---|
| Läufe | Tuning 48 von 48 ok, Bestätigung 25 von 25 ok, 0 Ausfälle |
| Hash-Prüfung | 16 von 16 OK |
| Kontrolle | kL2B_tr, Seed 42168, B10-Wahl: bitgleich zu B10 |
| Schlaf | `pmset -g log`: kein Sleep- oder Wake-Eintrag zwischen 03:50 und 05:57. Am Netzteil |
| Laufzeit | 03:50–05:57 |
| Sicherung | `sicherung/b14_2026-11-01/`, 16 von 16 Dateien OK |
| Unabhängige Nachrechnung | P1, S1, S6, S7 gleich. p für S6 0,258 statt 0,266 (gleiche Beträge), Urteil gleich |

Befehl: `.venv/bin/python -I b14_analyze.py ~/…/studien/b14 ~/…/studien/b10 ~/…/studien/b13`, Ausgabe in `logs/analyse_ausgabe.txt`.

## Tuning (mediane Val-Genauigkeit in % am Ende, 2 Seeds)

| Form | LR-Spitze | WD 0 | WD 0,3 | WD 3,0 |
|---|---|---|---|---|
| kL2Bp_tr | 5e-5 | 51,3 | 53,2 | 46,1 |
| | 1e-4 | 52,6 | **54,1** | 46,9 |
| | 2e-4 | 15,6 | 44,0 | 42,2 |
| | 5e-4 | 15,7 | 15,5 | 9,5 |
| LBp_tr | 5e-5 | 49,6 | **52,2** | 45,5 |
| | 1e-4 | 49,7 | 51,1 | 46,2 |
| | 2e-4 | 13,5 | 14,1 | 32,0 |
| | 5e-4 | 14,3 | 16,3 | 9,5 |

- **Wahl:** kL2Bp_tr 1e-4/0,3, LBp_tr 5e-5/0,3 (LR am unteren Rand). Die Vorab-Erwartung „WD > 0 bei mindestens einer PACT-Form“ (0,7) trifft zu, ist aber nach dem Befund oben kein Zeichen für gelernte Grenzen.
- **Gelernte Grenze im Tuning (Seed 42000, Median über die Schichten):** Sie folgt dem WD und nicht dem Gradienten. Ohne WD bleibt sie bei 3,00, mit WD 3,0 und LR 5e-4 fällt sie auf 0,03.

## Primärtest

| Test | Median | S/N | p | Urteil |
|---|---|---|---|---|
| kL2B_tr (B10) − kL2Bp_tr, w8a4a, ES | +0,04 | 6/6 | 1,0 | **unklar** |

## Sekundärtests (Holm über 7)

| Nr. | Vergleich (ES) | Median | S/N | p (Holm) | Urteil |
|---|---|---|---|---|---|
| S1 | LB_tr − LBp_tr, w8a4a | −4,70 | 12/0 | 0,0034 | besser |
| S2 | kL2B_tr − kL2Bp_tr, fp32 | −0,32 | 9/3 | 0,21 | unklar |
| S3 | LB_tr − LBp_tr, fp32 | −0,05 | 7/5 | 1,0 | unklar |
| S4 | kL2B_tr − kL2Bp_tr, w8a4 | −0,33 | 8/4 | 1,0 | unklar |
| S5 | LB_tr − LBp_tr, w8a4 | −5,59 | 12/0 | 0,0034 | besser |
| S6 | kL2Bp_tr − kL2Bh_tr (B13), w8a4a | +0,21 | 4/8 | 1,0 | unklar |
| S7 | LBp_tr − LBh_tr (B13), w8a4a | −0,16 | 7/5 | 1,0 | unklar |

## Beschreibend (Mediane am Stopp-Punkt, Testfehler in %)

| Form | fp32 | w8a4 | w8a4a | w8a3 | 4-Bit-Preis | gelernte Grenze am Ende (Median, Spanne über Schichten) |
|---|---|---|---|---|---|---|
| kL2B_tr (fest, weich) | 46,5 | 50,7 | 49,6 | 65,0 | +3,0 | 3,0 fest |
| kL2Bh_tr (fest, hart, B13) | 46,9 | 50,5 | 49,5 | 64,6 | +2,7 | 3,0 fest |
| kL2Bp_tr (lernbar, hart) | 47,0 | 50,8 | 49,6 | 64,4 | +2,8 | 2,711 (2,69–2,82) |
| LB_tr (fest, weich) | 48,5 | 50,6 | 50,8 | 58,7 | +2,3 | 2,22 fest |
| LBh_tr (fest, hart, B13) | 48,3 | 55,8 | 55,9 | 75,8 | +7,7 | 2,30 fest |
| LBp_tr (lernbar, hart) | 48,6 | 56,3 | 55,9 | 76,2 | +7,4 | 2,180 (2,17–2,21) |

## Deutung nach den Vorab-Regeln

- **Primär „unklar“:** Bei 4 Bit gleichauf mit einem „gelernten“ Clip. Nach der Vorab-Regel bleibt als Vorteil von BPreCap nur: kein zusätzlicher Parameter, kein Tuning der Strafe.
- **Zusätzlich, und wichtiger:** Die Grenze wurde nicht gelernt. Die Regel setzt voraus, dass ein PACT-artiges Lernen stattfindet. Das war nicht der Fall. B14 ist deshalb **kein Test von PACT**, sondern eine Wiederholung von B13 mit WD-geschrumpfter Grenze. Die Ergebnisse decken sich mit B13.
- **S6 „besser“ (Lernen hilft der harten Form):** nein.

## Abgleich mit den Vorab-Erwartungen

| Größe | Erwartung | Ergebnis |
|---|---|---|
| Primär „besser“ | 0,4 | nein |
| Primär „unklar“ | 0,45 | ja |
| S6 „besser“ | 0,4 | nein |
| WD-Wahl > 0 bei mindestens einer PACT-Form | 0,7 | ja (beide), aber siehe oben |
| Kontrolle bitgleich | 0,95 | ja |

## Grenzen und was fehlt

- Unter Adam/AdamW mit gemeinsamer LR ist eine einzelne Skalar-Grenze je Schicht praktisch unbeweglich. Ihre Gradienten sind klein, und die Schrittweite von Adam ist auf etwa die LR begrenzt.
- **Ein fairer PACT-Vergleich bräuchte:**
  - eine eigene, deutlich größere LR für α oder SGD wie im Original,
  - die L2-Strafe von PACT statt WD,
  - Quantisierung während des Trainings (QAT), für die PACT gebaut ist.
  - Das ist eine eigene Studie, kein Zwischentest.
- Nur d100, CIFAR-10, 2 Tuning-Seeds.

**Vertrauen:** 0,9, dass die Grenze in B14 nicht gelernt wurde (Rechnung stimmt auf ±0,005). 0,85 für „beim linearen Kern ist fest-weich besser als der hier gemessene harte Clip“. Zur Frage „fest-weich gegen echtes PACT“ ist keine Aussage möglich.
