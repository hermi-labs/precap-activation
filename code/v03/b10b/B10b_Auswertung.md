# B10b – Auswertung: Gegenprobe zu B10 mit TAT in den Einstellungen mit Gewichtsabfall (CIFAR-10, d100)

Stand: 31.10.2026, 23:40. Die Auswertung folgt strikt `B10b_Praeregistrierung.md`. PreCap, die Tiefenfamilie und Lagma sind Aarons Ideen. Lauf und Auswertung habe ich mit Claude gemacht.

## Kurz

- **Erfolg.** Auch gegen TAT mit Gewichtsabfall ist BPreCap bei fairen 4 Bit besser:
  - **TAT a** (LR 1e-4, WD 0,3): **−4,48 pp** (12/0, p = 0,0005).
  - **TAT b** (LR 5e-5, WD 3,0): −6,83 pp (12/0, S1).
- **Gewichtsabfall hilft TAT bei 4 Bit stark:** −6,55 pp gegenüber der B10-Wahl (12/0, S7), in fp32 ohne nachweisbare Kosten (S6 unklar, +0,44).
  - Der B10-Vorsprung von −11,7 pp war also zu gut der Hälfte eine Folge von TATs Einstellung.
- **Nach der Vorab-Regel** wird für v0.3 der kleinste Vorsprung aus B9, B10 und B10b als vorsichtige Größe genannt: **−3,8 pp (B9)**. Gegen TAT mit WD sind es −4,5 pp.
- **In fp32** liegt BPreCap gleichauf mit TAT a (−0,52, 9/3, Holm 0,085, unklar) und vor TAT b (−2,24, 12/0).
- **Wichtiger Nebenbefund (beschreibend, nicht präregistriert):** Mit Gewichtsabfall ist TATs Bereich so klein wie der von BPreCap oder kleiner. Trotzdem zahlt TAT bei 4 Bit mehr als doppelt so viel. **Der absolute Bereich ist also nicht der Mechanismus.** Die Deutung aus B9/B10 („TATs Bereich wächst, deshalb verliert TAT“) ist damit zu einfach und muss für v0.3 korrigiert werden. Siehe „Mechanismus“.

## Integrität

| Prüfung | Ergebnis |
|---|---|
| Läufe | 25 von 25 ok, 0 Ausfälle |
| Hash-Prüfung | 15 von 15 OK |
| Kontrolle | TAT mit der B10-Wahl (2e-4, WD 0), Seed 42168: alle Kennzahlen an allen Messpunkten bitgleich zu B10 |
| Schlaf | `pmset -g log` meldet zwischen 22:49 und 23:37 keinen Sleep- oder Wake-Eintrag. Am Netzteil |
| Laufzeit | 22:49–23:35 |
| Sicherung | `sicherung/b10b_2026-10-31/`, 13 von 13 Dateien OK |
| Unabhängige Nachrechnung | `analysis_check/nachrechnen.py`: Primär, S1, S2, S7 exakt gleich (Differenzen, Median, Siege/Niederlagen, p) |

Befehl: `.venv/bin/python -I b10b_analyze.py ~/…/studien/b10b ~/…/studien/b10`, Ausgabe in `logs/analyse_ausgabe.txt`, Ergebnis in `analysis/b10b_result.json`.

## Primärtest

| Test | Median | S/N | p | Urteil |
|---|---|---|---|---|
| kL2B_tr (B10) − TAT a, w8a4a, ES | **−4,48 pp** | 12/0 | 0,0005 | **besser, Erfolg** |

## Sekundärtests (Holm über 7)

| Nr. | Vergleich | Median | S/N | p (Holm) | Urteil |
|---|---|---|---|---|---|
| S1 | kL2B_tr − TAT b, w8a4a | −6,83 | 12/0 | 0,0034 | besser |
| S2 | kL2B_tr − TAT a, fp32 | −0,52 | 9/3 | 0,085 | unklar |
| S3 | kL2B_tr − TAT b, fp32 | −2,24 | 12/0 | 0,0034 | besser |
| S4 | kL2B_tr − TAT a, w8a4 | −8,13 | 12/0 | 0,0034 | besser |
| S5 | kL2B_tr − TAT b, w8a4 | −11,54 | 12/0 | 0,0034 | besser |
| S6 | TAT a − TAT (B10-Wahl), fp32 | +0,44 | 3/9 | 0,092 | unklar |
| S7 | TAT a − TAT (B10-Wahl), w8a4a | −6,55 | 12/0 | 0,0034 | besser |

## Beschreibend (Mediane am Stopp-Punkt, Testfehler in %)

| Form | LR / WD | fp32 | w8a4 | w8a4a | w8a3 | 4-Bit-Preis | Bereich am Stopp |
|---|---|---|---|---|---|---|---|
| BPreCap (B10) | 1e-4 / 0,3 | 46,5 | 50,7 | 49,6 | 65,0 | +3,0 | 3,0 |
| TAT a | 1e-4 / 0,3 | 47,1 | 58,8 | 54,2 | 85,8 | +7,3 | 3,13 |
| TAT b | 5e-5 / 3,0 | 49,0 | 62,2 | 56,6 | 84,7 | +7,8 | 0,36 |
| TAT (B10-Wahl) | 2e-4 / 0 | 46,6 | 65,6 | 61,2 | 85,8 | +14,6 | 27,9 |

- **Stopp-Punkte:** TAT a stoppt früher (Median 5160) als BPreCap und TAT b (meist 6880).
- **Bei 3 Bit** liegen alle TAT-Varianten bei 85 %, BPreCap bei 65 %.

## Mechanismus (beschreibend, nicht präregistriert)

Am Ende des Trainings, Mediane über 12 Seeds, Bereich = Median über die Schichten der maximalen |Aktivierung| auf den Kalibrierbildern:

| Form | Max | 99,99-%-Quantil | Max / Quantil | 4-Bit-Preis am Ende |
|---|---|---|---|---|
| BPreCap | 3,0 | 3,0 | 1,0 | +2,9 |
| LB | 2,22 | 2,22 | 1,0 | +2,4 |
| TAT a | 2,94 | 2,02 | 1,46 | +9,1 |
| TAT b | 0,36 | 0,29 | 1,20 | +7,8 |
| TAT (B10-Wahl) | 27,4 | 17,1 | 1,60 | +15,5 |

- **Leaky ReLU ist positiv homogen** (f(cx) = c·f(x) für c > 0), und die Quantisierung je Tensor mit Kalibrierung ist skalierungsinvariant. Ein kleinerer absoluter Bereich allein kann TAT bei 4 Bit deshalb nicht helfen. TAT b zeigt das: Bereich 0,36, Preis trotzdem +7,8.
- **Was bei TAT mit dem Preis mitgeht**, ist die Form der Verteilung. Das Verhältnis Max/Quantil sinkt mit dem WD von 1,60 auf 1,20, der Preis von 15,5 auf 7,8.
- **Bei den beschränkten Formen** liegt das Verhältnis bei 1,0: Die Schranke wird erreicht, ein Teil der Einheiten ist gesättigt.
- **Zwei mögliche Erklärungen** für den verbleibenden Abstand (+3 gegen +7 bis +9 pp), beide **nicht geprüft**:
  1. **Form der Verteilung:** Eine beschränkte, teils gesättigte Verteilung nutzt die 16 Stufen gleichmäßiger als eine spitze Verteilung mit Ausläufern.
  2. **Fortpflanzung des Rundungsfehlers über 100 Schichten:** Gesättigte Einheiten haben eine Ableitung nahe 0 und dämpfen den Rundungsfehler der vorigen Schichten. Leaky ReLU ist überall fast linear und reicht ihn ungedämpft weiter.
- **Folge für v0.3:** Die Aussage „der Bereich von TAT wächst, deshalb verliert TAT“ wird ersetzt durch: „Die feste Schranke macht den 4-Bit-Preis klein und unabhängig von LR und WD. Bei TAT hängt er stark davon ab (+7 bis +15 pp). Der absolute Bereich allein erklärt das nicht.“ Eine Diagnose der beiden Erklärungen ist als Zwischentest D1 vorgesehen.

## Deutung nach den Vorab-Regeln

- **Erfolg:** Der 4-Bit-Vorteil bleibt auch gegen TAT mit Gewichtsabfall. Für v0.3 wird der kleinste Vorsprung (B9, −3,8 pp) als vorsichtige Größe genannt.
- **S6 „schlechter“ und S7 „besser“:** nur S7 trifft zu, S6 ist unklar. Die Regel „Zielkonflikt bei TAT, über WD gesteuert“ ist damit **nicht belegt**. Belegt ist nur: WD verbessert TAT bei 4 Bit deutlich, ohne nachweisbaren fp32-Preis.

## Abgleich mit den Vorab-Erwartungen

| Größe | Erwartung | Ergebnis |
|---|---|---|
| Primär „besser“ | 0,6 | ja |
| Bereich von TAT a am Stopp-Punkt < 15 | 0,6 | ja (3,13) |
| S7 „besser“ | 0,7 | ja |
| S6 „schlechter“ | 0,4 | nein, unklar |
| Kontrolle bitgleich | 0,95 | ja |

## Grenzen

- Nur zwei TAT-Einstellungen, ausgewählt nach fp32-Tuning, nicht nach 4 Bit. Eine auf 4 Bit getunte TAT-Einstellung könnte näher kommen.
- Die BPreCap-Läufe stammen aus B10 und waren beim Entwurf bekannt.
- Die Mechanismus-Tabelle ist beschreibend und nachträglich. Gemessen werden nur Max und 99,99-%-Quantil am Ende, keine vollständige Verteilung.
- Nur d100, CIFAR-10, Quantisierung nach dem Training.

**Vertrauen:** 0,85 für den Primärbefund. 0,8, dass der absolute Bereich nicht der Mechanismus ist. 0,4 für jede der beiden Erklärungen einzeln.
