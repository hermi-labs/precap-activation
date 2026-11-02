# B10b – Präregistrierung: Gegenprobe zu B10 mit TAT in den Einstellungen mit Gewichtsabfall (CIFAR-10, d100, privat)

Stand: 31.10.2026, 19:45, vor dem Start. PreCap, die Tiefenfamilie und Lagma sind Aarons Ideen. BPreCap (= kL2B) ist ein Arbeitsname. Das Design habe ich mit Claude entworfen. Aaron hat B10b am 31.10. um etwa 19:35 freigegeben („ja mach am besten b10b“), Start nach B11, nichts veröffentlichen.

## Ausgangslage

- **B10:** BPreCap ist bei w4a 11,67 pp besser als TAT (12/0), fp32 gleichauf.
- Das Tuning (beste fp32-Validierung) wählte für TAT LR 2e-4 ohne Gewichtsabfall, am oberen LR-Rand. Damit wuchs TATs Bereich bis 36, der 4-Bit-Preis lag bei +14,6 pp.
- **Einwand** (Aaron und ich): Die Größe −11,7 pp hängt an dieser Einstellung. Mit Gewichtsabfall wäre TAT bei 4 Bit vermutlich viel besser. Im Tuning lag TAT mit 1e-4/WD 0,3 in fp32 fast gleich auf (52,7 % gegen 53,9 % Val).

## Frage

Bleibt BPreCap bei fairen 4 Bit besser als TAT, wenn TAT mit seiner besten Tuning-Einstellung **mit** Gewichtsabfall trainiert wird? Diese Einstellung wählen wir bewusst zu TATs Gunsten.

## Eine Änderung gegenüber B10: TATs Hyperparameter

- **TAT a:** LR 1e-4, WD 0,3. Die beste B10-Tuning-Kombination mit WD > 0 (52,7 % Val).
- **TAT b:** LR 5e-5, WD 3,0. Die zweitbeste mit WD > 0 (51,2 % Val).
- Die Regel „die beiden besten mit WD > 0“ steht vor dem Start fest. Gelesen habe ich dafür nur die B10-Tuningdaten. Die B10-Bestätigungsdaten der Gegner (kL2B_tr) waren zu dem Zeitpunkt allerdings bekannt. Das nenne ich offen.
- **Vergleichsform:** kL2B_tr aus B10 (1e-4, WD 0,3), dieselben 12 Seeds 42168–42179, gepaart. Der Code ist identisch zu B10: Der Mechanik-Test belegt die Bitgleichheit je TAT-Einstellung.
- **Konstanten wie B10:** Regime (AdamW, Kosinus auf 0, ohne Anlauf), Netz, Daten, Rezept, Messpunkte, frühes Stoppen, Quantisierung.

## Läufe

- 2 TAT-Einstellungen × 12 Seeds = 24.
- **Kontrolle:** TAT mit der B10-Wahl (2e-4, 0), Seed 42168. Alle Kennzahlen an allen Messpunkten müssen bitgleich zu B10 sein.
- Zusammen 25 Läufe, etwa 1 h. Start automatisch nach B11, nur am Netzteil.

## Primärtest

**kL2B_tr (B10) − TAT a bei w8a4a_best am Punkt des frühen Stoppens.** Wilcoxon zweiseitig exakt, gepaart über Seeds, α = 0,05.

**Erfolg:** p < 0,05 und Median < 0.

## Sekundär (Holm über 7)

| Nr. | Vergleich |
|---|---|
| S1 | kL2B_tr − TAT b, w8a4a, ES |
| S2 | kL2B_tr − TAT a, fp32, ES |
| S3 | kL2B_tr − TAT b, fp32, ES |
| S4 | kL2B_tr − TAT a, w8a4, ES |
| S5 | kL2B_tr − TAT b, w8a4, ES |
| S6 | TAT a − TAT (B10-Wahl), fp32, ES. Verliert TAT mit WD in fp32? |
| S7 | TAT a − TAT (B10-Wahl), w8a4a, ES. Gewinnt TAT mit WD bei 4 Bit? |

**Beschreibend:** Bereich am Stopp-Punkt und Verlauf je TAT-Einstellung, 4-Bit-Preis, Stopp-Punkte.

## Deutungsregeln (vorab)

| Ergebnis | Deutung |
|---|---|
| Erfolg | Der 4-Bit-Vorteil bleibt auch gegen TAT mit Gewichtsabfall. Für v0.3 wird der kleinste Vorsprung aus B9, B10 und B10b als vorsichtige Größe genannt |
| Primär „unklar“ | Mit Gewichtsabfall holt TAT bei 4 Bit auf. Die Aussage wird zu „BPreCap braucht keine sorgfältige Wahl von LR und WD, um 4-Bit-tauglich zu sein; TAT schon“. Die Größe aus B10 wird nicht als Vorteil der Form berichtet |
| Primär „schlechter“ | TAT mit Gewichtsabfall ist bei 4 Bit besser. Die Kernaussage von v0.3 muss neu gefasst werden |
| S6 „schlechter“ und S7 „besser“ | Bei TAT gibt es einen Zielkonflikt zwischen fp32 und 4 Bit, der über WD gesteuert wird. BPreCap hat ihn nicht (Bereich fest) |

## Vorab-Erwartung

| Größe | Erwartung |
|---|---|
| Primär „besser“ | 0,6 |
| Primär „unklar“ | 0,3 |
| Primär „schlechter“ | 0,1 |
| Bereich von TAT a am Stopp-Punkt < 15 | 0,6 |
| S7 „besser“ (WD hilft TAT bei 4 Bit) | 0,7 |
| S6 „schlechter“ (WD kostet TAT fp32) | 0,4 |
| Kontrolle bitgleich | 0,95 |

## Grenzen

- Die Vergleichsläufe von kL2B_tr stammen aus B10, ihre Ergebnisse waren beim Entwurf bekannt. Die Regel für TATs Einstellungen hängt aber nur an den Tuningdaten.
- Nur zwei TAT-Einstellungen, kein neues Tuning auf 4 Bit.
- Nur d100, CIFAR-10, Quantisierung nach dem Training.

## Freeze

Eingefroren werden `b10b_run.py`, `b10b_analyze.py`, `test_b10b_mechanik.py`, `test_b10b_analyse.py`, diese Datei, `src/precap/*.py` und die CIFAR-10-Dateien. B10 ist mit eigenen Hashes eingefroren und gesichert (`sicherung/b10_2026-10-31/`).

Vorab gemachte Tests (31.10., 19:40):

- **Selbsttest:** n_confirm 25.
- **Mechanik** (`test_b10b_mechanik.py`): **MECHANIK OK.** Geprüft wird:
  - Form, Init, Daten und Konstanten gleich B10.
  - 25 Läufe, Varianten und Seeds wie geplant, Namen eindeutig, Kontrolle gleich der B10-Wahl.
  - Je TAT-Einstellung ist ein kurzer Lauf bitgleich zum B10-Code, auch an der Zwischenmessung.
- **Auswertung an künstlichen Daten** (`test_b10b_analyse.py`): AUSWERTUNG OK.
  - Erfolg, „schlechter“, Null mit 20 Wiederholungen (1 Fehlalarm).
  - Eine falsche Kontrolle wird erkannt, am Ende wie an einem Messpunkt.
  - Ein Ausfall ohne Messpunkt zählt als 90 %. Der relative Pfad findet B10.
- **Rauchtest:** im Startskript (3 Läufe mit 40 Schritten).
