# B15 – Auswertung: BPreCap gegen SELU im TAT-Rezept bei d200 (CIFAR-10)

Stand: 01.11.2026, 13:35. Die Auswertung folgt strikt `B15_Praeregistrierung.md` samt Nachtrag zum macOS-Update. PreCap, die Tiefenfamilie und Lagma sind Aarons Ideen. Lauf und Auswertung habe ich mit Claude gemacht.

## Kurz

- **Erfolg, deutlich.** Bei 200 Schichten ist BPreCap besser als SELU:
  - 4 Bit (w8a4a): **−8,95 pp**, 12/0, p = 0,0005
  - fp32: **−11,97 pp**, 12/0
  - symmetrische 4 Bit: −8,61 pp, 12/0
- **SELU trainiert bei d200 zweigeteilt:**
  - 5 von 12 Seeds lernen kaum: Der Endverlust liegt bei 2,2–2,3, nahe ln 10, der Endfehler bei 83–86 %.
  - 7 Seeds lernen normal, Endfehler 56–59 %.
  - BPreCap liegt bei allen 12 Seeds bei 46,2–47,0 %. Auch die gut trainierten SELU-Läufe sind etwa 10 pp schlechter.
  - Die kleinste 4-Bit-Differenz über alle Seeds beträgt −7,1 pp. Der Vorsprung hängt also nicht an den gescheiterten SELU-Läufen.
- **Nach der Vorab-Regel** („Primär und S1 besser“): Bei größerer Tiefe trägt die Tiefenkalibrierung. BPreCap ist SELU bei d200 in fp32 und bei 4 Bit überlegen. Bei d100 war SELU noch gleichauf (B12).
- **Ehrlich dazu:** BPreCaps eigener 4-Bit-Preis steigt bei d200 auf +7,3 pp (bei d100 +3,0). SELU zahlt nur +3,7 pp, aber auf viel schlechterem fp32-Niveau. Tiefe kostet also auch BPreCap 4-Bit-Genauigkeit, passend zu B8 und zu den Tiefengesetzen bei Tarawneh 2026.

## Integrität

| Prüfung | Ergebnis |
|---|---|
| Läufe | Tuning 48 von 48 ok (keine Divergenz), Bestätigung 25 von 25 ok, 0 Ausfälle |
| Hash-Prüfung | 16 von 16 OK |
| Kontrolle | kL2B_tr d100, Seed 42168, B10-Wahl: **bitgleich zu B10**, auch nach dem macOS-Update (27.0.1 → 27.2) |
| Plattform | Der C4-Neulauf nach dem Update war bitgleich (93/93), siehe `meta_nach_update.txt` |
| Schlaf | `pmset -g log`: kein Sleep- oder Wake-Eintrag seit 10:27. Am Netzteil |
| Laufzeit | 10:27–13:25 |
| Sicherung | `sicherung/b15_2026-11-01/`, 18 von 18 Dateien OK |
| Unabhängige Nachrechnung | Primär, S1, S2, S3 exakt gleich, auch die p-Werte (B15 nutzt bereits den exakten Test) |

Befehl: `.venv/bin/python -I b15_analyze.py ~/…/studien/b15 ~/…/studien/b10`, Ausgabe in `logs/analyse_ausgabe.txt`.

## Tuning (mediane Val-Genauigkeit in % am Ende, 2 Seeds)

| Form | LR-Spitze | WD 0 | WD 0,3 | WD 3,0 |
|---|---|---|---|---|
| BPreCap (kL2B_tr, d200) | 2e-5 | 48,5 | 51,5 | 45,4 |
| | 5e-5 | 52,4 | **53,6** | 45,3 |
| | 1e-4 | 14,5 | 14,4 | 40,3 |
| | 2e-4 | 15,6 | 16,9 | 22,5 |
| SELU_tr (d200) | 2e-5 | 25,7 | 28,6 | 37,8 |
| | 5e-5 | 32,1 | 34,8 | 41,1 |
| | 1e-4 | 35,7 | 40,1 | 9,5 |
| | 2e-4 | 14,4 | **41,7** | 9,5 |

- **Wahl:**
  - BPreCap: 5e-5 / 0,3 (innen).
  - SELU: 2e-4 / 0,3, **am oberen LR-Rand**. Die Nachbarwerte sind 41,1 (5e-5 / 3,0) und 40,1 (1e-4 / 0,3), also fast gleich.
  - Bei d100 divergierte SELU mit 5e-4 (B12). Ein Gewinn durch eine noch höhere LR ist deshalb unwahrscheinlich, aber nicht geprüft.
- **BPreCap bricht bei d200 schon ab LR 1e-4 ein** (14,5 %), bei d100 erst ab 2e-4. Das passt zu „BPreCap verträgt nur kleine LR“ und gehört in die Grenzen von v0.3.

## Primärtest

| Test | Median | S/N | p | Urteil |
|---|---|---|---|---|
| kL2B_tr − SELU_tr, w8a4a, ES, d200 | **−8,95 pp** | 12/0 | 0,0005 | **besser, Erfolg** |

Differenzen (pp): −16,36, −7,20, −26,87, −19,58, −15,27, −8,87, −7,10, −8,16, −7,34, −14,77, −8,95, −8,95.

## Sekundärtests (Holm über 3)

| Nr. | Vergleich (ES) | Median | S/N | p (Holm) | Urteil |
|---|---|---|---|---|---|
| S1 | fp32 | −11,97 | 12/0 | 0,0015 | besser |
| S2 | w8a4 | −8,61 | 12/0 | 0,0015 | besser |
| S3 | w8a3 | +0,34 | 6/6 | 0,62 | unklar |

## Beschreibend (Mediane am Stopp-Punkt, Testfehler in %)

| Form | fp32 | w8a4 | w8a4a | w8a3 | 4-Bit-Preis | Bereich am Ende |
|---|---|---|---|---|---|---|
| BPreCap d200 | 46,4 | 55,5 | 53,6 | 80,0 | +7,3 | 3,0 |
| SELU d200 | 58,3 | 64,1 | 62,7 | 79,7 | +3,7 | 2,37 |

- **Stopp-Punkte:** BPreCap stoppt spät (6020–6880). SELU stoppt bei 5 Seeds früh (860–2580), das sind die Seeds, die nicht weiterlernen.
- **3 Bit** ist bei d200 für beide fast unbrauchbar (80 %).
- **Vergleich mit d100:**
  - BPreCap fp32 ist gleich (46,4 % gegen 46,5 % in B10).
  - BPreCap w4a ist schlechter (53,6 gegen 49,6 %).
  - SELU fp32 fällt von 47,5 % (d100) auf 58,3 %.

## Deutung nach den Vorab-Regeln

- **Primär „besser“ und S1 „besser“:** Bei größerer Tiefe trägt die Tiefenkalibrierung. BPreCap ist SELU bei d200 in fp32 und bei 4 Bit überlegen.
- **Zusammen mit B12:** Bei d100 ist SELU ein gleichwertiger Gegner, bei d200 nicht mehr.
  - Das ist genau der Unterschied, den die Theorie vorhersagt: festes χ1 = 1,07 gegen 1 + 2/L.
  - Der Gradient wächst bei der Init etwa 970-fach gegen etwa 2,7-fach.
  - Für v0.3 ist das die Antwort auf „why not just SELU?“.

## Abgleich mit den Vorab-Erwartungen

| Größe | Erwartung | Ergebnis |
|---|---|---|
| S1 „besser“ | 0,6 | ja |
| Primär „besser“ | 0,5 | ja |
| SELU divergiert in mindestens einer Tuning-Kombination | 0,8 | nein (keine Divergenz, aber 5 von 12 Bestätigungsläufen lernen kaum) |
| Wahl mindestens einer Form am LR-Rand | 0,4 | ja (SELU) |
| Kontrolle bitgleich | 0,95 | ja |

## Grenzen

- Nur 2 Tuning-Seeds. SELU am oberen LR-Rand, die Nachbarn sind fast gleich gut.
- Nur CIFAR-10, nur d200 als zweite Tiefe.
- BPreCaps 4-Bit-Preis steigt mit der Tiefe (+7,3 pp bei d200).
- Ob SELU mit einem anderen Rezept (z. B. Anlauf der LR, anderer Gain) bei d200 stabil trainiert, ist nicht geprüft.

**Vertrauen:** 0,9 für „BPreCap bei d200 in fp32 und 4 Bit besser als SELU im gleichen Rezept“. 0,7 für die Deutung „wegen der Tiefenkalibrierung“: plausibel und vorhergesagt, aber nicht isoliert getestet.
