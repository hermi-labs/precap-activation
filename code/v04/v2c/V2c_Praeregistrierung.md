# V2c – Präregistrierung: Kommt der 4-Bit-Vorteil von BPreCap allein aus der Begrenzung?

Stand: 18.11.2026, 17:05, ergänzt 17:55 nach dem Pilot (siehe „Änderung nach dem Pilot“), vor dem Start (privat, M1 Pro). PreCap, BPreCap, die Tiefenfamilie und die Hülle sind Aarons Ideen. Das Design habe ich mit Claude entworfen, Grundlage ist `../v2c_entwurf/V2c_Entwurf.md` (Vorschlag A). Freigabe: Aaron, 18.11. nachmittags („du kannst starten, was du für richtig hältst“) und Mac bis 19.11. 6:30. Eingefroren wird mit dem Start (`meta/frozen_sha256.txt`).

## Frage

Ist BPreCap im V2a2-Netz (CNN, L = 50, ohne Normierung, ohne Residuen, Delta-orthogonal, CIFAR-10) bei 4-Bit-Aktivierungen (w8a4a_best) besser als **tanh**, eine ebenfalls begrenzte Form ohne Tiefenkalibrierung?

## Hintergrund

V2a2 (`../v2a2/V2a2_Auswertung.md`): BPreCap ist bei w8a4a_best besser als SELU (−2,55 pp) und als TAT (−13,73 pp), je 12/0, bei 1,6–3,2 pp Preis in fp32. Offen ist, ob der Vorteil aus der Begrenzung allein kommt. tanh mit Delta-orthogonaler Init ohne Normierung ist die klassische trainierbare begrenzte Form (Xiao et al. 2018, https://arxiv.org/abs/1806.05393).

## Die eine Änderung

Aktivierung tanh statt BPreCap: f(x) = tanh(x), Gewichtsverstärkung 1 (Delta-orthogonal wie V2a2), Bias 0. Keine Skalierung auf E f² = 1, das ist bei tanh nicht möglich.

Alles andere ist der unveränderte, eingefrorene Code `../v2a2/v2a2_run.py`. `v2c_run.py` importiert ihn und ergänzt nur `make_act` um 'tanh'. Gleich sind also: Netz, Init, Daten (inkl. Val-Satz), AdamW mit Kosinus auf 0, 3440 Schritte, Tuning-Regel, frühes Stoppen (8 Punkte), PTQ (w8 je Kanal, a4 asymmetrisch je Tensor, max/p999/p9999/p99999/mse4, Wahl w8a4a_best auf den Kalibrierbildern), Seeds.

## Historische Arme

BPreCap, SELU und TAT werden nicht neu gerechnet. Sie kommen aus `../v2a2/workers/w0.csv` (gleicher Code, gleiche Seeds, gleiches Gerät, eingefroren 16.11. 08:04:21).

**Bedingung vor dem Start (volle Kontrolle):** `v2c_run.py --control` rechnet die V2a2-Zeile BPreCap, Seed 42350, LR 3e-5, WD 0 vollständig neu. Alle Messspalten (außer Laufzeit und Zeitstempel) müssen als Text gleich sein. Ohne „KONTROLLE OK“ kein Start.

## Tuning

- 3 LR × 2 WD (0 und 3,0, wie V2a2), Tuning-Seeds 42000 und 42001, also 12 Läufe. Auswahl wie V2a2: höchste mediane Val-Genauigkeit am Ende, bei Gleichstand kleinere LR, dann kleinerer WD.
- **LR-Raster aus dem Pilot** (`../v2c_entwurf/v2c_pilot.py`, Seed 42000, 1720 Schritte, LR 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, nur Val): Regel vor dem Pilotergebnis festgelegt: das Raster ist die beste Pilot-LR (kleinster Val-Fehler am Ende) und ihre beiden Nachbarn im Pilotgitter. Liegt die beste am Rand (1e-5 oder 1e-3), wird das Raster um eine Stufe nach außen verschoben (z. B. 3e-4, 1e-3, 3e-3). Das Raster steht in `v2c_run.py` und wird mit eingefroren.

## Bestätigung

12 Seeds 42350–42361 (wie V2a2, gepaart). **Soll:** 12 Tuning-Zeilen, 12 Bestätigungszeilen. Ein Worker. Dauer etwa 9 h (Tuning etwa 20 min, Bestätigung etwa 25 min je Lauf).

## Pilot und Raster

Pilot (Seed 42000, 1720 Schritte, Kosinus, WD 0, Val-Fehler fp32 am Ende): 1e-5 73,7 %, **3e-5 69,6 %**, 1e-4 72,2 %, 3e-4 73,6 %, 1e-3 74,8 %. Raster nach der Regel: **1e-5, 3e-5, 1e-4**. Zum Vergleich BPreCap in Pilot 4 (1720 Schritte, konstante LR 3e-5): 44,6 %. tanh mit Verstärkung 1 trainiert in diesem Netz also deutlich schlechter.

## Änderung nach dem Pilot (vor dem Freeze, 17:55)

Weil tanh in fp32 etwa 25 pp hinter BPreCap liegt, wäre ein Vergleich nur der absoluten 4-Bit-Fehler fast sicher „BPreCap besser“ und würde die eigentliche Frage nicht beantworten (Begrenzung gegen Kalibrierung beim **Quantisierungspreis**). Deshalb gibt es zwei Primärtests mit Holm über 2. P1 ist unverändert, P2 ist neu. Die Vorab-Wahrscheinlichkeiten von 17:05 bleiben stehen, die für P2 sind nach dem Pilot ergänzt und als solche markiert. Außerdem korrigiert: Tuning hat 12, nicht 6 Läufe (2 Seeds).

## Primärtests (Holm über 2)

- **P1:** BPreCap − tanh, Test-Fehler w8a4a_best am frühen Stopp (pp).
- **P2:** BPreCap − tanh, 4-Bit-Preis = w8a4a_best − fp32 am frühen Stopp (pp), je Seed.

Wilcoxon-Vorzeichen-Rang-Test gepaart über 12 Seeds, zweiseitig. Ausfall oder fehlende Zeile zählt 90 % (beim Preis: Preis 90 pp).

- Holm-p < 0,05 und Median < 0: **„BPreCap besser“**.
- Holm-p < 0,05 und Median > 0: **„tanh besser“**.
- Sonst: „unklar“.

## Sekundärtests (Holm über 3)

| Nr. | Vergleich | Größe |
|---|---|---|
| S1 | BPreCap − tanh | fp32 |
| S2 | tanh − SELU | w8a4a_best |
| S3 | tanh − TAT | w8a4a_best |

Urteil aus Sicht des erstgenannten Arms: Holm-p < 0,05 und Median < 0 „besser“, > 0 „schlechter“, sonst „unklar“.

**Beschreibend:** w8a4a_max und Endpunkt-Werte BPreCap − tanh, Mediane je Form, 4-Bit-Preis, Kalibrierwahl, Aktivierungsbereich, Tuning-Wahl.

## Deutung (vorab)

| Ausgang | Bedeutung für v0.4 |
|---|---|
| P1 „BPreCap besser“, P2 „unklar“ oder „tanh besser“ | Ein kleiner 4-Bit-Preis kommt aus der Begrenzung allein; BPreCap gewinnt über die Trainierbarkeit (Kalibrierung). v0.4: „Begrenzung macht den Preis klein, die Tiefenkalibrierung macht das Netz gut.“ |
| P1 und P2 „BPreCap besser“ | Die Kalibrierung senkt auch den Preis; Begrenzung allein reicht weder für Güte noch für Robustheit. Stärkste Aussage. |
| P1 „unklar“ | tanh ist bei 4 Bit nicht klar schlechter, trotz schlechterem fp32: Hinweis, dass die Begrenzung den Hauptteil trägt. |
| P1 „tanh besser“ | BPreCap ist bei 4 Bit nicht die beste begrenzte Wahl. v0.4 berichtet das offen und schränkt die Kernaussage ein. |

S1 sagt zusätzlich, ob BPreCap in fp32 gegen tanh etwas gewinnt (Trainierbarkeit).

## Vorab-Wahrscheinlichkeiten (vor dem Pilotergebnis)

| Aussage | p |
|---|---|
| alle 12 tanh-Läufe ok | 0,9 |
| P1 „BPreCap besser“ | 0,4 |
| P1 „tanh besser“ | 0,2 |
| S1 „besser“ (BPreCap in fp32 vor tanh) | 0,45 |
| S2 „besser“ (tanh vor SELU bei 4 Bit) | 0,55 |
| S3 „besser“ (tanh vor TAT bei 4 Bit) | 0,85 |
| Raster am Rand (gewählte LR am Rand) | 0,35 |
| *nach dem Pilot ergänzt:* P1 „BPreCap besser“ | 0,85 |
| *nach dem Pilot ergänzt:* P2 „BPreCap besser“ | 0,35 |
| *nach dem Pilot ergänzt:* P2 „tanh besser“ | 0,25 |

## Grenzen

- Historische Arme, nicht zeitgleich gerechnet; abgesichert durch die volle Kontrolle einer Zeile.
- Eine tanh-Variante (Verstärkung 1, Bias 0). Eine tanh an der geordnet-chaotischen Grenze mit q* > 0 (σ_w² > 1, σ_b > 0) ist nicht geprüft.
- tanh bekommt sein Raster aus einem eigenen Pilot, die anderen Formen aus Pilot 4. Beides folgt derselben Logik (Raster um die beste Pilot-LR).
- Alle Grenzen von V2a2 (ein Datensatz, ein Netz, nur PTQ je Tensor, M1 Pro).

## Ablauf

1. Pilot fertig, Raster nach der Regel in `v2c_run.py` eintragen.
2. `launch_v2c.sh`: Netzteil und andere Worker prüfen, Freeze (auch `../v2a2/v2a2_run.py`, Konstanten, V2a2-CSV, Pilot-JSON, Daten), Selbsttest, volle Kontrolle, Auswertungstest, Rauchtest, Start in screen `v2cw0` mit caffeinate, Wächter `v2cwache`.
3. Auswertung `v2c_analyze.py <V2c-Ordner absolut>` (scipy-venv), Hash-Prüfung vorher und nachher, Bericht `V2c_Auswertung.md`.
