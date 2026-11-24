# V2d – Präregistrierung: BPreCap gegen eine gleich kalibrierte tanh

Stand: 19.11.2026, 02:55, vor dem ersten Pilotergebnis (privat, M1 Pro). PreCap, BPreCap, das Kalibrierungsrezept (E f = 0, E f² = 1, χ1 = 1 + 2/L) und die Hülle sind Aarons Ideen. Das Design habe ich mit Claude entworfen. Anlass ist V2c (`../v2c/V2c_Auswertung.md`). Eingefroren wird mit dem Start (`meta/frozen_sha256.txt`). **Start nur mit Aarons Freigabe** (Lauf etwa 9 h, länger als die Freigabe bis 19.11. 6:30).

## Frage

Ist BPreCap im V2a2-Netz (CNN, L = 50, ohne Normierung, ohne Residuen, Delta-orthogonal, CIFAR-10) bei 4-Bit-Aktivierungen (w8a4a_best) besser als eine **tanh, die nach demselben Rezept auf die Tiefe kalibriert ist**?

## Hintergrund

- V2a2: BPreCap ist bei w8a4a_best besser als SELU (−2,55 pp) und TAT (−13,73 pp), je 12/0.
- V2c: Eine unkalibrierte tanh (Verstärkung 1, Bias 0) hat den kleinsten 4-Bit-Preis (1,26 pp gegen 4,64 pp), trainiert aber schlecht (fp32 66,5 % gegen 42,9 %). BPreCap liegt bei 4 Bit 20,06 pp vorn. Deutung: Begrenzung macht den Preis klein, Kalibrierung macht das Netz gut.
- Offen: Kommt BPreCaps Vorsprung von der Kalibrierung allein (dann sollte eine kalibrierte tanh gleichziehen) oder auch von der Form (Hülle, Asymmetrie, SELU-artiger Fixpunkt)?
- Literatur: Xiao et al. 2018 (https://arxiv.org/abs/1806.05393) trainieren tanh-CNNs an der kritischen Linie (σ_b > 0). Wir kalibrieren tanh hier mit BPreCaps Rezept, damit nur die Form verschieden ist.

## Die eine Änderung

Aktivierung **ctanh(x) = λ · tanh(b · x)** mit λ = 2,2948782077912586, b = 0,5328431680484216. Die Werte lösen bei z ~ N(0, 1):
- E f = 0 (tanh ist ungerade),
- E f² = 1,
- χ1 = E[f'(z)²] = 1 + 2/L = 1,04, wie BPreCap bei L = 50.

Gerechnet mit scipy brentq auf Gauß-Hermite mit 301 Punkten (19.11., 02:50). Der Bereich ist ±2,29, symmetrisch (BPreCap in V2a2: −2,24 bis 3,00).

Alles andere ist der unveränderte, eingefrorene Code `../v2a2/v2a2_run.py`. `v2d_run.py` importiert ihn und ergänzt nur `make_act` um 'ctanh'. Gleich sind also: Netz, Init, Daten (inkl. Val-Satz), AdamW mit Kosinus auf 0, 3440 Schritte, Tuning-Regel, frühes Stoppen (8 Punkte), PTQ (w8 je Kanal, a4 asymmetrisch je Tensor, max/p999/p9999/p99999/mse4, Wahl w8a4a_best auf den Kalibrierbildern), Seeds.

## Historische Arme

BPreCap, SELU und TAT kommen aus `../v2a2/workers/w0.csv` (eingefroren 16.11. 08:04:21), tanh aus `../v2c/workers/w0.csv` (eingefroren 18.11. 17:47:09). Gleicher Code, gleiche Seeds, gleiches Gerät.

**Bedingung vor dem Start (volle Kontrolle):** `v2d_run.py --control` rechnet die V2a2-Zeile BPreCap, Seed 42350, LR 3e-5, WD 0 vollständig neu. Alle Messspalten (außer Laufzeit und Zeitstempel) müssen als Text gleich sein. Ohne „KONTROLLE OK“ kein Start.

## Tuning

- 3 LR × 2 WD (0 und 3,0), Tuning-Seeds 42000 und 42001, also 12 Läufe. Auswahl wie V2a2: höchste mediane Val-Genauigkeit am Ende, bei Gleichstand kleinere LR, dann kleinerer WD.
- **LR-Raster aus dem Pilot** (`../v2d_entwurf/v2d_pilot.py`, Seed 42000, 1720 Schritte, LR 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, nur Val). Die Regel steht vor dem Pilotergebnis fest und ist dieselbe wie in V2c: Das Raster ist die beste Pilot-LR (kleinster Val-Fehler am Ende) mit ihren beiden Nachbarn im Pilotgitter. Liegt die beste am Rand (1e-5 oder 1e-3), wird das Raster um eine Stufe nach außen verschoben. Das Raster wird in `v2d_run.py` eingetragen und mit eingefroren.

## Pilot und Raster (ergänzt 19.11., 03:43, nach dem Pilot, vor dem Freeze)

Pilot (Seed 42000, 1720 Schritte, Kosinus, WD 0, Val-Fehler fp32 am Ende, 5/5 ok, 03:02 bis 03:42): 1e-5 61,4 %, **3e-5 51,9 %**, 1e-4 68,3 %, 3e-4 72,1 %, 1e-3 75,4 %. Raster nach der Regel: **1e-5, 3e-5, 1e-4**, eingetragen in `v2d_run.py`. Selbsttest danach: E f = 0,0, E f² = 1,0, χ1 = 1,04, Bereich 2,2949, 12 Tuning- und 12 Bestätigungsläufe.

Zum Vergleich bei gleicher Schrittzahl: tanh (V2c-Pilot) 69,6 %, BPreCap (Pilot 4, konstante LR) 44,6 %. Die Kalibrierung hilft der tanh also stark, BPreCap bleibt im Pilot aber etwa 7 pp vorn (ein Seed, verschiedene LR-Pläne, nur ein Hinweis).

**Nach dem Pilot ergänzte Vorab-Wahrscheinlichkeiten** (die Werte von 02:55 bleiben stehen): P1 „BPreCap besser“ 0,6, P1 „ctanh besser“ 0,1, P2 „ctanh besser“ 0,35, S1 „besser“ 0,7, S3 „besser“ 0,15.

## Bestätigung

12 Seeds 42350–42361 (wie V2a2 und V2c, gepaart). **Soll:** 12 Tuning-Zeilen, 12 Bestätigungszeilen. Ein Worker. Dauer etwa 9 h.

## Primärtests (Holm über 2)

- **P1:** BPreCap − ctanh, Test-Fehler w8a4a_best am frühen Stopp (pp).
- **P2:** BPreCap − ctanh, 4-Bit-Preis = w8a4a_best − fp32 am frühen Stopp (pp), je Seed.

Wilcoxon-Vorzeichen-Rang-Test gepaart über 12 Seeds, zweiseitig. Ausfall oder fehlende Zeile zählt 90 % (beim Preis 90 pp).

- Holm-p < 0,05 und Median < 0: **„BPreCap besser“**.
- Holm-p < 0,05 und Median > 0: **„ctanh besser“**.
- Sonst: „unklar“.

## Sekundärtests (Holm über 4)

| Nr. | Vergleich | Größe | Frage |
|---|---|---|---|
| S1 | BPreCap − ctanh | fp32 | Trainierbarkeit bei gleicher Kalibrierung |
| S2 | ctanh − tanh | w8a4a_best | Hilft die Kalibrierung der tanh? |
| S3 | ctanh − SELU | w8a4a_best | Reicht eine kalibrierte begrenzte Form, um SELU zu schlagen? |
| S4 | ctanh − TAT | w8a4a_best | ebenso gegen TAT |

Urteil aus Sicht des erstgenannten Arms: Holm-p < 0,05 und Median < 0 „besser“, > 0 „schlechter“, sonst „unklar“.

**Beschreibend:** w8a4a_max und Endpunkt-Werte BPreCap − ctanh, Mediane je Form, 4-Bit-Preis, Kalibrierwahl, Aktivierungsbereich, Tuning-Wahl.

## Deutung (vorab)

| Ausgang | Bedeutung für v0.4 |
|---|---|
| P1 „BPreCap besser“ | Bei gleicher Kalibrierung trägt BPreCaps Form (Hülle, Asymmetrie) zum 4-Bit-Vorteil bei. Stärkste Aussage: nicht nur das Rezept, auch die Form zählt. |
| P1 „unklar“ | Der Vorteil kommt im Wesentlichen aus Rezept und Begrenzung. v0.4 formuliert den Beitrag als Rezept („kalibrierte begrenzte Aktivierung“), BPreCap als eine Umsetzung. |
| P1 „ctanh besser“ | Eine kalibrierte tanh ist bei 4 Bit die bessere Wahl. v0.4 berichtet das offen. Der Beitrag ist dann das Rezept, nicht die BPreCap-Form. |
| P2 | sagt, ob die Form den Preis beeinflusst, bei ähnlichem Bereich (±2,29 gegen −2,24…3,00). |

## Vorab-Wahrscheinlichkeiten (vor dem Pilotergebnis, 02:55)

| Aussage | p |
|---|---|
| alle 12 ctanh-Läufe ok | 0,9 |
| P1 „BPreCap besser“ | 0,35 |
| P1 „unklar“ | 0,4 |
| P1 „ctanh besser“ | 0,25 |
| P2 „BPreCap besser“ | 0,3 |
| P2 „ctanh besser“ | 0,3 |
| S1 „besser“ (BPreCap in fp32 vor ctanh) | 0,35 |
| S2 „besser“ (ctanh vor tanh) | 0,95 |
| S3 „besser“ (ctanh vor SELU bei 4 Bit) | 0,45 |
| S4 „besser“ (ctanh vor TAT bei 4 Bit) | 0,9 |
| Raster am Rand | 0,2 |

Begründung kurz: Der Bereich ist ähnlich, die Kalibrierung gleich. Ich erwarte eher einen kleinen Unterschied in die eine oder andere Richtung. Symmetrisch begrenzte Aktivierungen passen besser zu einer symmetrischen Verteilung, die asymmetrische a4-Quantisierung gleicht das aber aus.

## Grenzen

- Historische Arme, nicht zeitgleich gerechnet; abgesichert durch die volle Kontrolle einer Zeile.
- Eine Kalibrierung (χ1 = 1,04 bei q = 1). Die kritische tanh nach Xiao et al. mit Zufallsbias ist nicht geprüft.
- ctanh bekommt sein Raster aus einem eigenen Pilot, die anderen Formen aus früheren Piloten. Beides folgt derselben Logik.
- Seeds sind dieselben wie in V2a2 und V2c, weil gepaart verglichen wird (keine neuen, unbenutzten Seeds).
- Alle Grenzen von V2a2 (ein Datensatz, ein Netz, nur PTQ je Tensor, M1 Pro).

## Ablauf

1. Pilot fertig, Raster nach der Regel in `v2d_run.py` eintragen, Abschnitt „Pilot und Raster“ ergänzen.
2. Aaron gibt den Start frei.
3. `launch_v2d.sh`: Netzteil und andere Worker prüfen, Freeze (auch `../v2a2/v2a2_run.py`, Konstanten, V2a2-CSV, V2c-CSV, Pilot-JSON, Daten), Selbsttest, volle Kontrolle, Auswertungstest, Rauchtest, Start in screen `v2dw0` mit caffeinate, Wächter `v2dwache`.
4. Auswertung `v2d_analyze.py <V2d-Ordner absolut>` (scipy-venv), Hash-Prüfung vorher und nachher, Bericht `V2d_Auswertung.md`.
