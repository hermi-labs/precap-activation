# V2gb – Präregistrierung: BPreCap gegen eine kritische tanh nach Xiao et al. (Fassung b von V2g)

Stand: 20.11.2026, 07:00. Die Präregistrierung entstand vor jedem Ergebnis mit dieser Form. PreCap, BPreCap, die Hülle und das Kalibrierungsrezept sind Aarons Ideen. Das Design habe ich mit Claude entworfen. Anlass ist die offene Grenze im v0.4-Entwurf (Abschnitt 7) und in V2c/V2d: Eine tanh an der kritischen Linie mit Zufallsbias nach Xiao et al. 2018 (https://arxiv.org/abs/1806.05393) wurde bisher nicht verglichen. Gestartet wird nach V2f, mit Aarons ausdrücklicher Freigabe vom 20.11. abends („Du kannst es gerne starten auf Grundlage des vorliegenden Ergebnis“) und der Freigabe vom 21.11. morgens, den Mac bis zum Abend zu nutzen. Eingefroren wird mit dem Start (`meta/frozen_sha256.txt`).

**Versionshinweis (21.11.2026, 10:00):** V2g (`../v2g/`) wurde am 21.11. um 09:28 eingefroren. Der Selbsttest brach um 09:32 ab, weil `bias_ok` ein numpy-bool war, den `json.dumps` nicht schreiben kann. Es wurde nichts trainiert, und es gibt keine Ergebnisse. V2gb ist dieselbe Studie mit drei Änderungen:
1. `bool()` um `bias_ok` in `v2gb_run.py`.
2. Neue Datei- und screen-Namen (v2gb, `v2gbw0`, `v2gbwache`). Der Selbsttest schreibt Fehler jetzt nach `logs/selfcheck_err.txt`.
3. Die Quellenangabe zu σ_b² ist präzisiert (siehe unten).

Design, Konstanten, Seeds, Tests und Entscheidungsregeln sind unverändert. Alle Vorab-Wahrscheinlichkeiten stammen weiter vom 20.11.

## Frage

Ist BPreCap im V2a2-Netz bei 4-Bit-Aktivierungen (w8a4a_best) besser als eine tanh, die nach Xiao et al. an der **kritischen Linie** initialisiert ist? Das V2a2-Netz ist ein CNN mit L = 50, ohne Normierung, ohne Residuen, Delta-orthogonal, auf CIFAR-10.

## Die eine Änderung

Die Form heißt **xtanh**: Aktivierung tanh mit Verstärkung 1. Geändert wird die Initialisierung der Faltungen:
- **Gewichte:** die Delta-orthogonalen Kerne aus V2a2, multipliziert mit σ_w = √σ_w².
- **Bias:** je Kanal unabhängig aus N(0, σ_b²) gezogen, statt 0. Gezogen wird mit einem eigenen Generator `np.random.default_rng([seed, 7])`, damit die Gewichts-Ziehung bitgleich zu V2a2 bleibt.
- **σ_b² = 2·10⁻⁵:** Xiao et al. nennen diesen Wert für die tanh-Heatmap auf MNIST (Abb. 2). Für ihre CIFAR-10-Läufe geben sie weder σ_w², σ_b² noch den Optimierer an (Volltext geprüft am 21.11.). Derselbe Punkt ist Fall (1) bei Pennington, Schoenholz und Ganguli 2017 (https://arxiv.org/abs/1711.04735): σ_b² = 2,01·10⁻⁵, q* = 0,025.
- **σ_w² = 1,04991163197033:** Er ist so gewählt, dass χ1 = σ_w² · E[tanh'(√q* z)²] = 1 im Fixpunkt q* = σ_w² · E[tanh(√q* z)²] + σ_b² gilt. Daraus folgt q* = 0,025874. Gelöst habe ich das mit scipy brentq auf Gauß-Hermite mit 301 Punkten (20.11., 07:00).
- Die Ausgabeschicht bleibt unverändert.

Nach der Mittelfeld-Rechnung fällt die Varianz vom Eingang (q = 1,05) auf etwa 0,029 nach 50 Schichten. Bei der unkalibrierten tanh aus V2c sind es 0,010. Ich erwarte deshalb ein Verhalten ähnlich wie bei V2c (schlecht trainierbar, kleiner 4-Bit-Preis).

Alles andere ist der unveränderte, eingefrorene Code `../v2a2/v2a2_run.py`: Netz, Daten (inkl. Val-Satz), AdamW mit Kosinus auf 0, 3440 Schritte, Tuning-Regel, frühes Stoppen (8 Punkte) und PTQ. Zur PTQ gehören w8 je Kanal, a4 asymmetrisch je Tensor, die Kalibrierungen max/p999/p9999/p99999/mse4 und die Wahl von w8a4a_best auf den Kalibrierbildern. `v2gb_run.py` importiert den Code, ergänzt `make_act` um 'xtanh' und ersetzt `R.CNN` durch eine Unterklasse, die nur bei xtanh anders initialisiert.

## Historische Arme

Alle Arme wurden mit gleichem Code, gleichen Seeds und auf demselben Gerät gerechnet:
- BPreCap, SELU und TAT aus `../v2a2/workers/w0.csv`,
- tanh aus `../v2c/workers/w0.csv`,
- ctanh aus `../v2d/workers/w0.csv`.

**Kontrolle:** Die volle Kontrolle wie in V2d (BPreCap 42350 neu rechnen, 46 min) entfällt. V2f rechnet gerade 18 Bestätigungsläufe aus V2a2 und V2d mit demselben eingefrorenen Code bitgleich nach. Stattdessen prüft der Selbsttest:
1. Die Unterklasse liefert für BPreCap bitgleiche Parameter wie die Original-Klasse.
2. Die xtanh-Gewichte sind genau σ_w mal die Originalgewichte.
3. 20 Trainingsschritte BPreCap mit der gepatchten und der Original-Klasse ergeben bitgleiche Parameter.

## Tuning

- 3 LR × 2 WD (0 und 3,0), Tuning-Seeds 42000 und 42001, also 12 Läufe. Auswahl wie in V2a2.
- **LR-Raster ohne eigenen Pilot:** 1e-5, 3e-5, 1e-4, also das Raster von V2c (tanh) und V2d (ctanh). Grund: xtanh unterscheidet sich von der V2c-tanh nur durch σ_w² = 1,05 und einen winzigen Bias, und beide Piloten fanden 3e-5 am besten. Liegt die Wahl am Rand, wird das berichtet (Grenze).

## Bestätigung

12 Seeds 42350–42361, gepaart mit allen historischen Armen. **Soll:** 12 Tuning-Zeilen und 12 Bestätigungszeilen. Es läuft ein Worker. Die Dauer schätze ich auf etwa 9 h (wie V2c und V2d).

## Primärtests (Holm über 2)

- **P1:** BPreCap − xtanh, Test-Fehler w8a4a_best am frühen Stopp (pp).
- **P2:** BPreCap − xtanh, 4-Bit-Preis = w8a4a_best − fp32 am frühen Stopp (pp), je Seed.

Getestet wird mit dem Wilcoxon-Vorzeichen-Rang-Test, gepaart über 12 Seeds und zweiseitig. Ein Ausfall oder eine fehlende Zeile zählt mit 90 % (beim Preis mit 90 pp). Die Urteile:
- Holm-p < 0,05 und Median < 0: **„BPreCap besser“**.
- Holm-p < 0,05 und Median > 0: **„xtanh besser“**.
- Sonst: „unklar“.

## Sekundärtests (Holm über 4)

| Nr. | Vergleich | Größe | Frage |
|---|---|---|---|
| S1 | BPreCap − xtanh | fp32 | Trainierbarkeit |
| S2 | xtanh − tanh (V2c) | w8a4a_best | Hilft die kritische Initialisierung der tanh? |
| S3 | xtanh − ctanh (V2d) | w8a4a_best | kritisch mit kleinem q* gegen BPreCaps Rezept bei q = 1 |
| S4 | xtanh − SELU | w8a4a_best | Reicht die kritische tanh, um SELU zu schlagen? |

Das Urteil steht aus Sicht des erstgenannten Arms: Holm-p < 0,05 und Median < 0 heißt „besser“, Median > 0 heißt „schlechter“, sonst „unklar“.

**Beschreibend:** w8a4a_max, die Endpunkt-Werte, die Mediane je Form, der 4-Bit-Preis, die Kalibrierwahl, der Aktivierungsbereich und die Tuning-Wahl.

## Deutung (vorab)

| Ausgang | Bedeutung für v0.4 |
|---|---|
| P1 „BPreCap besser“ | Auch gegen die kritische tanh nach Xiao liegt BPreCap bei 4 Bit vorn. Die Grenze in Abschnitt 7 wird geschlossen. |
| P1 „unklar“ oder „xtanh besser“ | Eine richtig initialisierte tanh ist ein ernsthafter Gegner. v0.4 berichtet das offen und schwächt die Aussage ab. |
| S2 | zeigt, ob die kritische Initialisierung gegenüber der Init aus V2c etwas ändert. Erwartet wird wenig, wegen des ähnlichen Varianzverlaufs. |

## Vorab-Wahrscheinlichkeiten

| Aussage | p |
|---|---|
| alle 12 xtanh-Läufe ok | 0,9 |
| P1 „BPreCap besser“ | 0,85 |
| P1 „xtanh besser“ | 0,03 |
| P2 „xtanh besser“ (kleinerer Preis) | 0,6 |
| S1 „besser“ (BPreCap in fp32 vor xtanh) | 0,85 |
| S2 „besser“ (xtanh vor tanh) | 0,25 |
| S3 „schlechter“ (xtanh hinter ctanh) | 0,75 |
| Wahl am Rand des Rasters | 0,25 |

Begründung: Bei q* ≈ 0,026 arbeitet die tanh fast linear. Das Netz verhält sich dann ähnlich wie die V2c-tanh (fp32 66,5 % Fehler, Preis 1,26 pp) und anders als die ctanh (q = 1).

## Grenzen

- Es wird nur ein Punkt der kritischen Linie geprüft (σ_b² = 2·10⁻⁵, kleines q*). Der Punkt q* = 1 (σ_w² = 2,153, σ_b² = 0,151) wird nicht geprüft. BPreCaps Rezept bei q = 1 entspricht ctanh (V2d).
- Xiao et al. trainieren mit SGD bzw. ohne LR-Abfall und mit sehr vielen Schritten. Hier gilt das V2a2-Regime (AdamW, 3440 Schritte). Die kritische Initialisierung hilft vor allem bei sehr großer Tiefe, L = 50 ist dafür klein.
- Das Raster kommt nicht aus einem eigenen Pilot.
- Die historischen Arme wurden nicht zeitgleich gerechnet. Es gibt keine volle Kontrolle einer Zeile, die Prüfungen des Selbsttests ersetzen sie.
- Alle Grenzen von V2a2 gelten auch hier: ein Datensatz, ein Netz, nur PTQ je Tensor, M1 Pro.

## Ablauf

1. `launch_v2gb.sh` wird nach V2f gestartet. Es prüft Netzteil und andere Worker. Dann folgen der Freeze, die Hash-Prüfung von V2a2/V2c/V2d, der Selbsttest (inkl. bitgleicher 20-Schritt-Probe), der Auswertungstest, der Rauchtest und der Start in screen `v2gbw0` mit caffeinate und dem Wächter `v2gbwache`.
2. Ausgewertet wird mit `v2gb_analyze.py <V2gb-Ordner absolut>` (scipy-venv), mit Hash-Prüfung vorher und nachher. Der Bericht kommt nach `V2gb_Auswertung.md`.
