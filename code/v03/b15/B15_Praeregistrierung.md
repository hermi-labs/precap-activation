# B15 – Präregistrierung: BPreCap gegen SELU im TAT-Rezept bei d200 (CIFAR-10, privat)

Stand: 01.11.2026, 06:30. Vor dem Start, **noch nicht eingefroren und nicht gestartet.** PreCap, die Tiefenfamilie und Lagma sind Aarons Ideen. Das Design habe ich mit Claude entworfen.

**Start nur nach Aarons ausdrücklicher Freigabe:** etwa 4 h GPU, über der 1-h-Grenze. Nichts veröffentlichen.

## Ausgangslage

- **B12:** Bei d100 im TAT-Rezept und B10-Regime ist SELU nur 0,95 pp hinter BPreCap in fp32 (12/0). Bei 4 Bit gibt es keinen nachweisbaren Unterschied (−0,47 pp, 9/3, p = 0,077). SELUs 4-Bit-Preis ist ebenso klein (+2,5 gegen +3,0 pp).
- **D1:** SELU hat leichte Ausläufer (Kurtosis 2,5) und dämpft Rundungsfehler über die Tiefe am stärksten.
- **Tiefenkalibrierung:** Sie ist der Teil von BPreCap, der bei größerer Tiefe zählen sollte. SELU hat festes χ1 = 1,0716, bei d200 wächst der Gradient bei der Init damit etwa 970-fach (wie in v0.2). BPreCap hat χ1 = 1 + 2/L = 1,01.
- **v0.2:** SELU ist bei d200 mit Gauß-Init und kurzem Budget gescheitert. Mit dem Rezept und dem B10-Regime ist das offen.

## Frage

Ist BPreCap bei d200 im TAT-Rezept und B10-Regime auf CIFAR-10 bei fairen 4 Bit besser als SELU, wenn beide Formen eigens getunt sind?

## Eine Änderung gegenüber B12: Tiefe 200, dafür nur BPreCap und SELU

- **Formen:**
  - kL2B_tr: Konstanten für d200 (λ = 4,9097, α = 1,2489, b = 0,1846, M = 3, χ1 = 1,01).
  - SELU_tr: Gain 1,0.
  - Beide im TAT-Rezept: orthogonale Init × Gain, Normierung je Beispiel.
- **Regime wie B10:** AdamW, Kosinus auf 0 über 6880 Schritte, ohne Anlauf, frühes Stoppen nach fp32-Val, Messpunkte alle 860 Schritte.
- **Tuning für beide Formen neu:** LR-Spitze {2e-5, 5e-5, 1e-4, 2e-4} × WD {0; 0,3; 3,0}, 2 Tuning-Seeds (42000, 42001), Val-Genauigkeit am Ende. Wahlregel wie B10 (Gleichstand: kleinere LR, dann kleinerer WD).
  - Das Raster liegt eine Stufe tiefer als in B12, weil bei d200 kleinere LR gewählt wurden (B5, B8).
  - Liegt die Wahl am Rand, wird das berichtet.
- **Bestätigung:** 12 neue Seeds 42192–42203, gepaart innerhalb der Studie.
- **Kontrolle:** kL2B_tr bei d100, Seed 42168, mit der B10-Wahl (1e-4, WD 0,3). Alle Kennzahlen an allen Messpunkten müssen bitgleich zu B10 sein.
- **Konstanten wie B10/B12:** Netz (Breite 256), Batch 128, Daten, Kalibrierung (1000 Trainingsbilder), Quantisierung, Cache-Grenze.
- **Code:** `b15_run.py` ist `b12_run.py` mit genau vier geänderten Zeilen (Tiefe, Formen, LR-Raster, Seeds), per `diff` geprüft.

## Läufe

- Tuning: 2 × 12 × 2 = 48 (bei d200).
- Bestätigung: 24 + 1 Kontrolle = 25.
- Etwa 4 h mit 2 Workern. Ein d200-Lauf mit 6880 Schritten dauert etwa 7 min.

## Primärtest

**kL2B_tr − SELU_tr bei w8a4a_best am Punkt des frühen Stoppens** (d200).
- Zweiseitiger exakter Wilcoxon-Vorzeichen-Rang-Test, gepaart, α = 0,05.
- **Neu:** eigener exakter Test über alle 2^n Vorzeichenmuster, Mittelränge bei gleichen Beträgen, Nullen verworfen. Grund: scipy wechselt bei Nullen still zur Normal-Näherung und ignoriert gleiche Beträge (siehe `analysis_check/`).
- **Erfolg:** p < 0,05 und Median < 0.
- Ein Ausfall zählt als 90 %.

## Sekundär (Holm über 3)

| Nr. | Vergleich (kL2B_tr − SELU_tr, Stopp-Punkt) |
|---|---|
| S1 | fp32 |
| S2 | w8a4_best (symmetrisch) |
| S3 | w8a3_best |

**Beschreibend:**
- gewählte LR und WD, Lage am Rand;
- divergierte Tuning-Läufe;
- Stopp-Punkte, Verlauf, Bereich;
- 4-Bit-Preis;
- Trainingsverlust.

## Deutungsregeln (vorab)

| Ergebnis | Deutung für v0.3 |
|---|---|
| Primär „besser“ und S1 „besser“ | Bei größerer Tiefe trägt die Tiefenkalibrierung. BPreCap ist SELU in fp32 und bei 4 Bit überlegen |
| S1 „besser“, Primär „unklar“ | BPreCap trainiert bei d200 besser, der 4-Bit-Preis ist gleich |
| Primär „besser“, S1 „unklar“ | Bei d200 trennt nur die 4-Bit-Robustheit |
| Primär und S1 „unklar“ | Im Rezept hält SELU bis d200 mit. v0.3 sagt das so, BPreCaps Vorteil gegenüber SELU ist dann klein |
| Primär oder S1 „schlechter“ | SELU ist besser. Die Kernaussage von v0.3 muss eingeschränkt werden |
| Kontrolle nicht bitgleich | Abweichung wird gesucht und berichtet |

## Vorab-Erwartung

| Größe | Erwartung |
|---|---|
| S1 „besser“ | 0,6 |
| Primär „besser“ | 0,5 |
| SELU divergiert in mindestens einer Tuning-Kombination | 0,8 |
| Wahl mindestens einer Form am LR-Rand | 0,4 |
| Kontrolle bitgleich | 0,95 |

## Grenzen

- Nur 2 Tuning-Seeds, nur CIFAR-10, nur Quantisierung nach dem Training.
- Nur eine Tiefe über 100. Die Aussage gilt für d200, nicht allgemein für „tiefer“.
- Ein Rauchtest mit 172 Schritten (Mechanik-Test) zeigte SELU bei d200 langsamer (85,7 % gegen 59,2 % Fehler). Das ist kein Ergebnis, aber ein Vorwissen beim Entwurf und wird hier genannt.

## Freeze

Eingefroren werden beim Start:
- `b15_run.py`, `b15_analyze.py`, `test_b15_mechanik.py`, `test_b15_analyse.py`,
- diese Datei,
- `hp_from_b10.csv`, `src/precap/*.py`,
- die CIFAR-10-Dateien.

Vorab gemachte Tests (01.11., 06:20–06:30):

- **Selbsttest:** n_confirm 25, n_tune 48, LR-Raster wie oben.
- **Mechanik** (`test_b15_mechanik.py`): **MECHANIK OK**, 27 Prüfungen.
  - Formen und Init bei d100 und d200 bitgleich zu B10.
  - χ1 von kL2B bei d200 = 1,00999.
  - SELU-Gain ergibt E[f²] = 1.
  - Kontrolle bitgleich zu B10.
  - Beide Formen laufen bei d200, die Zwischenmessung ändert das Training nicht.
- **Auswertung an künstlichen Daten** (`test_b15_analyse.py`): **AUSWERTUNG OK**.
  - Erfolg und „schlechter“ werden erkannt.
  - Null mit 20 Wiederholungen: 1 Fehlalarm primär, 1 sekundär.
  - Eine falsche Kontrolle wird erkannt, am Ende wie an einem Messpunkt.
  - Ein Ausfall zählt als 90 %.
  - Nullen in den Differenzen sind exakt behandelt.
  - Der relative Pfad findet B10.
- **Rauchtest** (Probe, 40 Schritte): Tuning 8, Bestätigung 3, Zwischenspalten 3. Das Startskript erwartet genau diese Zahlen.
- **Start:** `bash studien/b15/launch_b15.sh`. Es bricht ab, wenn Ergebnisordner existieren, andere Worker laufen oder der Mac nicht am Netzteil ist.

## Nachtrag vor dem Freeze (01.11., 10:05): Freigabe und macOS-Update

- **Freigabe:** Aaron hat B15 am 01.11. morgens freigegeben („dann kannst du b15 machen“). Der Start erfolgt nach einem Neustart für ein macOS-Update.
- **Softwarestand vor dem Update:** `meta_vor_update.txt` (macOS 27.0.1, Build 26A434, MLX 0.32.4, Python 3.15.0, Metal 4).
- **Risiko, vorab benannt:** Die Kontrolle (kL2B_tr d100, Seed 42168) wird mit der B10-Zeile verglichen, die vor dem Update entstand. Ändert das Update die Numerik von Metal oder MLX, kann die Kontrolle abweichen, obwohl der Code gleich ist.
- **Regel:**
  - Weicht die Kontrolle nach dem Update ab, ist das **kein Fehler von B15**, sondern ein Plattformwechsel. Er wird so berichtet.
  - Die Tests innerhalb von B15 bleiben gültig, weil alle B15-Läufe auf derselben Plattform laufen.
  - Zusätzlich wird der C4-Neulauf (`analysis_check/c4_rerun.py`, B10 Seed 42173) nach dem Update wiederholt, um zu dokumentieren, ob die Plattform noch bitgleich rechnet.
  - Der Mechanik-Test vergleicht B15- und B10-Code auf derselben Plattform und ist davon nicht betroffen.
- **Nach dem Update (01.11., 10:20):** macOS 27.2 (Build 26B123), MLX 0.32.4 unverändert. Der C4-Neulauf (B10, Seed 42173) ist **bitgleich** (93/93 Spalten). Die Plattform rechnet also wie vorher, und die Kontrolle gegen B10 bleibt aussagekräftig. Siehe `meta_nach_update.txt`.
