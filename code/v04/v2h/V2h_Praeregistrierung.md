# V2h – Präregistrierung: BPreCap gegen eine zweite kritische tanh, Punkt nach der Regel von Hayou et al.

Stand: 21.11.2026, 18:50. Die Präregistrierung entstand vor jedem Ergebnis mit dieser Form. PreCap, BPreCap, die Hülle und das Kalibrierungsrezept sind Aarons Ideen. Das Design habe ich mit Claude entworfen. Freigabe: Aaron am 21.11. abends („mach beides hast den mac bis morgen“). Eingefroren wird mit dem Start (`meta/frozen_sha256.txt`).

## Anlass

V2gb hat BPreCap gegen eine tanh an der kritischen Linie mit Zufallsbias verglichen (xtanh). Ergebnis: P1 „BPreCap besser“, −18,51 pp, 12/0. Diese tanh lag aber am fast linearen Ende der Linie, mit σ_b² = 2·10⁻⁵ und q* = 0,026 (Pennington 2017 Fall 1). Ein Gutachter kann einwenden, dass Hayou, Doucet und Rousseau 2019 (http://proceedings.mlr.press/v97/hayou19a/hayou19a.pdf) für eine gegebene Tiefe einen größeren Bias empfehlen. V2h prüft diesen Punkt.

**Am Original geprüft** (21.11., Text aus dem PDF):
- Ihre Regel: σ_b auf der kritischen Linie so wählen, dass βq ≈ L. Dabei ist βq = 2 E[φ'(√q Z)²] / (q E[φ''(√q Z)²]) (Prop. 3 und 4, Abschnitt „Choice of σb on the Edge of Chaos“).
- Tabelle 3 (tanh, MNIST, L = 30/50/200): bester Wert aus dem Raster bei L = 50 ist σ_b = 0,040; „with rule βq ≈ L“ ist 0,030 angegeben.
- Im Text zu Abb. 6 steht, σ_b = 0,05 ergebe βq ∼ 50.
- Meine Nachrechnung für tanh ergibt βq = 50 bei σ_b = 0,0412 und βq = 39 bei σ_b = 0,05. Die Angaben im Paper sind also untereinander nicht ganz stimmig.
- Ich nehme deshalb den Wert, der ihre Regel exakt erfüllt: **σ_b = 0,0412.** Er liegt auch nahe an ihrem besten Rasterwert 0,040. Der V2gb-Punkt hat βq = 861, also weit über L.

## Frage

Ist BPreCap im V2a2-Netz bei 4-Bit-Aktivierungen (w8a4a_best) besser als eine tanh an der kritischen Linie, deren Punkt nach der Regel von Hayou et al. zur Tiefe L = 50 passt? Das V2a2-Netz ist ein CNN mit L = 50, ohne Normierung, ohne Residuen, Delta-orthogonal, auf CIFAR-10.

## Die eine Änderung gegenüber V2gb

Die Form heißt **htanh**: tanh mit Verstärkung 1. Initialisiert wird wie in V2gb, nur mit einem anderen Punkt der kritischen Linie:
- **σ_b = 0,041221694095456875** (σ_b² = 0,0016992), so gewählt, dass βq = 50 = L.
- **σ_w² = 1,2273528768534143:** χ1 = σ_w² E[tanh'(√q* z)²] = 1 im Fixpunkt **q* = 0,131862**.
- Gelöst mit scipy brentq auf Gauß-Hermite mit 301 Punkten (21.11., 18:35). Der Selbsttest rechnet χ1, den Fixpunkt und βq nach.
- **Gewichte:** die Delta-orthogonalen Kerne aus V2a2 mal σ_w.
- **Bias:** je Kanal aus N(0, σ_b²) mit demselben eigenen Generator wie in V2gb (`np.random.default_rng([seed, 7])`). Die Bias-Werte sind also die V2gb-Werte mal 9,22, die Gewichts-Ziehung bleibt bitgleich zu V2a2.
- Die Ausgabeschicht bleibt unverändert.

**Varianzverlauf bei der Init:** Vor dem Freeze am 21.11. um 18:40 gemessen, Seed 42000, 256 Trainingsbilder, Selbsttest komplett gelaufen:

| Schicht | 1 | 2 | 5 | 10 | 25 | 50 |
|---|---|---|---|---|---|---|
| gemessen | 1,179 | 0,418 | 0,190 | 0,149 | 0,138 | 0,136 |
| Mittelfeld | 1,179 | 0,522 | 0,226 | 0,154 | 0,133 | 0,132 |

Das CNN mit Breite 32 folgt der Mittelfeld-Rechnung also gut. Bei der xtanh (V2gb) lag die Varianz nach 50 Schichten bei 0,028, bei der unkalibrierten tanh (V2c) bei 0,010. Das Produkt der χ über 50 Schichten (Mittelfeld, Eingang q = 1,05) ist bei htanh 0,16, bei xtanh 0,04.

Alles andere ist der unveränderte, eingefrorene Code `../v2a2/v2a2_run.py`, wie in V2gb:
- Netz, Daten (inkl. Val-Satz)
- AdamW mit Kosinus auf 0, 3440 Schritte
- Tuning-Regel und frühes Stoppen (8 Punkte)
- PTQ: w8 je Kanal, a4 asymmetrisch je Tensor, Kalibrierungen max/p999/p9999/p99999/mse4, w8a4a_best auf den Kalibrierbildern

## Historische Arme

Alle Arme wurden mit gleichem Code, gleichen Seeds und auf demselben Gerät gerechnet:
- BPreCap, SELU und TAT aus `../v2a2/workers/w0.csv`
- tanh aus `../v2c/workers/w0.csv`
- ctanh aus `../v2d/workers/w0.csv`
- xtanh aus `../v2gb/workers/w0.csv`

Die Kontrolle läuft wie in V2gb über den Selbsttest:
1. BPreCap-Init bitgleich mit der Original-Klasse.
2. htanh-Gewichte genau σ_w mal Original.
3. 20 Trainingsschritte BPreCap bitgleich.

## Tuning

- 3 LR × 2 WD (0 und 3,0), Tuning-Seeds 42000 und 42001, also 12 Läufe. Auswahl wie in V2a2.
- LR-Raster 1e-5, 3e-5, 1e-4 wie in V2c, V2d und V2gb. In V2gb lag die Wahl (3e-5) in der Mitte. Liegt sie hier am Rand, wird das als Grenze berichtet.

## Bestätigung

- 12 Seeds 42350–42361, gepaart mit allen historischen Armen.
- **Soll:** 12 Tuning-Zeilen und 12 Bestätigungszeilen.
- 1 Worker, etwa 8,5 h wie V2gb.

## Primärtests (Holm über 2)

- **P1:** BPreCap − htanh, Test-Fehler w8a4a_best am frühen Stopp (pp).
- **P2:** BPreCap − htanh, 4-Bit-Preis = w8a4a_best − fp32 am frühen Stopp (pp).

Testverfahren und Regeln:
- Wilcoxon-Vorzeichen-Rang-Test, gepaart über 12 Seeds, zweiseitig.
- Ausfall oder fehlende Zeile = 90 %, beim Preis 90 pp.
- Holm-p < 0,05 und Median < 0: „BPreCap besser“.
- Holm-p < 0,05 und Median > 0: „htanh besser“.
- Sonst: „unklar“.

## Sekundärtests (Holm über 4)

| Nr. | Vergleich | Größe | Frage |
|---|---|---|---|
| S1 | BPreCap − htanh | fp32 | Trainierbarkeit |
| S2 | htanh − xtanh (V2gb) | w8a4a_best | Hilft der Punkt nach Hayou gegenüber dem fast linearen Punkt? |
| S3 | htanh − ctanh (V2d) | w8a4a_best | kritische tanh gegen BPreCaps Rezept bei q = 1 |
| S4 | htanh − SELU | w8a4a_best | Reicht die kritische tanh, um SELU zu schlagen? |

Das Urteil steht aus Sicht des erstgenannten Arms: Holm-p < 0,05 und Median < 0 heißt „besser“, Median > 0 heißt „schlechter“, sonst „unklar“.

**Beschreibend:** w8a4a_max, die Endpunkt-Werte, die Mediane je Form, der 4-Bit-Preis, die Kalibrierwahl, der Aktivierungsbereich und die Tuning-Wahl.

## Deutung (vorab)

| Ausgang | Bedeutung für v0.4 |
|---|---|
| P1 „BPreCap besser“ | Auch am Punkt nach Hayou liegt BPreCap vorn. Die Grenze „nur ein Punkt der kritischen Linie“ wird auf „zwei Punkte, q* ≤ 0,13“ verengt. |
| P1 „unklar“ oder „htanh besser“ | Eine tanh am passenden Punkt der Linie ist ein ernsthafter Gegner. v0.4 berichtet das offen und schwächt die Aussage ab. |
| S2 | zeigt, ob die Wahl des Punkts auf der Linie für diese Tiefe und dieses Regime etwas ändert. |

## Vorab-Wahrscheinlichkeiten

| Aussage | p |
|---|---|
| alle 12 htanh-Läufe ok | 0,9 |
| P1 „BPreCap besser“ | 0,85 |
| P1 „htanh besser“ | 0,03 |
| P2 „htanh besser“ (kleinerer Preis) | 0,55 |
| S1 „besser“ (BPreCap in fp32 vor htanh) | 0,85 |
| S2 „besser“ (htanh vor xtanh) | 0,45 |
| S3 „schlechter“ (htanh hinter ctanh) | 0,75 |
| S4 „schlechter“ (htanh hinter SELU) | 0,85 |
| Wahl am Rand des Rasters | 0,25 |

Begründung: Bei q* ≈ 0,13 ist √q* ≈ 0,36. tanh krümmt sich dort etwas stärker als bei xtanh, arbeitet aber noch weitgehend linear (tanh(0,36) ≈ 0,35). Das Gradientenprodukt ist größer (0,16 gegen 0,04), das sollte das Training etwas verbessern. Den Abstand zu ctanh (q = 1, etwa 13 pp bei xtanh) schließt das voraussichtlich nicht.

## Grenzen

- **Weiter nur zwei Punkte der kritischen Linie** (q* = 0,026 und 0,132). Der Punkt q* = 1 mit Bias (σ_w² = 2,153, σ_b² = 0,151) bleibt ungeprüft.
- Hayous Regel stammt aus MLPs mit Breite 300 und SGD/RMSProp über 100 Epochen. Hier: CNN mit Breite 32, AdamW, 3440 Schritte. Der Varianzverlauf bei der Init folgt der Mittelfeld-Rechnung aber gut (Tabelle oben).
- Hayous eigene Angaben zum Regelwert sind nicht stimmig (0,030 in Tabelle 3, 0,05 im Text, nachgerechnet 0,0412).
- Das LR-Raster stammt nicht aus einem eigenen Pilot.
- Die historischen Arme wurden nicht zeitgleich gerechnet.
- Alle Grenzen von V2a2 gelten auch hier: ein Datensatz, ein Netz, nur PTQ je Tensor, M1 Pro.

## Ablauf

1. `launch_v2h.sh` prüft Netzteil und andere Worker. Dann folgen:
   - Freeze und Hash-Prüfung von V2a2, V2c, V2d und V2gb
   - Selbsttest (Fehler nach `logs/selfcheck_err.txt`), Auswertungstest, Rauchtest
   - Start in screen `v2hw0` mit caffeinate und dem Wächter `v2hwache`
2. Ausgewertet wird mit `v2h_analyze.py <V2h-Ordner absolut>` (scipy-venv), mit Hash-Prüfung vorher und nachher. Der Bericht kommt nach `V2h_Auswertung.md`.
