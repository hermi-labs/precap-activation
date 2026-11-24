# V2h – Auswertung

Stand: 22.11.2026, 02:59 (mit `date` geprüft), mit Claude. Präregistrierung: `V2h_Praeregistrierung.md`, Freeze 2026-11-21T18:32:29+0100. Hash-Prüfung vor und nach der Analyse: 23/23 OK. Analyse: `v2h_analyze.py` unverändert, Ausgabe `analysis/v2h_result.json`. PreCap, BPreCap, die Hülle und das Kalibrierungsrezept sind Aarons Ideen.

**Hinweis zu Uhrzeiten:** Drei Uhrzeiten in der eingefrorenen Präregistrierung und im Kommentar von `v2h_run.py` sind falsch (geschätzt, nicht mit `date` geprüft). Die richtigen Zeiten stehen in `meta/zeitkorrektur.md`. Die Reihenfolge Konstanten → Selbsttest → Präregistrierung → Freeze → Start stimmt. Am Inhalt ändert das nichts.

## Urteil

- **P1 (Test-Fehler w8a4a_best am frühen Stopp): „BPreCap besser“.** Median BPreCap − htanh = −13,68 pp, 12 von 12 Seeds, Holm-p = 0,001.
- **P2 (4-Bit-Preis): „htanh besser“.** Median der Differenz +2,56 pp, BPreCap nur an 1 von 12 Seeds kleiner, Holm-p = 0,001. Die tanh am Hayou-Punkt verliert bei 4 Bit weniger. Sie startet aber von einem fp32-Fehler von 59,6 %.
- Deutung nach der vorab festgelegten Tabelle: **Auch am Punkt nach Hayou et al. (2019) liegt BPreCap vorn. Die Grenze „nur ein Punkt der kritischen Linie“ wird auf „zwei Punkte, q* ≤ 0,13“ verengt.**

## Primärtests (Wilcoxon gepaart, zweiseitig, 12 Seeds 42350–42361, Holm über 2)

| Test | Größe | Median BPreCap | Median htanh | Median Δ (pp) | BPreCap kleiner | p | p Holm | Urteil |
|---|---|---|---|---|---|---|---|---|
| P1 | es_test_err_w8a4a_best | 48,07 % | 61,98 % | −13,68 | 12/12 | 0,0005 | 0,001 | BPreCap besser |
| P2 | Preis = w8a4a_best − fp32 | 4,64 pp | 2,04 pp | +2,56 | 1/12 | 0,001 | 0,001 | htanh besser |

Δ je Seed:
- P1: −13,60; −13,74; −15,98; −15,73; −15,74; −12,40; −12,69; −14,55; −13,82; −13,31; −13,62; −12,16
- P2: 2,92; 1,65; −0,55; 3,00; 2,49; 3,11; 1,65; 4,49; 2,63; 1,28; 3,86; 1,08

## Sekundärtests (Holm über 4)

| Nr. | Vergleich | Größe | Median Δ (pp) | erstgenannter kleiner | p Holm | Urteil |
|---|---|---|---|---|---|---|
| S1 | BPreCap − htanh | fp32 | −15,98 | 12/12 | 0,002 | besser |
| S2 | htanh − xtanh (V2gb) | w8a4a_best | −3,95 | 12/12 | 0,002 | besser |
| S3 | htanh − ctanh (V2d) | w8a4a_best | +8,94 | 0/12 | 0,002 | schlechter |
| S4 | htanh − SELU | w8a4a_best | +11,48 | 0/12 | 0,002 | schlechter |

S2: Der Hayou-Punkt hilft der tanh gegenüber dem fast linearen Punkt aus V2gb messbar, um knapp 4 pp an allen 12 Seeds. Es bleibt aber ein großer Abstand zu ctanh (Aarons Rezept, q = 1) und zu SELU.

## Beschreibend

Mediane über 12 Seeds am frühen Stopp (alle Läufe stoppen bei Schritt 3440, also am Ende):

| Form | fp32 | w8a8 | w8a4a_max | w8a4a_best | Preis (pp) | Aktivierungsbereich (Median lo / hi) | Kalibrierwahl |
|---|---|---|---|---|---|---|---|
| BPreCap | 42,88 % | 43,71 % | 49,08 % | 48,07 % | 4,64 | −2,24 / 3,00 | mse4 9, p999 1, p9999 1, p99999 1 |
| SELU | 41,40 % | 43,02 % | 55,87 % | 50,33 % | 8,83 | −1,75 / 5,94 | mse4 7, p999 5 |
| ctanh | 46,25 % | 48,13 % | 53,73 % | 52,54 % | 6,50 | −2,27 / 2,27 | mse4 4, p999 4, p99999 3, max 1 |
| TAT | 40,27 % | 41,36 % | 89,89 % | 61,63 % | 21,52 | −3,70 / 9,10 | mse4 12 |
| **htanh (neu)** | **59,62 %** | **60,34 %** | **62,59 %** | **61,98 %** | **2,04** | −0,97 / 0,97 | mse4 4, p9999 3, p999 2, p99999 2, max 1 |
| xtanh (V2gb) | 64,72 % | 65,15 % | 66,10 % | 65,74 % | 1,06 | −0,94 / 0,94 | p999 4, p99999 4, p9999 3, max 1 |
| tanh (V2c) | 66,53 % | 67,01 % | 67,83 % | 68,11 % | 1,26 | −0,93 / 0,93 | p9999 5, mse4 3, p99999 2, p999 1, max 1 |

- Endpunkt: BPreCap − htanh bei w8a4a_best −13,68 pp (12/0), bei fp32 −15,88 pp (12/0).
- w8a4a_max am frühen Stopp: −12,71 pp (12/0, p = 0,0005).
- Train-Loss (Mittel der letzten 50 Schritte): htanh 1,59, xtanh 1,72, tanh 1,76, BPreCap 1,18.
- Rangfolge bei w8a4a_best, jetzt sieben Formen im selben Netz: **BPreCap 48,1 % < SELU 50,3 % < ctanh 52,5 % < TAT 61,6 % ≈ htanh 62,0 % < xtanh 65,7 % < tanh 68,1 %.** TAT und htanh liegen nahe beieinander, ein gepaarter Test dazu war nicht präregistriert.
- Tuning: Gewählt wurde LR 3e-5, WD 0 (Median Val-Genauigkeit 42,8 %). **Die Wahl liegt nicht am Rand des Rasters.** Val-Fehler am Ende je Tuning-Seed (42000 / 42001): 1e-5 64,9 / 66,0 %; 3e-5 57,0 / 57,5 %; 1e-4 58,6 / 65,1 % (jeweils WD 0). 1e-4 war also an beiden Seeds schlechter, an Seed 42000 aber nur um 1,6 pp. Der Einwand, ein kleines q* vertrage größere Lernraten, trifft in diesem Raster weiter nicht zu.

## Integrität

- Bestätigung 12/12 Zeilen, alle ok, run_ids eindeutig, 0 nicht-endliche Werte, keine fehlenden Seeds.
- Tuning 12/12 ok. Historische Arme 72/72 ok (BPreCap, TAT, SELU aus V2a2, tanh aus V2c, ctanh aus V2d, xtanh aus V2gb).
- Freeze-Hashes der Vorgängerstudien und die CIFAR-Dateien sind über `frozen_sha256.txt` mitgeprüft, bei beiden Prüfungen OK.
- Selbsttest (`meta/selfcheck.json`, vor dem Freeze einmal komplett gelaufen):
  - Fixpunkt ok (Fehler 3·10⁻¹⁷), χ1(q*) = 1,0, βq = 50,0
  - Bias-Std 0,04093 (Soll 0,04122)
  - Varianz der Voraktivierung bei Initialisierung, gemessen gegen Mittelfeld: Schicht 10 0,149 / 0,154, Schicht 25 0,138 / 0,133, Schicht 50 0,136 / 0,132
  - BPreCap-Initialisierung und 20-Schritt-Probe bitgleich, htanh-Gewichte genau σ_w mal Original, Ausgabeschicht unverändert
- Lauf: 21.11. 18:41 (Start) bis 22.11. 02:57, etwa 20–22 min je Lauf, ohne Unterbrechung.
- Vorab-Wahrscheinlichkeiten gegen Ergebnis:
  - alle 12 ok 0,9 → ja
  - P1 BPreCap besser 0,85 → ja
  - P1 htanh besser 0,03 → nein
  - P2 htanh besser 0,55 → ja
  - S1 besser 0,85 → ja
  - S2 besser 0,45 → ja
  - S3 schlechter 0,75 → ja
  - S4 schlechter 0,85 → ja
  - Wahl am Rand 0,25 → nein

## Einordnung

- Am Hayou-Punkt (σ_b = 0,0412, q* = 0,132, βq = 50 = L) arbeitet die tanh etwas weniger linear als am V2gb-Punkt (q* = 0,026). Das hilft ihr messbar: fp32 59,6 % statt 64,7 %, w8a4a_best 62,0 % statt 65,7 %.
- Das Grundbild bleibt wie in V2c und V2gb:
  - Die kritisch initialisierte tanh lernt in diesem Regime deutlich schlechter als BPreCap, ctanh oder SELU.
  - Ihr kleiner 4-Bit-Preis (2,0 pp) ist eine Folge davon, dass sie wenig gelernt hat, kein Vorteil bei der Quantisierung.
- Bemerkenswert: Der Preis der htanh ist mit 2,0 pp fast doppelt so hoch wie bei xtanh (1,1 pp). Je mehr die tanh lernt, desto mehr verliert sie bei 4 Bit. Das passt zum Bild, dass der kleine Preis an der fast linearen Arbeitsweise hängt. Das ist eine Beobachtung an zwei Punkten, kein Test.
- Prior art: Die Recherche vom 21.11. (`docs/literatur/reports/Quantisierungsfehler über die Tiefe.md`) fand keinen Vergleich begrenzter Aktivierungen bei 4-Bit-PTQ gegen eine kritisch initialisierte tanh in normierungsfreien CNNs. Neuheit behaupte ich damit nicht. Das Vertrauen der Recherche liegt bei 0,6–0,65.

## Grenzen

- **Zwei Punkte der kritischen Linie**, beide mit kleinem q* (0,026 und 0,132). Der Punkt q* = 1 mit Bias ist nicht geprüft. BPreCaps Rezept bei q = 1 ohne Bias entspricht ctanh (V2d).
- Hayous eigene Angaben für L = 50 sind nicht ganz stimmig (Rasterbestwert 0,040, „with rule“ 0,030, im Text σ_b = 0,05 für βq ∼ 50). Ich habe den Wert genommen, der ihre Regel exakt erfüllt. Er liegt am nächsten an ihrem Rasterbestwert.
- Regime: AdamW, 3440 Schritte, Kosinus auf 0. Hayou et al. trainieren MLPs der Breite 300 mit SGD/RMSProp über 100 Epochen. Die kritische Initialisierung könnte bei längerem Training mehr bringen.
- LR-Raster ohne eigenen Pilot. Die Wahl lag in der Mitte, 1e-4 war an einem Tuning-Seed knapp dahinter.
- Die historischen Arme wurden nicht zeitgleich gerechnet. Die Prüfungen des Selbsttests ersetzen eine volle Kontrollzeile.
- Ein Datensatz (CIFAR-10), ein Netz (CNN, L = 50, ohne Normierung), nur PTQ der Aktivierungen je Tensor, M1 Pro.

## Nächster Schritt

- v0.4-Entwurf, Abschnitt 6.5 und Grenze in Abschnitt 7: den Hayou-Punkt ergänzen, Grenze auf „zwei Punkte, q* ≤ 0,13“ verengen.
- V2k (Zerlegung des 4-Bit-Fehlers in Abschneiden und Runden an den V2f-Netzen) startet direkt im Anschluss.
