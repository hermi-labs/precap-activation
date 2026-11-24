# V2e – Auswertung: Woher kommt BPreCaps kleinerer 4-Bit-Preis?

Stand: 20.11.2026, 03:33. Präregistrierung: `V2e_Praeregistrierung.md`, Freeze 2026-11-19T18:42:34+0100. Hash-Prüfung vor und nach der Analyse: 22/22 OK. Läufe 19.11. 18:59 bis 20.11. 02:26 (M1 Pro, 1 Worker). Ausgewertet mit dem eingefrorenen `v2e_analyze.py`, Ausgabe `analysis/v2e_result.json`. Zusätzliche Auswertungen sind als explorativ markiert (`analysis/v2e_explorativ.py`, `../v2e_kal/`). PreCap, BPreCap, die Hülle und das Kalibrierungsrezept sind Aarons Ideen, das Messdesign und dieser Bericht entstanden mit Claude.

## Urteil

- **Pa (Rundungsrauschen r̄):** „kein klarer Unterschied“. q = 0,996 / 1,105 / 0,978.
- **Pb (Weitergabe ḡ^Q):** „kein klarer Unterschied“. q = 1,148 / 0,859 / 1,172.
- **Prüfung der Messung:** erfüllt. D^Q ist bei BPreCap in 3 von 3 Seeds kleiner.
- **Deutung nach Vorab-Tabelle (Zeile „sonst“):** BPreCaps kleinerer Preis steckt in keiner der beiden Größen allein. Er entsteht im Zusammenwirken der Schichten. Das wird unten beschreibend berichtet.

In einem Satz: BPreCap rundet nicht feiner als ctanh, bei gleicher Kalibrierung sogar etwa 10 % gröber. Der Vorteil entsteht erst im Zusammenspiel der Schichten. Bei ctanh addieren sich die Rundungsfehler gleichgerichteter (9 von 9 Vergleichen in der Nachmessung), und ctanh verträgt die abschneidende Kalibrierung schlechter. Das ist explorativ und mit 3 Seeds gemessen.

## Primärtests (BPreCap gegen ctanh, je Seed)

| Seed | r̄ BPreCap | r̄ ctanh | q (Pa) | ḡ^Q BPreCap | ḡ^Q ctanh | q (Pb) | D^Q BPreCap | D^Q ctanh |
|---|---|---|---|---|---|---|---|---|
| 42350 | 0,0813 | 0,0817 | 0,996 | 0,205 | 0,179 | 1,148 | 0,259 | 0,366 |
| 42351 | 0,0828 | 0,0749 | 1,105 | 0,229 | 0,267 | 0,859 | 0,268 | 0,290 |
| 42352 | 0,0809 | 0,0827 | 0,978 | 0,207 | 0,177 | 1,172 | 0,274 | 0,384 |
| Regel | | | kein klarer Unterschied | | | kein klarer Unterschied | 3/3 kleiner | |

## Sekundär (Hinweise, ohne Korrektur)

- **S1, Gauß-Weitergabe ḡ^G:** „kein klarer Unterschied“ (q = 1,009 / 1,113 / 1,065).
- **S2, BPreCap gegen SELU:** r̄ „BPreCap kleiner“ (q = 0,793 / 0,725 / 0,769), ḡ^Q „kein klarer Unterschied“ (0,945 / 0,673 / 1,092). Gegen SELU entsteht der Vorteil also beim Runden: Der begrenzte Bereich gibt feinere Stufen. Das passt zu V2a2.
- **S3, g^Q/g^G (Median über Seeds):** Der Wert liegt bei allen Formen über 1: BPreCap 1,54, ctanh 1,40, SELU 1,40, tanh 2,16, TAT 2,19. Rundungsrauschen wirkt stärker als Gauß-Rauschen gleicher Norm. Es ist strukturiert, etwa durch Abschneiden und Korrelation zwischen Kanälen.
- **S4, Rangfolge über 5 Formen** gegen den 4-Bit-Preis aus V2a2, V2c und V2d (Median über 12 Seeds: tanh 1,26, BPreCap 4,64, ctanh 6,50, SELU 8,83, TAT 21,52 pp). Spearman: D^Q 1,0, ḡ^Q 0,9, ḡ^G 0,9, r̄ 0,7. Bei 5 Punkten ist das kein Test. Die Messung ordnet die Formen aber genau wie der Preis.
- **Zerlegung** log(D^Q-Verhältnis) = log r̄ + log ḡ^Q + Rest:

| Seed | log D | log r̄ | log ḡ^Q | Rest |
|---|---|---|---|---|
| 42350 | −0,346 | −0,004 | +0,138 | −0,480 |
| 42351 | −0,080 | +0,100 | −0,152 | −0,029 |
| 42352 | −0,336 | −0,023 | +0,158 | −0,472 |

  Bei zwei Seeds steckt der ganze Unterschied im Rest. Der Rest ist der Teil, den die beiden Mittelwerte nicht erfassen.

### Mediane über die 3 Seeds (beschreibend)

| Form | r̄ | ḡ^Q | ḡ^G | D^Q | D^G | Superposition Q | Superposition G | Kippquote | Fehler fp32 / a4 (1000 Bilder) | Bias-Anteil |
|---|---|---|---|---|---|---|---|---|---|---|
| BPreCap | 0,081 | 0,207 | 0,135 | 0,268 | 0,280 | 1,05 | 1,34 | 0,239 | 44,4 / 48,4 % | 0,12 % |
| ctanh | 0,082 | 0,179 | 0,128 | 0,366 | 0,334 | 1,20 | 1,33 | 0,324 | 46,9 / 52,3 % | 0,008 % |
| SELU | 0,105 | 0,217 | 0,155 | 0,396 | 0,345 | 1,04 | 1,17 | 0,377 | 42,2 / 52,0 % | 0,05 % |
| tanh | 0,092 | 0,127 | 0,059 | 0,156 | 0,104 | 1,11 | 1,13 | 0,198 | 66,8 / 69,2 % | 0,04 % |
| TAT | 0,209 | 0,438 | 0,200 | 0,692 | 0,515 | 0,98 | 1,13 | 0,600 | 40,7 / 62,7 % | 0,50 % |

## Explorativ: Wo steckt der Unterschied? (nicht präregistriert)

Quelle: `analysis/v2e_explorativ.txt`.

**1. Superposition.** Werden alle Schichten zugleich gerundet, addieren sich die Fehler bei BPreCap fast wie unabhängige Beiträge: D^Q / √(Σ d_i²) = 1,05 / 1,00 / 1,06. Bei ctanh addieren sie sich stärker gleichgerichtet: 1,20 / 1,05 / 1,25. Bei Gauß-Rauschen sind beide Formen gleich (1,34 gegen 1,33). Die stärkere gleichgerichtete Addition gibt es bei ctanh also nur beim Rundungsfehler. Deshalb ist bei BPreCap D^Q kleiner als D^G (0,268 gegen 0,280), bei ctanh größer (0,366 gegen 0,334).

**2. Frühe Schichten.** Fasst man die Einzelbeiträge als √(Σ d_i²) zusammen, statt das geometrische Mittel zu nehmen, ist ctanh bei 42350 und 42352 um etwa 20 % schlechter. Das liegt fast ganz am Bild-Eingang (0,151 gegen 0,122–0,127) und an den Schichten 1–10 (0,25 gegen 0,18–0,19). Ab Schicht 11 ist ctanh dort gleich oder etwas besser. Das geometrische Mittel über 49 Eingänge gewichtet alle Schichten gleich und verdeckt diesen Unterschied. Das ist eine Schwäche der vorab gewählten Kennzahl.

| Zerlegung log D^Q | 42350 | 42351 | 42352 |
|---|---|---|---|
| log √(Σ d_i²) (Einzelbeiträge) | −0,219 | −0,032 | −0,173 |
| log Superposition | −0,127 | −0,049 | −0,163 |

**3. Kalibrierung.** BPreCap wählte in allen 3 Seeds die Kalibrierung mse4, die abschneidet. ctanh wählte bei 42350 p99999 und bei 42352 max, beide weit, und nur bei 42351 mse4. Gerade bei 42351 ist der Unterschied in D^Q klein (0,268 gegen 0,290). Über alle 12 Seeds aus V2a2 und V2d:
- BPreCap wählt mse4 9-mal, ctanh nur 4-mal.
- BPreCap hat bei **jeder** festen Kalibrierung den kleineren Preis, etwa bei mse4 −2,04 pp (10/12) und bei max −0,95 pp (10/12).

Die Kalibrierwahl erklärt den Vorteil also nicht allein. Sie verstärkt ihn aber: BPreCap verträgt das Abschneiden besser.

**4. Nachmessung mit fester Kalibrierung** (`../v2e_kal/`, gleiche Gewichte, eingefrorene Messfunktion): Kontrolle bestanden: BPreCap 42350 mse4 gibt die V2e-Messung exakt wieder. Werte BPreCap/ctanh je Seed (42350 / 42351 / 42352), Tabelle in `../v2e_kal/kal_tabelle.txt`:

| Kalibrierung | r̄: BPreCap kleiner | ḡ^Q: q | Superposition: q | D^Q: q | D^Q: BPreCap kleiner |
|---|---|---|---|---|---|
| mse4 (schneidet ab) | 0/3 (q 1,08–1,11) | 0,76 / 0,86 / 0,75 | 0,90 / 0,95 / 0,98 | 0,78 / 0,92 / 0,85 | 3/3 |
| max | 0/3 (q 1,11–1,13) | 1,03 / 1,23 / 1,08 | 0,96 / 0,99 / 0,84 | 0,89 / 1,08 / 0,84 | 2/3 |
| p99999 | 0/3 (q 1,11–1,14) | 1,03 / 1,22 / 1,08 | 0,96 / 0,98 / 0,81 | 0,90 / 1,05 / 0,82 | 2/3 |

Bei gleicher Kalibrierung ergibt sich ein klareres Bild:
- **ctanh rundet feiner** (r̄ etwa 10 % kleiner, 9/9). Das passt zum engeren Bereich. In der präregistrierten Messung war das verdeckt, weil ctanh dort meist weite Kalibrierungen nutzte.
- **Bei ctanh addieren sich die Rundungsfehler gleichgerichteter** (Superposition 9/9 größer). Das ist der beständigste Befund.
- **ctanh verträgt das Abschneiden schlechter.** Bei mse4 gibt ctanh die Fehler um 15–33 % stärker weiter (3/3). Bei weiten Kalibrierungen ist die Weitergabe ähnlich.
- Netto ist D^Q bei mse4 3/3 kleiner für BPreCap, bei den weiten Kalibrierungen 2/3. BPreCaps feinere Rundung ist es also nicht. Es ist eine geringere Kohärenz der Fehler über die Tiefe und eine größere Robustheit gegen abgeschnittene Ränder. Beides ist mit 3 Seeds explorativ.

**5. Bias-Anteil.** Der je Kanal konstante Anteil des Rundungsfehlers ist winzig (Median BPreCap 0,12 %, ctanh 0,008 %). Die Vermutung aus dem Rauchtest, ein gleichgerichteter Anteil überlebe das Mitteln und erkläre den Preis, war falsch.

**6. Das Bild vom geraden Rohr** aus der V2d-Übersicht (BPreCap dämpft Störungen stärker) stützt die Messung so nicht. Eine einzelne gestörte Schicht wirkt bei BPreCap im Mittel nicht schwächer, bei 2 von 3 Seeds sogar etwas stärker.

## Integrität

- 15/15 Zeilen, 15 eindeutige run_id, 15 ok, keine nicht-endlichen Werte, alle Paare vollständig.
- Alle 15 Läufe sind bitgleich zu den historischen CSV-Zeilen (`kontrolle_n_diff = 0`). Gemessen wurden also genau die Netze aus V2a2, V2c und V2d.
- Hash-Prüfung vor und nach der Analyse 22/22 OK.
- Sicherung: `sicherung/v2e_2026-11-20/` mit `SHA256.txt` (Gewichte, Messungen, CSV, Analyse, Logs).

## Vorab-Wahrscheinlichkeiten gegen Ergebnis

| Aussage | p vorab | eingetreten |
|---|---|---|
| alle 15 bitgleich | 0,85 | ja |
| Prüfung der Messung erfüllt | 0,8 | ja (3/3) |
| Pa „BPreCap kleiner“ | 0,4 | nein |
| Pa „BPreCap größer“ | 0,15 | nein |
| Pb „BPreCap kleiner“ | 0,4 | nein |
| Pb „BPreCap größer“ | 0,15 | nein (2 von 3 Seeds über 1,1) |
| S3 g^Q/g^G > 1 bei BPreCap | 0,5 | ja (1,54) |

Für beide Primärtests war „kein klarer Unterschied“ mit 0,45 vorab ebenso wahrscheinlich wie „kleiner“. Die Deutungszeile „sonst“ war vorab vorgesehen.

## Einordnung

- **V2d** zeigte: Bei gleichem Rezept hat BPreCap den kleineren 4-Bit-Preis (−2,04 pp, 11/12). V2e zeigt, dass das nicht an feineren Stufen liegt (r̄ gleich, bei gleicher Kalibrierung bei ctanh sogar kleiner). Eine im Mittel schwächere Weitergabe einzelner Störungen ist es nach der Vorab-Regel auch nicht. Der Unterschied liegt im Zusammenspiel: Die Rundungsfehler der Schichten addieren sich bei ctanh gleichgerichteter, die frühen Schichten tragen mehr bei, und ctanh verträgt die abschneidende Kalibrierung schlechter.
- **V2c und V2a2:** Gegen SELU entsteht der Vorteil dagegen schon beim Runden (r̄ 3/3 kleiner). Der Bereich wirkt also gegen unbegrenzte Formen, die Form wirkt gegen eine gleich begrenzte tanh. Beides passt zur v0.4-Aussage, nur der Mechanismus der Form ist jetzt genauer und anders als vermutet.
- **Literatur:** Dass Quantisierungsfehler über Schichten korreliert sind und sich nicht wie unabhängiges Rauschen addieren, ist bekannt (z. B. [Nagel et al. 2020, AdaRound](https://arxiv.org/abs/2004.10568), [Nagel et al. 2021, White Paper](https://arxiv.org/abs/2106.08295)). Neu wäre höchstens der Befund, dass die Form der Aktivierung diese Kohärenz bei gleicher Kalibrierung ändert. Das ist nicht gegen die Literatur geprüft und wird nicht als neu behauptet.

## Grenzen

- 3 Seeds je Form und eine Schwellenregel statt eines Tests.
- Je Seed passt die Messung nicht immer zum V2d-Preis. Bei 42351 ist der Preisunterschied am größten (etwa 3,7 gegen 5,8 pp), der D^Q-Unterschied aber am kleinsten. Die Frobenius-Norm der Logits erfasst nicht alles, was über die Klasse entscheidet.
- Gemessen wurden nur die Aktivierungen bei fp32-Gewichten und 1000 Testbildern.
- Die vorab gewählten Kennzahlen (geometrisches Mittel über die Schichten) gewichten alle Schichten gleich. Der Fund zu den frühen Schichten und zur Superposition ist explorativ.
- Ein Netz, ein Datensatz, PTQ je Tensor, M1 Pro. Es gelten alle Grenzen aus V2a2, V2c und V2d.

## Nächster Schritt (Vorschlag, Aaron entscheidet)

- In v0.4 Abschnitt 6.4 bleibt die Formerklärung eine Vermutung. Ergänzt wird ein vorsichtiger Absatz: Gemessen ist BPreCaps Vorteil nicht feinere Rundung, sondern geringere Kohärenz der Rundungsfehler über die Tiefe und Robustheit gegen abgeschnittene Ränder. Das gilt mit 3 Seeds und ist explorativ. Der Absatz steht schon im privaten Entwurf.
- Wenn der Mechanismus in v0.4 tragen soll: eine kleine präregistrierte Folgestudie an den gespeicherten Gewichten (ohne Training, unter 1 h). Primärgröße wäre die Superposition bei fester Kalibrierung, dazu die Korrelation der Fehler zwischen den Schichten. Dafür bräuchte es mehr Seeds mit Gewichten, also neue Läufe (BPreCap etwa 40 min, ctanh etwa 25 min je Lauf).

**Vertrauen:** 0,85 für die Primärurteile (Messung bitgleich und vollständig), 0,6 für die Deutung (explorativ, 3 Seeds).
