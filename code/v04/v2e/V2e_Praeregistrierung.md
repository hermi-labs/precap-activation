# V2e – Präregistrierung: Woher kommt BPreCaps kleinerer 4-Bit-Preis? (Rauschmessung)

Stand: 19.11.2026, 18:40, vor jedem Messergebnis an trainierten Netzen. Gesehen habe ich nur den Selbsttest an einem untrainierten Netz und den Rauchtest nach 20 Schritten (privat, M1 Pro). PreCap, BPreCap, die Hülle und das Kalibrierungsrezept sind Aarons Ideen, das Messdesign habe ich mit Claude entworfen. Anlass ist V2d (`../v2d/V2d_Auswertung.md`). Start mit Aarons Freigabe vom 19.11. abends („kannst gerne starten“, Mac bis 20.11. 6:30). Eingefroren wird mit dem Start (`meta/frozen_sha256.txt`).

## Frage

In V2d verliert BPreCap durch 4-Bit-Aktivierungen weniger als die gleich kalibrierte tanh (Preis 4,64 gegen 6,50 pp, 11/12), obwohl ctanh den etwas engeren Bereich hat. Liegt das
- **(a)** am **Rundungsrauschen je Schicht**, das also bei BPreCap relativ zum Signal kleiner ist, oder
- **(b)** an der **Weitergabe**, sodass dasselbe relative Rauschen bei BPreCap schwächer bis zu den Logits durchkommt?

## Art der Studie

Eine Messstudie ohne neue Trainingsvariante. Je Form werden die Bestätigungsläufe mit den Seeds 42350, 42351 und 42352 mit dem eingefrorenen Code nachgerechnet:
- BPreCap, SELU und TAT aus V2a2,
- tanh aus V2c,
- ctanh aus V2d.

`v2e_run.py` importiert `v2a2_run.py`, `v2c_run.py` und `v2d_run.py` unverändert. Es wiederholt `run_one` und behält dabei das Netz am frühen Stopp. Die Gewichte speichert es in `gewichte/`.

**Kontrolle je Lauf:** Alle Messspalten (außer Laufzeit und Zeitstempel) müssen als Text gleich der historischen CSV-Zeile sein (`kontrolle_n_diff = 0`). Damit sind die gemessenen Netze genau die ausgewerteten Netze.

## Messung (je Lauf, 1000 Testbilder 0–999, Gewichte fp32, nur Aktivierungen)

Grundlagen:
- Die Kalibrierung ist die im Lauf gewählte `es_w8a4a_best_cal`. Bereiche wie im Lauf: `R.calib` auf den 1000 Kalibrierbildern, Seed des Laufs, asymmetrisch je Tensor, 4 Bit.
- Die Eingänge heißen i = 0…50. Dabei ist 0 das Bild, 1…49 sind die inneren Faltungseingänge, 50 ist der Linear-Eingang nach dem Mittel.
- Alle Normen sind Frobenius-Normen über alle 1000 Bilder.

Messgrößen:
- **r_i** = ‖Q_i(h_i) − h_i‖ / ‖h_i‖ im sauberen Pfad. Das ist das relative Rundungsrauschen am Eingang i, mit Abschneiden.
- **d_i^Q** = ‖z(nur Eingang i gerundet) − z‖ / ‖z‖, gemessen an den Logits z.
- **g_i^Q** = d_i^Q / r_i ist die relative Weitergabe von Eingang i bis zu den Logits.
- **d_i^G, g_i^G**: dasselbe mit Gauß-Rauschen. Es hat je Bildstapel dieselbe Norm wie der Rundungsfehler, und jede Stelle hat einen eigenen Generator.
- **D^Q** = ‖z(alle Eingänge gerundet) − z‖ / ‖z‖, dazu **D^G** mit Gauß-Rauschen an allen Eingängen.
- **Verlauf** ε_l = ‖h_l^Q − h_l‖ / ‖h_l‖ nach jeder Schicht.
- **Superposition** D / √(Σ d_i²).
- Weiter: die Kippquote, die Fehler auf den 1000 Bildern (fp32, a4, w8a4), der mittlere relative Logit-Abstand und das empirische χ je Schicht, also E[f'(u)²] an den echten Vor-Aktivierungen, und der Anteil des Rundungsfehlers, der je Kanal konstant ist (Bias-Anteil).

Für die Tests werden je Lauf Kennzahlen gebildet:
- **r̄** = geometrisches Mittel von r_i über i = 1…49,
- **ḡ^Q** und **ḡ^G** ebenso.

## Primärtests (BPreCap gegen ctanh, je Seed gepaart)

Mit 3 Seeds ist ein Wilcoxon-Test nicht möglich, deshalb gilt eine feste Vorzeichen- und Schwellenregel. Das Verhältnis ist q_s = Kennzahl(BPreCap) / Kennzahl(ctanh) je Seed s.

- **Pa (Rundungsrauschen):** Die Kennzahl ist r̄.
- **Pb (Weitergabe):** Die Kennzahl ist ḡ^Q.

Entscheidungsregel, für Pa und Pb einzeln:
- q_s < 1/1,1 bei allen 3 Seeds: **„BPreCap kleiner“**.
- q_s > 1,1 bei allen 3 Seeds: **„BPreCap größer“**.
- Sonst: **„kein klarer Unterschied“**.

Ein Seed ohne gültiges Paar (Ausfall oder `kontrolle_n_diff ≠ 0`) verhindert ein gerichtetes Urteil. Das Ergebnis ist dann „unklar (Kontrolle)“.

**Prüfung der Messung (vorab):** D^Q(BPreCap) < D^Q(ctanh) bei mindestens 2 von 3 Seeds. Ohne das gibt die Messung an den 1000 Bildern den V2d-Preis nicht wieder, und die Deutung von Pa und Pb wird nur beschreibend berichtet.

## Deutung (vorab)

| Pa | Pb | Deutung |
|---|---|---|
| kleiner | kein klarer | BPreCaps Vorteil entsteht **beim Runden**: Die Aktivierungen passen relativ besser ins 4-Bit-Raster. |
| kein klarer | kleiner | BPreCaps Vorteil entsteht **bei der Weitergabe**: Das Netz dämpft Rauschen stärker (Bild vom geraden Rohr). |
| kleiner | kleiner | beides |
| sonst | | Der Unterschied liegt in keiner der beiden Größen allein, etwa in nichtlinearem Zusammenwirken (Superposition) oder in den Logit-Abständen. Das wird beschreibend berichtet. |

## Sekundär (gleiche Regel, ohne Korrektur, als Hinweise)

- **S1:** ḡ^G, BPreCap gegen ctanh. Zählt bei Gauß-Rauschen nur die Weitergabe, oder ist der Rundungsfehler strukturiert (Abschneiden, Korrelation)?
- **S2:** BPreCap gegen SELU, r̄ und ḡ^Q.
- **S3:** g^Q / g^G je Form (Median über Seeds), beschreibend. Ist der Wert über 1, wirkt Rundungsrauschen stärker als zufälliges Rauschen gleicher Norm.
- **S4:** über alle 5 Formen die Spearman-Rangfolge der Mediane von D^Q, r̄ und ḡ^Q gegen den 4-Bit-Preis aus V2a2/V2c/V2d (Median über alle 12 Seeds), beschreibend. Bei 5 Punkten ist das kein Test.
- **Zerlegung:** log(D^Q-Verhältnis) gegen log(r̄-Verhältnis) + log(ḡ^Q-Verhältnis), beschreibend.

## Vorab-Wahrscheinlichkeiten

| Aussage | p |
|---|---|
| alle 15 Läufe bitgleich (`kontrolle_n_diff = 0`) | 0,85 |
| Prüfung der Messung erfüllt | 0,8 |
| Pa „BPreCap kleiner“ | 0,4 |
| Pa „BPreCap größer“ | 0,15 |
| Pb „BPreCap kleiner“ | 0,4 |
| Pb „BPreCap größer“ | 0,15 |
| S3: g^Q / g^G > 1 bei BPreCap | 0,5 |

Begründung: Die Schrittweite ist ähnlich (Bereich 5,24 gegen 4,59). Bei r entscheidet deshalb, wie die Werte im Bereich verteilt sind. Bei g entscheidet die Steigung an den echten Arbeitspunkten. Ich habe keine starke Erwartung, welche der beiden Größen den Unterschied trägt.

## Grenzen

- 3 Seeds je Form, eine Schwellenregel statt eines Tests, 1000 Testbilder.
- Nur die Aktivierungsquantisierung (Gewichte fp32). Der Preis in V2d enthält auch w8.
- Die Weitergabe ist eine relative Größe über Frobenius-Normen. Sie sagt nichts darüber, welche Richtungen im Logit-Raum betroffen sind.
- Es sind Netze aus V2a2/V2c/V2d. Es gelten alle Grenzen dieser Studien: ein Datensatz, ein Netz, nur PTQ je Tensor, M1 Pro.

## Ablauf

1. `launch_v2e.sh` prüft das Netzteil und andere Worker. Dann folgen der Freeze (dieser Ordner, V2a2-, V2c- und V2d-Code, CSVs, Daten), die Hash-Prüfung von V2a2/V2c/V2d, der Selbsttest, der Auswertungstest, der Rauchtest (2 Läufe à 20 Schritte, 250 Bilder) und der Start in screen `v2ew0` mit caffeinate und dem Wächter `v2ewache`.
2. Reihenfolge: erst die Seeds 42350–42352 mit BPreCap, ctanh und SELU, dann tanh und TAT. Soll: 15 Zeilen.
3. Auswertung mit `v2e_analyze.py <V2e-Ordner absolut>` (scipy-venv), Hash-Prüfung vorher und nachher, Bericht `V2e_Auswertung.md`.
