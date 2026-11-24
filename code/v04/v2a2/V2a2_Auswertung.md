# V2a2 – Auswertung (privat, mit Claude, 17.11.2026, 16:50)

Stand: 17.11.2026. Präregistrierung: `V2a2_Praeregistrierung.md`, Freeze 2026-11-16T08:04:21+0100, Hash-Prüfung vor und nach der Analyse: OK (keine Abweichung). PreCap, BPreCap und die Hülle sind Aarons Ideen, die Auswertung habe ich mit Claude gemacht. Lauf: 16.11. 08:53 bis 17.11. 16:45, ein Worker (`v2a2w0`), am Netzteil.

Befehle (im Projektordner, 16:48):
```
(cd studien/v2a2 && shasum -a 256 -c meta/frozen_sha256.txt)        # alle OK, vorher und nachher
python studien/v2a2/v2a2_analyze.py <absoluter Pfad>/studien/v2a2
```
Ausgabe: `analysis/v2a2_result.json`.

## Urteil

- **P1 (gegen SELU): „besser“.** BPreCap − SELU, Test-Fehler w8a4a_best am frühen Stopp: −2,55 pp, 12/0, Holm-p = 0,0010.
- **P2 (gegen TAT): „besser“.** BPreCap − TAT, ebenso: −13,73 pp, 12/0, Holm-p = 0,0010.
- **Gesamt: Erfolg.** Nach der Deutungsregel trägt der 4-Bit-Vorteil vom MLP (B10) und von V2a-1 ins tiefe CNN ohne Normierung (L = 50), mit getuntem Regime, gegen beide Referenzen.
- **Preis:** S1 und S2 sind „schlechter“. BPreCap ist in fp32 schlechter als SELU (+1,64 pp) und als TAT (+3,16 pp). Nach der Vorab-Regel heißt das: Der Vorteil kostet fp32-Güte, und das gehört in die Zusammenfassung.

## Primärtests (Holm über 2)

| Test | Kandidat | Referenz | Median Kandidat | Median Referenz | Δ (pp, Median je Seed) | Siege | p (Holm) | Urteil |
|---|---|---|---|---|---|---|---|---|
| P1 | BPreCap | SELU | 48,07 % | 50,33 % | −2,55 | 12/0 | 0,0010 | besser |
| P2 | BPreCap | TAT | 48,07 % | 61,63 % | −13,73 | 12/0 | 0,0010 | besser |

Größe: es_test_err_w8a4a_best. Die Mediane der Arme sind Mediane über Seeds. Δ ist der Median der gepaarten Differenzen, deshalb ist Δ nicht genau die Differenz der Mediane.

## Sekundärtests (Holm über 5)

| Test | Größe | Δ BPreCap − Ref. (pp) | Siege | p (Holm) | Urteil | vorab |
|---|---|---|---|---|---|---|
| S1 | fp32 gegen SELU | +1,64 | 1/11 | 0,0024 | schlechter | „unklar“ 0,55 |
| S2 | fp32 gegen TAT | +3,16 | 1/11 | 0,0024 | schlechter | „unklar“ 0,5 |
| S3 | w8a4a_max gegen SELU | −6,08 | 12/0 | 0,0024 | besser | 0,75 |
| S4 | w8a4a_max gegen TAT | −40,72 | 12/0 | 0,0024 | besser | 0,95 |
| S5 | w8a4a_best gegen TAT nach CLE | −15,53 | 12/0 | 0,0024 | besser | 0,6 |

## Beschreibend (präregistriert, ohne Test)

Mediane über 12 Seeds, Test-Fehler am frühen Stopp:

| Form | LR / WD | fp32 | w8a8 | w8a4a_max | w8a4a_best | 4-Bit-Preis (best − fp32) | häufigste Kalibrierung | Aktivierung unten / oben / |max| |
|---|---|---|---|---|---|---|---|---|
| BPreCap | 3e-5 / 0 | 42,88 % | 43,71 % | 49,08 % | 48,07 % | 4,64 pp | mse4 (9×), p999/p9999/p99999 je 1× | −2,24 / 3,00 / 3,00 |
| SELU | 3e-5 / 0 | 41,40 % | 43,02 % | 55,87 % | 50,33 % | 8,83 pp | mse4 (7×), p999 (5×) | −1,75 / 5,94 / 29,1 |
| TAT | 1e-4 / 0 | 40,27 % | 41,36 % | 89,89 % | 61,63 % | 21,52 pp | mse4 (12×) | −3,70 / 9,10 / 158,2 |
| TAT nach CLE | wie TAT | 40,27 % | 41,81 % | 89,97 % | 63,91 % | – | mse4 | −3,90 / 9,68 / 150,2 |

- **Tuning:** 36/36 ok. Keine Wahl liegt am Rand des Rasters (vorab 0,6 für „mindestens eine am Rand“). WD 3,0 hat bei allen Formen geschadet; gewählt ist überall WD 0.
- **Frühes Stoppen:** Der Stopp-Punkt liegt im Median bei Schritt 3440 (Ende), Minimum 3010. Messwerte am Ende und am Stopp sind fast gleich (BPreCap w8a4a_best am Ende 48,17 %).
- **CLE:** Die fp32-Funktion bleibt gleich (es_cle_fp32_delta = 0,0). Getestet ist nur S5.
- **Bereich:** BPreCap bleibt durch die Hülle bei höchstens 3,0. SELU (|max| 29) und TAT (|max| 158) haben breite Ausreißer. Der Mechanismus aus V2a-1 ist also vorhanden; die Deutungsregel „Bereich bleibt klein“ greift nicht.
- **Ende statt Stopp:** Gegen SELU −2,60 pp (12/0), gegen TAT −13,70 pp (12/0) bei w8a4a_best, gleiche Richtung.

## Explorativ (nicht präregistriert, nachträglich gerechnet)

- **4-Bit-Preis je Seed** (w8a4a_best − fp32): BPreCap 3,5–6,2 pp, SELU 6,8–12,1 pp, TAT 16,9–41,2 pp. Gepaart ist der Preis von BPreCap gegenüber SELU im Median um 4,0 pp kleiner (12/12), gegenüber TAT um 16,5 pp (12/12).
- **CLE schadet TAT hier eher:** w8a4a_best mit CLE minus ohne CLE: Median +1,98 pp, schlechter bei 7 von 12 Seeds. Eine Erklärung, die ich nicht geprüft habe: CLE gleicht die Gewichtsbereiche an. Bei Aktivierungen je Tensor verschiebt es Ausreißer zwischen Kanälen, ohne sie zu entfernen. |max| bleibt bei etwa 150, der obere Bereich wird sogar etwas breiter (9,68 statt 9,10).

## Integrität

| Prüfung | Ergebnis |
|---|---|
| Zeilen | Bestätigung 36/36 ok, Tuning 36/36 ok, 0 Ausfälle, 0 fehlend |
| run_id | eindeutig |
| nicht-endliche Werte | 0 |
| Tracebacks | 0 in `w0.log` und `smoke.log` |
| Hashes | `shasum -a 256 -c meta/frozen_sha256.txt`: alle OK, vor und nach der Analyse |
| Schlaf | ein Software-Sleep (DarkWake) am 17.11. um 15:19:44 für 13 s, am Netzteil. Die Ergebnisse ändert das nicht, nur die Laufzeit |
| Mechanik vorab | `meta_vorab/` (bitgleich zum Pilot-Code bei 20 Schritten, CLE funktionsgleich) |

## Vorab-Wahrscheinlichkeiten gegen Ausgang

| Aussage | vorab | eingetreten |
|---|---|---|
| P1 „besser“ | 0,65 | ja |
| P2 „besser“ | 0,8 | ja |
| Gesamt „Erfolg“ | 0,55 | ja |
| ein Primärtest „schlechter“ | 0,05 | nein |
| S1 „unklar“ | 0,55 | nein (schlechter) |
| S2 „unklar“ | 0,5 | nein (schlechter) |
| S3 „besser“ | 0,75 | ja |
| S4 „besser“ | 0,95 | ja |
| S5 „besser“ | 0,6 | ja |
| mindestens eine Wahl am Rand | 0,6 | nein |

Die fp32-Kosten habe ich unterschätzt: Erwartet war „unklar“, eingetreten ist „schlechter“ mit 11/1.

## Einordnung

- **V2a-1 (CNN L = 32, einfaches Training, ohne Tuning):** BPreCap war dort auch in fp32 vorn (−2,68 pp gegen SELU). Mit L = 50, Tuning je Form und Kosinus-Plan dreht sich das: In fp32 sind SELU und TAT besser. Der 4-Bit-Vorteil bleibt trotzdem, weil der 4-Bit-Preis von BPreCap etwa halb so groß ist wie bei SELU und etwa ein Fünftel von TAT.
- **B10–B13 (MLP, AdamW, Kosinus):** Dort war der Vorteil gegen TAT bei 4 Bit ebenfalls deutlich. Hier ist er mit −13,7 pp am frühen Stopp noch größer, weil TATs Aktivierungen im tiefen CNN extreme Ausreißer haben.
- **Literatur:** Dass Begrenzung der Quantisierung hilft, ist alt (ReLU6, PACT, Clipping bei PTQ). In `docs/literatur/` ist kein Vergleich einer tiefenkalibrierten begrenzten Aktivierung mit SELU und TAT in normierungsfreien CNNs bei 4-Bit-PTQ verzeichnet. Das ist eine Lücke in der geprüften Literatur, keine bewiesene Neuheit.

## Grenzen

- **Absolute Güte:** Niedrig, etwa 58 % Testgenauigkeit in fp32. Getestet ist ein sehr tiefes, schlichtes CNN ohne Normierung und ohne Residuen, kein praxisnahes Netz.
- **Umfang:** ein Datensatz (CIFAR-10), eine Tiefe (L = 50), eine Breite (32), ein Gerät (M1 Pro).
- **fp32-Preis:** Wer nicht quantisiert, ist hier mit SELU oder TAT besser bedient.
- **Methoden:** nur PTQ, ohne Bias-Ausgleich, AdaRound, Rotationen (QuaRot-artig), QAT, PACT oder LSQ. Stärkere PTQ-Methoden könnten vor allem TATs Abstand verkleinern.
- **Aktivierungen je Tensor:** je Kanal würde TAT und SELU vermutlich stärker helfen als BPreCap.
- **Frühes Stoppen:** fast immer am Ende. Längeres Training ist nicht geprüft.
- **Ohne tanh:** Der Vergleich trennt nicht, ob der Vorteil von der Begrenzung allein oder von der Tiefenkalibrierung kommt.

## Nächster Schritt

- **v0.4:** Abschnitt 6.2 mit diesen Zahlen schreiben. Die Kernaussage: „bei 4-Bit-PTQ besser als SELU und TAT, auf Kosten von 1,6–3,2 pp in fp32“.
- **Lücken schließen:** tanh als vierte Form, um Begrenzung und Kalibrierung zu trennen, und eine Robustheitsprüfung mit Rotation oder Aktivierungen je Kanal. Beides ist offen und die naheliegende Folge-Studie. Den Start entscheidet Aaron, solange der Mac nicht frei ist.
- **Sicherung:** `sicherung/v2a2_2026-11-17/`.
