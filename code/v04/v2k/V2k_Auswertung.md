# V2k – Auswertung

Stand: 22.11.2026, 05:55 (mit `date` geprüft), mit Claude. Präregistrierung: `V2k_Praeregistrierung.md`, Freeze 2026-11-22T03:00:32+0100, nach komplettem Selbsttest (02:58–03:00). Hash-Prüfung vor und nach der Analyse: 71/71 OK. Analyse: `v2k_analyze.py` unverändert, Ausgabe `analysis/v2k_result.json`. PreCap, BPreCap, die Hülle und das Kalibrierungsrezept sind Aarons Ideen. Messstudie an den 18 V2f-Netzen, kein Training.

## Urteil

- **K1 (Abschneiden allein, alle Schichten, mse4): „bestätigt“.** D^C ist bei BPreCap kleiner, Median q = 0,81, 9 von 9 Seeds, p_Holm = 0,004.
- **K2 (Vorteil beim Abschneiden größer als beim Runden): „bestätigt“.** Median log q(D^C) − log q(D^R) = −0,29, 8 von 9 in Richtung, p_Holm = 0,004.
- Deutung nach der vorab festgelegten Tabelle: **Der Vorteil sitzt im Abschneiden. Die V2f-Deutung („BPreCap verträgt abgeschnittene Ränder besser“) ist gestützt.**

## Primärtests (Kalibrierung mse4, q = BPreCap / ctanh je Seed, 9 Paare, exakter Wilcoxon einseitig, Holm über 2)

| Test | Größe | Median | q (Median) | in Richtung | p einseitig | p zweiseitig | p Holm | Urteil |
|---|---|---|---|---|---|---|---|---|
| K1 | log q(D^C) | −0,213 | 0,81 | 9/9 | 0,002 | 0,004 | 0,004 | bestätigt |
| K2 | log q(D^C) − log q(D^R) | −0,288 | – | 8/9 | 0,004 | 0,008 | 0,004 | bestätigt |

Werte je Seed (42353 … 42361):
- K1: −0,213; −0,632; −0,367; −0,199; −0,200; −0,143; −0,590; −0,058; −0,375
- K2: −0,288; −0,500; −0,351; −0,120; −0,325; −0,096; −0,506; +0,051; −0,268

## Sekundärtests (mse4, zweiseitig, Holm über 4)

| Nr. | Größe | Median | q | BPreCap kleiner | p Holm | Urteil |
|---|---|---|---|---|---|---|
| S1 | log q(ḡ^R), Weitergabe des reinen Rundungsfehlers | +0,096 | 1,10 | 1/9 | 0,016 | Unterschied (BPreCap größer) |
| S2 | log q(ḡ^C), Weitergabe des reinen Abschneidefehlers | −0,263 | 0,77 | 9/9 | 0,016 | Unterschied (BPreCap kleiner) |
| S3 | κ(BPreCap) − κ(ctanh) | +0,371 | – | 0/9 | 0,016 | Unterschied (BPreCap größer) |
| S4 | ν(BPreCap) − ν(ctanh) | +0,211 | – | 0/9 | 0,016 | Unterschied (BPreCap größer) |

S1 und S2 zusammen geben das klarste Bild:
- Ein Abschneidefehler gegebener Größe erreicht die Logits bei BPreCap etwa 23 % schwächer.
- Ein reiner Rundungsfehler gegebener Größe erreicht sie dagegen etwa 10 % stärker.
- Der Vorteil aus V2f (ḡ^Q 16 % schwächer) ist also die Summe aus zwei gegenläufigen Teilen, und der Abschneide-Teil überwiegt.

## Beschreibend

Mediane über 9 Seeds:

| Form, Kalibrierung | D^Q | D^C | D^R | r̄^C | r̄^R | ḡ^C | ḡ^R | κ | ν | sup^Q | Energie-Anteil Abschneiden |
|---|---|---|---|---|---|---|---|---|---|---|---|
| BPreCap, mse4 | 0,274 | 0,120 | 0,256 | 0,022 | 0,079 | 0,495 | 0,173 | 1,16 | 1,08 | 1,07 | 15,6 % |
| ctanh, mse4 | 0,303 | 0,151 | 0,268 | 0,024 | 0,071 | 0,648 | 0,157 | 0,77 | 0,89 | 1,10 | 25,6 % |
| BPreCap, max | 0,349 | 0,007 | 0,348 | 0,001 | 0,093 | 0,332 | 0,202 | 1,78 | 1,22 | 1,08 | 0,04 % |
| ctanh, max | 0,347 | 0,017 | 0,344 | 0,002 | 0,083 | 0,418 | 0,169 | 1,36 | 1,10 | 1,09 | 0,25 % |

Median log q (BPreCap / ctanh) und wie oft BPreCap kleiner ist:
- mse4: D^Q −0,17 (7/9), D^C −0,21 (9/9), D^R −0,08 (7/9), ḡ^Q −0,17 (9/9), ḡ^C −0,26 (9/9), ḡ^R +0,10 (1/9), r̄^C −0,06, r̄^R +0,10
- max: D^Q +0,007 (4/9), D^C −0,97 (9/9), D^R 0,000 (4/9), ḡ^Q +0,16 (0/9), ḡ^C −0,28 (9/9), ḡ^R +0,17 (0/9)

Was das heißt:
- **Bei max** wird fast nichts abgeschnitten (Energie-Anteil 0,04 % bzw. 0,25 %). Der Abschneide-Vorteil kann dann nicht wirken. Übrig bleibt die stärkere Weitergabe des Rundungsfehlers, und D^Q ist gleich (4/9). Das erklärt die Umkehr von ḡ^Q bei max aus V2f.
- **Ganz ohne Abschneiden ist bei max keine innere Schicht** (n_ohne_clip = 0 in allen 36 Messungen). Der Median-Anteil abgeschnittener Werte liegt bei 0,1 % bzw. 0,3 %, weil Testbilder über das Kalibrier-Maximum hinausgehen. Die Vorab-Aussage „bei max wird in den meisten inneren Schichten nichts abgeschnitten“ trifft deshalb nicht zu.
- **Wo im Netz** (mse4, log q(g_i^Q) je Block, BPreCap kleiner von 9): 1–10 −0,11 (8), 11–20 −0,14 (9), 21–30 −0,28 (9), 31–40 −0,27 (8), 41–49 −0,03 (5). Der Vorteil entsteht in der Mitte des Netzes, nicht an den letzten Schichten. BPreCap schneidet in den ersten zehn Schichten mehr ab (log q(r^C) +0,22), ab Schicht 21 weniger (−0,20 bis −0,24).
- **Zerlegung:** Q(clip(h)) = Q(h) und e_C + e_R = e_Q gelten exakt. Die Energie-Summe (r^C)² + (r^R)² ist im Selbsttest gleich (r^Q)² (Quotient 1,0), weil sich die Träger von e_C und e_R nicht überlappen. Die Vorsichtsklausel in den Grenzen der Präregistrierung war also unnötig.

### Kreuzterme: κ und ν gemeinsam lesen

- **κ ist groß und positiv**: 1,16 (BPreCap) und 0,77 (ctanh) bei mse4. Die Logit-Abweichungen der einzelnen Schichten zeigen also im Mittel leicht in dieselbe Richtung. Die mittleren Kosinus sind klein (Nachbarn 0,05 bzw. 0,04, innen 0,016 bzw. 0,003), summieren sich über 49 Schichten aber auf.
- **ν liegt bei etwa 1**: 1,08 bzw. 0,89. Die tatsächliche Abweichung, wenn alle Schichten zugleich gerundet werden, ist also **nicht** die Summe der Einzelabweichungen. Das lineare Bild bricht in diesen Netzen zusammen.
- Zusammen: Die Summe der Einzelabweichungen würde bei BPreCap eine größere Gesamtstörung vorhersagen (κ größer). Tatsächlich ist die Gesamtstörung bei BPreCap aber kleiner (D^Q 7/9), und sup^Q ist bei beiden fast gleich (1,07 gegen 1,10, wie in V2f).
- **S3 ist deshalb kein Befund über die Gesamtstörung.** Er beschreibt nur, wie die Einzelabweichungen zueinander stehen. Wer aus κ auf Fehlerakkumulation schließt, liegt hier falsch.
- Vergleich mit Tarawneh 2026 (Median c = −0,05 in Plain-CNNs): Die Größen sind verschieden definiert. Tarawnehs c vergleicht je Schicht den neuen mit dem geerbten Fehler, κ vergleicht Logit-Beiträge über alle Schichtpaare. Ein direkter Zahlvergleich ist nicht möglich. Die Vorab-Erwartung „κ zwischen −0,2 und +0,2“ beruhte auf dieser Gleichsetzung und war falsch.

## Integrität

- 36/36 Zeilen ok, run_ids eindeutig (je run_id und Kalibrierung), 9 gültige Paare.
- Abgleich mit V2f: d^Q maximale relative Abweichung 0,0, D^Q 1,9·10⁻⁹ (Schwelle 10⁻³).
- Selbsttest vor dem Freeze komplett (`meta/selfcheck.json`): Zerlegung exakt (Summe 0,0, Q(clip) = Q 0,0, Grenzen auf dem Gitter), d^Q und D^Q gleich `v2e_run.measure` (bis 3·10⁻⁹), κ direkt nachgerechnet, alles endlich.
- Auswertungstest 11/11 ok, Rauchtest 4/4 ok.
- Freeze-Hashes von V2e, V2f, V2a2, V2c, V2d sowie Gewichte, V2f-Messungen und CIFAR mitgeprüft.
- Lauf: 22.11. 03:03 bis 05:54, etwa 7,3 min je BPreCap-Messung und 2,2 min je ctanh-Messung.
- Vorab-Wahrscheinlichkeiten gegen Ergebnis:
  - alle 36 ok und Abgleich < 10⁻³ 0,85 → ja
  - K1 bestätigt 0,6 → ja
  - K2 bestätigt 0,5 → ja
  - S1 Unterschied 0,35 → ja (BPreCap größer)
  - S3 Unterschied 0,2 → ja
  - κ bei beiden zwischen −0,2 und +0,2 0,7 → **nein** (0,77 und 1,16)
  - bei max in den meisten inneren Schichten nichts abgeschnitten 0,6 → **nein**

## Einordnung

- V2f hatte gezeigt, *dass* ein Rundungsfehler bei mse4 die Logits bei BPreCap schwächer erreicht. V2k zeigt, *welcher Teil*: der Abschneidefehler. Der reine Rundungsfehler wird bei BPreCap sogar stärker weitergegeben, wie es die größere lineare Steigung erwarten lässt.
- Damit ist das Bild aus V2e, V2f und V2k jetzt stimmig:
  - BPreCap rundet etwas gröber (V2f H3, hier r̄^R q ≈ 1,11).
  - Sie gibt reine Rundungsfehler etwas stärker weiter (S1).
  - Sie gibt Abschneidefehler deutlich schwächer weiter (S2) und verliert durch das Abschneiden insgesamt weniger (K1).
  - Bei einer abschneidenden Kalibrierung überwiegt der letzte Punkt. Bei max entfällt er, und beide Formen sind gleich.
- Die Kohärenz-Deutung aus V2e bleibt zurückgezogen. Die Einzelabweichungen sind bei BPreCap zwar gleichgerichteter (κ), das wirkt sich auf die Gesamtstörung aber nicht aus (ν ≈ 1, sup gleich).
- Warum die Form das bewirkt, ist weiter nicht gezeigt. Plausibel, aber ungeprüft: BPreCap sättigt spät und steil, die Hülle ist nahe am Rand fast flach. Ein abgeschnittener Wert landet dann in einem Bereich, den die nächste Schicht kaum noch unterscheidet. ctanh krümmt früher und gleichmäßiger.
- Prior art: Zerlegungen des Quantisierungsfehlers in Clipping- und Rundungsanteil sind für die Wahl der Clipping-Grenze üblich, etwa bei ACIQ ([Banner et al. 2019](https://arxiv.org/abs/1810.05723), aus dem Gedächtnis, nicht an der Quelle geprüft) und den MSE-Kalibrierungen. Einen Formvergleich der Weitergabe beider Anteile über die Tiefe habe ich in der Recherche vom 21.11. nicht gefunden. Neuheit behaupte ich nicht. Die Recherche hat dazu nicht gezielt gesucht.

## Grenzen

- Messstudie, kein Eingriff. Die Zerlegung zeigt, welcher Teil des Fehlers wirkt, nicht warum.
- Die Netze und die Hypothese stammen aus V2f. Neu sind nur die Zerlegung und die Kreuzterme.
- ν ≈ 1 zeigt, dass die Einzelbeiträge die Gesamtstörung schlecht beschreiben. Das gilt auch für ḡ und d_i aus V2e und V2f. Gesicherte Aussagen über die Gesamtstörung stützen sich deshalb auf D^Q, D^C und D^R, also auf alle Schichten zugleich.
- 9 Seeds, ein Netz (CNN, L = 50, ohne Normierung), ein Datensatz, 1000 Testbilder, Aktivierungen je Tensor, Gewichte fp32, M1 Pro.

## Nächster Schritt

- v0.4-Entwurf 6.4: Die Abschneide-Deutung ist jetzt gemessen. Ich formuliere sie als Befund mit V2k als Beleg, das „warum“ bleibt Vermutung.
- Möglicher Eingriff (Aaron entscheidet): Hülle oder Sättigung gezielt verändern und prüfen, ob ḡ^C folgt. Das würde das „warum“ testen.
