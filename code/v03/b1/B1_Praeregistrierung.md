# B1 – Präregistrierung: Tiefenkalibrierung und schmaler Bereich in einer Form (privat)

Stand: 29.10.2026, nachts, vor dem Start. PreCap/SPreCap, die Tiefenfamilie und Lagma sind Aarons Ideen. Ich habe mit Claude die Vorrechnung gemacht, die neue Form daraus abgeleitet und die Studie entworfen. Aaron hat Studien für diese Nacht freigegeben („arbeite allein, auch zwischen Studien, nichts veröffentlichen“). Den Arbeitsnamen kL2B legt Aaron später endgültig fest.

## Ausgangslage

- **Was v0.2 sagt:** „Kein Mitglied der Familie schafft beides.“ kL2 trainiert bei 100–200 Schichten gut, hat aber einen weiten Bereich ([−3,0; 8,2] bei d100, [−4,2; 12,8] bei d200) und verliert bei 4 Bit 18–31 pp. cs ist schmal ([−1,8; 3,4]) und robust bei 4 Bit, trainiert bei d200 aber schlecht (χ1 = 1,07).
- **Was Lagma bisher erreicht:** Lagma-5 hat diese Lücke verkleinert, aber nicht geschlossen. Mit LD100 liegt der Fehler bei MNIST d100 in w8a4 bei 24 % statt 29 %, in fp32 bei 5 %. Bei d200 kostet jede Bremse gegenüber kL2 in fp32 etwa 1 pp.
- **Vorrechnung** (`math/p4_bounded_depth.py`, Ausgabe `p4_bounded_depth.out`, Kopie `p4_bounded_depth.json`):
  - **Kein mathematisches Gesetz:** Nach der Poincaré-Ungleichung gilt E[f′²] ≥ Var f für N(0, 1), Gleichheit nur für lineare f. χ1 = 1 + k/L verlangt also einen fast linearen Kern. Es verlangt aber **keinen weiten Bereich**. Eine fast lineare Funktion mit Grenze bei 3σ hat χ1 ≈ 1,0006.
  - **Woran es in v0.2 lag:** Die Kopplung in der PreCap-Parametrisierung. Der Tiefenregler b linearisiert den Kern, und λ schiebt dabei die Grenze nach außen.
  - **Neue Form kL2B:** f(x) = M · φ_ρ(PreCap(x; λ, α, b) / M), mit M = 3 und ρ = 0,8, wobei φ_ρ die Lagma-Bremse ist.
    - (λ, α, b) werden **mit** der Grenze exakt gelöst: E[f] = 0, E[f²] = 1, E[f′²] = 1 + 2/L.
    - Ergebnis bei d100: λ = 2,98437, α = 1,40673, b = 0,28967, Bereich [−2,94; 3,00].
    - Ergebnis bei d200: λ = 4,90973, α = 1,24891, b = 0,18464, Bereich [−3,00; 3,00].
    - Am Fixpunkt greift die Grenze bei 1,6 % der Werte (|Kern| > 0,8 · M).
    - Die Varianzabbildung zieht stärker an als bei kL2 (Steigung 0,92 bzw. 0,94 statt 0,97 bzw. 0,99).
  - **Gegenprobe:** Die Gauss-Hermite-Prüfung im Selbsttest und das feine Gitter im Mechanik-Test bestätigen Mittel 0, Varianz 1 und χ1 auf 1e-5.
- **Ehrliche Vorwarnung aus der Vorrechnung:**
  - **Exakte Lösung bringt an der Init wenig:** kL2-Konstanten mit derselben Grenze, ohne Neulösung, wären durch den üblichen Init-Gain g = 1/√E[f²] an der Init fast gleich zu kL2B. Effektiv wäre χ1 ≈ 1,023 statt 1,020 (d100) und 1,012 statt 1,010 (d200), das Mittel −0,003. Diese Kontrolle wird deshalb **nicht** gerechnet; ihr Unterschied wäre an der Init klein (Selbsttest-Werte stehen in der Vorrechnung).
  - **Woher die fp32-Kosten der festen Kappe vermutlich kommen:** Die fp32-Kosten von L1 bei d200 entstanden daher vermutlich im Training (Aktivierungen wachsen über die Grenze), nicht an der Init. kL2B hat eine engere Grenze als L1 (3 statt etwa 5,3) und könnte deshalb in fp32 eher mehr kosten.
- **Wichtigster Vorläufer, beim Prüfen gefunden:**
  - **Kritische hard-tanh-Familie:** Pennington, Schoenholz und Ganguli 2017 (https://arxiv.org/abs/1711.04735) beschreiben eine Ein-Parameter-Familie kritischer hard-tanh-Netze (χ = 1, Steigung erf(1/√(2q*))). Ein linearer Kern mit Grenze und χ1 ≈ 1 ist also bekannt.
  - **Gegenprobe LB:** Deshalb rechne ich als Gegenprobe **LB** mit, also einen linearen Kern mit derselben weichen Grenze: f(x) = M · φ_ρ(β x / M). (β, M) sind exakt gelöst für Varianz 1 und χ1 = 1 + 2/L (`math/p5_linear_bound.py`): bei d100 β = 1,03532 und M = 2,2165, bei d200 β = 1,01871 und M = 2,4543. Am Fixpunkt sind 8,7 % bzw. 5,4 % der Werte in der Sättigung.
  - **Was LB beantwortet:** ob der PreCap-Kern überhaupt etwas beiträgt oder ob „linear plus Grenze“ (bekannt) dasselbe leistet.

## Frage

Ist kL2B bei 4 Bit besser als die bisher beste Lagma-Form LD100 und dabei in fp32 so gut wie kL2? Das hieße: Tiefe und 4 Bit in einer Funktion.

## Varianten (eine Änderung der Frage: die Grenze als Teil der Aktivierung mit exakt gelösten Konstanten)

| Variante | Aktivierung | Rolle |
|---|---|---|
| **kL2B** | M · φ_ρ(PreCap(x; kL2B-Konstanten) / M), M = 3 | neue Form |
| kL2 | PreCap kL2 (aus A) | Tiefe ohne Grenze |
| LB | M · φ_ρ(β x / M), linearer Kern, (β, M) exakt gelöst | Gegenprobe zum Vorläufer (kritische hard-tanh-Familie): Braucht es den PreCap-Kern? |
| kL2_LD100 | kL2 + Lagma LD100 (aus Lagma-5) | bisher beste Bremse |
| spreccap_cs | SPreCap-cs | schmal, nicht tiefenkalibriert |

## Konstanten (wie A und Lagma-5)

- MLP ohne Normierung und ohne Sprungverbindungen, Breite 256, Batch 128, 1720 Schritte, Adam.
- MNIST und FashionMNIST, Tiefe 100 und 200, also 4 Zellen.
- **Lernraten** wie in A (`lr_from_a.csv`, `lr_from_p3a3.csv`). kL2B und LB nehmen die Lernrate von kL2 und sind **nicht eigens getunt**.
- **Quantisierung:**
  - Gewichte int8 pro Ausgangskanal.
  - Aktivierungen int4 pro Tensor, symmetrisch (w8a4_best: beste aus max/p999/p9999/mse4 nach dem Fehler auf 1000 Kalibrierbildern).
  - **Neu:** zusätzlich asymmetrisch mit Nullpunkt (w8a4a_best, Code aus Q3, gleiche Wahlregel).
- Lagma LD100 mit allen Werten wie Lagma-5 (q = 0,999, κ = 1,5, ρ = 0,8, η = 0,01, Anlauf 25 %, Deckel 1,0 · c0).

## Seeds und Umfang

- Neue Seeds 42060–42071: 12 je Zelle und Variante, gepaart (gleiche Init-Zufallszahlen und gleiche Batch-Reihenfolge).
- 4 Zellen × 5 Varianten × 12 = 240 Läufe. Dazu kommen 2 Kontrollläufe: kL2 und kL2_LD100, MNIST d100, Seed 42045, müssen bitgleich zu Lagma-5 sein.
- 242 Läufe, etwa 3 h auf 2 Workern.
- Ein Ausfall zählt als 90 % Fehler.

## Primärtest

kL2B gegen kL2_LD100 bei **w8a4_best**, gepaart je Seed. 4 Zellen, Wilcoxon zweiseitig exakt, Holm über 4, α = 0,05.

Das Gesamturteil folgt den Regeln aus A und Lagma-1 bis 5:

| Ergebnis | Urteil |
|---|---|
| mindestens 2 Zellen besser, keine schlechter | „besser“ |
| mindestens 2 Zellen schlechter, keine besser | „schlechter“ |
| keine signifikante Zelle | „kein Effekt“ |
| sonst | „gemischt“ |

**H1 (Bedingung für Erfolg):** In jeder Zelle ist die mediane fp32-Differenz kL2B − kL2 höchstens **+0,5 pp**. Die Schwelle liegt bei etwa der Hälfte der bekannten Bremsen-Kosten bei d200 (L1 +1,9, LD100 +1,3 pp).

**Erfolg:** Der Primärtest ergibt „besser als kL2_LD100“, und H1 hält.

## Sekundär (Holm je Familie über die 4 Zellen)

| Familie | Vergleich | Frage |
|---|---|---|
| S1 | kL2B − kL2, fp32 | Test zu H1: Kostet die Grenze Tiefe? |
| S2 | kL2B − kL2, w8a4_best | Gewinn gegenüber der Form ohne Grenze |
| S3 | kL2B − LB, fp32 | Trägt der PreCap-Kern in fp32 etwas bei? |
| S4 | kL2B − LB, w8a4_best | Was kostet er bei 4 Bit gegenüber der schmaleren LB? |
| S5 | kL2B − kL2_LD100, w8a4a_best | Hält der Primärbefund bei asymmetrischer Quantisierung? |
| S6 | kL2B − spreccap_cs, fp32 | Ist kL2B tiefentauglicher als cs (vor allem d200)? |
| S7 | kL2B − spreccap_cs, w8a4_best | Wie nah kommt kL2B an cs bei 4 Bit? |

**Nur beschreibend:**
- Mediane je Variante: fp32, w8a8, w8a4 max, w8a4_best, w8a4a_best.
- Bereich (max, p99,99), Gradientengefälle an der Init, Trainingsverlust.
- Sättigungsanteil der Grenze am Ende (kL2B, LB), c/c0 und Eingriffsrate (LD100).
- Einbrüche: fp32 über dem Doppelten des kL2-Medians der Zelle.
- ms je Schritt.

## Deutungsregeln (vorab)

| Ergebnis | Deutung |
|---|---|
| Erfolg | Tiefe und 4 Bit gehen in einer Form. Der Satz „kein Mitglied schafft beides“ aus v0.2 gilt für die alte Parametrisierung, nicht grundsätzlich. Kandidat für v0.3, nach Literaturprüfung und Wiederholung auf neuen Seeds |
| Primär besser, H1 verletzt | Die feste Grenze kauft 4 Bit mit fp32-Kosten bei der Tiefe, wie die Bremsen vorher. Der Zielkonflikt liegt dann im Training (Wachstum über die Grenze), nicht in der Init. Nächster Schritt: Grenze mitwachsen lassen (M als Vielfaches eines Quantils) oder M = 4 |
| Primär „kein Effekt“ oder „schlechter“ | Die engere Grenze bringt gegenüber LD100 nichts bei 4 Bit. Die Regelung von Lagma bleibt die bessere Lösung |
| S3 „kein Effekt“ oder „schlechter als LB“ | Der PreCap-Kern bringt in fp32 nichts gegenüber „linear plus Grenze“. Dann ist der Beitrag nicht die Form, sondern die bekannte kritische Begrenzung. LB wäre die ehrlichere Empfehlung |
| S3 „besser als LB“ | Der nichtlineare PreCap-Kern trägt bei der Tiefe etwas bei, das reine Begrenzung nicht leistet |
| S4 „schlechter als LB“ | LB ist bei 4 Bit besser, wie wegen der engeren Grenze (2,2–2,5 statt 3) zu erwarten. Abzuwägen mit S3 |
| S6 bei d200 „besser“, S7 nur wenig schlechter | kL2B verbindet die Stärken von cs und kL2 zum großen Teil |
| Kontrolle nicht bitgleich | Abweichung wird gesucht und berichtet. Die Tests gelten trotzdem (Paarung bleibt) |

## Vorab-Erwartung

| Größe | Erwartung |
|---|---|
| Primär „besser als LD100“ | 0,7 |
| H1 hält | 0,3 |
| Erfolg | 0,25 |
| S2 „besser als kL2“ (4 Bit) | 0,95 |
| S1 bei Fashion d200 einzeln „schlechter“ | 0,55 |
| S3 „besser als LB“ (fp32) | 0,45 |
| S4 „schlechter als LB“ (4 Bit) | 0,5 |
| S6 „besser als cs“ (fp32) | 0,8 |
| S7 „schlechter als cs“ (4 Bit) | 0,5 |
| Kontrolle bitgleich | 0,95 |

## Neuheit und Literatur (Kurzprüfung 29.10. nachts)

- **Bekannt:**
  - χ1 = 1 + ε mit ε < 1/L für selbstnormierende Funktionen (Chen et al. 2020, https://arxiv.org/abs/2001.00254), DKS und TAT.
  - Dynamische Isometrie mit beschränktem tanh im linearen Bereich (Pennington et al. 2017). Dort ist die Signalvarianz klein gegenüber der Sättigung, also der Bereich relativ zur Standardabweichung weit.
  - Clipping und Bremsen für Quantisierung: PACT, SCULPT (https://arxiv.org/abs/2609.01743). SaluNet 2026 (https://arxiv.org/html/2606.02927) ist eine lernbare beschränkte Aktivierung ohne Normierung, ohne 4-Bit-Fragestellung.
  - Kritische hard-tanh-Familie (Pennington et al. 2017, siehe oben, Gegenprobe LB). bSELU-BSTD („Large-Scale Self-Normalizing Neural Networks“, bi-scaled ELU) ist nicht beidseitig beschränkt, soweit aus der Beschreibung erkennbar.
- **Nicht gefunden** (2 Suchen, eine davon erweitert): eine Form, die Mittel 0, Varianz 1 und χ1 = 1 + k/L exakt mit einer Grenze bei etwa 3σ verbindet und auf 4 Bit gemessen wird.
- **Einordnung:** Die Konstruktion ist im Nachhinein naheliegend (Kern linearisieren, bei 3σ begrenzen). Ihr Wert wäre empirisch. Vor jeder Veröffentlichung braucht es eine systematische Suche, besonders zu „bounded“ und „hard-tanh“ bei dynamischer Isometrie und zu bSELU (JAI 2024, Konstanten nicht geprüft).

## Grenzen

- Nur MLPs, MNIST und FashionMNIST, 1720 Schritte, eine Breite.
- Lernrate für kL2B nicht getunt.
- Nur M = 3 und ρ = 0,8, keine Suche über M.
- Nur Post-Training-Quantisierung pro Tensor.
- 12 statt 15 Seeds (kleinster exakter p-Wert 0,0005, mit Holm über 4 also 0,002).

## Freeze

Eingefroren werden `b1_run.py`, `b1_analyze.py`, `test_b1_mechanik.py`, `test_b1_analyse.py`, diese Datei, `lr_from_a.csv`, `lr_from_p3a3.csv`, `p4_bounded_depth.json`, `p5_linear_bound.json` und `src/precap/*.py`.

Vorab gemachte Tests (29.10. nachts):
- **Selbsttest:** 242 Läufe. Gauss-Hermite für kL2B ergibt χ1 = 1,01999 bzw. 1,00999, Varianz 1,000. Für LB ergibt er χ1 1,0200 bzw. 1,0100 und Bereich ±2,2165 bzw. ±2,4543. (Vorab-Version mit kL2_F3: χ1 1,0066 bzw. 0,9985 ohne Gain, Gain 1,008 bzw. 1,007.)
- **Mechanik** (`test_b1_mechanik.py`): MECHANIK OK.
  - Fixpunkt von kL2B exakt (Mittel < 1e-10, Varianz 1,000000, χ1 1,019992 bzw. 1,009995).
  - Grenze hält, und die Aktivierung entspricht exakt brake(PreCap).
  - **kL2 und kL2_LD100 laufen bitgleich zu Lagma-5.**
  - Lernraten wie A bzw. Lagma-5, neue Seeds.
  - Die asymmetrische Quantisierung ist bitgleich zu Q3.
  - LB hat einen exakten Fixpunkt (Varianz 1,000000, χ1 1,020000 bzw. 1,010000), und die Grenze hält.
  - Alle Varianten laufen, der Sättigungsanteil wird gemessen.
- **Auswertung an künstlichen Daten** (`test_b1_analyse.py`): AUSWERTUNG OK.
  - „Erfolg“ wird erkannt.
  - „teuer bei d200“ ergibt Primär besser und H1 verletzt.
  - „Null“ ergibt in 6 von 6 Fällen „kein Effekt“.
  - Eine falsche Kontrolle wird erkannt, ein Ausfall wird gezählt.
- **Rauchtest:** im Startskript nach dem Freeze, MNIST d100, 6 Läufe mit 40 Schritten.
