# Q2 – Präregistrierung: SPreCap-cs gegen TAT bei 4-Bit-Aktivierungen (privat)

Stand: 27.10.2026, vor dem Start (Start automatisch nach dem Ende von A3). Entworfen von Claude nach A2, mit Aarons Freigabe für Folgeversuche in dieser Nacht.

## Frage

A2 zeigt: TAT ist in fp32 bei tiefen MLPs besser als kL2. Q1 und Q1b zeigen: Das beschränkte SPreCap-cs ist bei 4-Bit-Aktivierungen mit fairer Kalibrierung bei Tiefe 100 robuster als SELU. TAT ist unbeschränkt (in A2 lag der Bereich bis 16,8). Ist cs bei w8a4 auch besser als TAT? Davon hängt ab, ob die Beschränktheit eine echte Nische für PreCap ist.

## Eine einzige Änderung

Gegenüber Q1b kommt nur eine Variante dazu: TAT im eigenen Rezept, wie in A2:

- Leaky ReLU mit Parametern nach dks bei η = 0,9
- SUO-Init, Bias 0
- Normierung je Beispiel
- LR eigens getunt

Training, Quantisierung, Kalibrierung (max, p99,9, p99,99, MSE-Clip, Wahl je Lauf über die Kalibrierbilder) und Testauswertung bleiben unverändert aus Q1b. Code: `q2_run.py` ist aus `q1b_run.py` abgeleitet.

TAT-Parameter:

| Tiefe | neg | Quelle |
|---|---|---|
| d100 | 0,5704395323991776 | dks, wie A2 |
| d50 | 0,43052294850240236 | Bisektion der 50-fachen C-Map auf 0,9 |

Dieselbe Bisektion reproduziert die dks-Werte für d100 und d200 auf 1e-9. Der Selbsttest prüft C_f(0) = 0,9.

## Vergleichsdaten

SPreCap-cs und SELU stammen aus Q1b: gleiche Seeds 42010–42024, gleiche Daten, gleicher Code-Pfad (MLX, Kernel-Backend), gleiche LR wie in Q1b.

Reproduktion: cs MNIST d100, Seeds 42010/42011, wird neu gerechnet. Bestanden, wenn fp32 und w8a4_best je höchstens 0,10 pp von Q1b abweichen. Sonst wird das vermerkt, und die Tests gelten nur als vorläufig.

## Läufe

- Tuning TAT: LR-Raster 1e-5 … 5e-4 (6 Stufen), Seeds 42000–42003, Tiefe 50/100, MNIST/Fashion, 1720 Schritte, nur Val-Fehler. Zusammen 96 Läufe. Auswahlregel wie in A2.
- Bestätigung: TAT mit 15 Seeds × 4 Zellen = 60 Läufe, dazu 2 Reproduktionsläufe.
- Ausfall zählt als 90 %.
- Geschätzte Dauer etwa 45–60 min auf 2 Workern.

## Primärtest

cs gegen TAT im Testfehler bei **w8a4_best**, gepaart je Seed. 4 Zellen (Tiefe 50/100 × MNIST/Fashion), Wilcoxon zweiseitig, Holm über 4, α = 0,05. „besser“ heißt: cs hat den kleineren Fehler.

Gesamturteil wie in A/A2:

| Urteil | Bedingung |
|---|---|
| besser als TAT | mindestens 2 Zellen besser, keine schlechter |
| schlechter als TAT | mindestens 2 Zellen schlechter, keine besser |
| kein Effekt | keine Zelle signifikant |
| gemischt | alles andere |

## Sekundär (Holm je Familie, je 4 Tests)

- **S1:** cs gegen TAT in fp32
- **S2:** cs gegen TAT im Verlust durch Quantisierung (w8a4_best − fp32)
- **S3:** TAT gegen SELU bei w8a4_best
- **Beschreibend:** w8a8, w8a4 mit Max-Kalibrierung, Wertebereich, MSE-Clip, gewählte Kalibrierung

## Vorab-Erwartung

cs gegen TAT bei w8a4_best:

| Urteil | Erwartung |
|---|---|
| besser als TAT | 0,30 |
| kein Effekt | 0,25 |
| schlechter als TAT | 0,35 |
| gemischt | 0,10 |

S2 (cs verliert durch Quantisierung weniger als TAT): 0,6. S1 (TAT in fp32 besser): 0,7.

## Bedeutung

- **cs „besser“:** Die Beschränktheit ist eine belegte Nische. In v0.2 heißt es dann: tiefe MLPs mit wenigen Bits.
- **„kein Effekt“ oder „schlechter“:** Auch bei 4 Bit hat cs keinen Vorteil gegen TAT. Die Nische trägt in dieser Form nicht. Dann ehrlich berichten, und Lagma wird wichtiger.
- **„gemischt“:** Die Zellen werden einzeln berichtet.

## Grenzen

- Nur Post-Training-Quantisierung, simuliert, pro Tensor symmetrisch, wie Q1b.
- Adam statt K-FAC für TAT (wie A2).
- 1720 Schritte, Breite 256, MNIST/Fashion.

## Freeze

Eingefroren werden `q2_run.py`, `q2_analyze.py`, `lr_from_p3a3.csv`, `src/precap/*.py` und diese Datei. Die Auswertung ist an künstlichen Daten getestet: Ein eingebauter Effekt von −3 pp bei d100 wurde erkannt, bei d50 ergab sich kein Effekt. Ein Trockenlauf gegen die echten Q1b-Daten ergab 15 cs- und 15 SELU-Läufe je Zelle und eine Reproduktion von 0,00 pp.
