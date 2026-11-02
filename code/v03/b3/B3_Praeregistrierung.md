# B3 – Präregistrierung: Trägt der PreCap-Kern auf schwereren Daten etwas bei? (CIFAR-10, privat)

Stand: 30.10.2026, vormittags, vor dem Start. PreCap, die Tiefenfamilie und Lagma sind Aarons Ideen. Aaron will die neue Form als Variante von PreCap führen, „BPreCap“ ist dafür der Arbeitsname (kL2B im Code). Das Design habe ich mit Claude entworfen, nach Aarons Auftrag „ob die neue Variante eine Chance hat, ob SPreCap noch eine Chance hat“. Aaron hat Studien für heute und den Download von CIFAR-10 freigegeben (nichts veröffentlichen, nichts pushen).

## Ausgangslage

- **B1:** BPreCap (PreCap-Kern mit Lagma-Hülle, kritisch eingestellt) und LB (linearer Kern, dieselbe Hülle) sind auf MNIST/Fashion in fp32 gleich. LB ist bei 4 Bit besser. Dort hat der PreCap-Kern nichts gebracht.
- **Rechnung p8** (`math/Mathe_Gesamt_PreCap_Lagma_Tiefe.md`, Abschnitt 12):
  - Bei gleicher Tiefenregel enthält BPreCap im ganzen Netz mehr Nichtlinearität als LB (d200: 0,553 gegen 0,484), kL2 noch mehr (0,672).
  - Der Grund: Gerade Anteile (Asymmetrie) kosten am wenigsten Tiefen-Budget.
  - Auf einfachen Daten sieht man das nicht, denn ein linearer Klassifikator schafft MNIST schon zu etwa 92 %.
- **Offen:** Zählt die zusätzliche Nichtlinearität auf schwereren Daten? CIFAR-10 ist für MLPs deutlich nichtlinearer: Ein linearer Klassifikator kommt auf etwa 40 % Genauigkeit, gute MLPs auf über 55 %.
- **TAT** (Zhang et al. 2022) war in v0.2 die stärkste Referenz für tiefe Netze ohne Normierung. Bei 4 Bit lag es nahe an LB. Es gehört deshalb als Referenz dazu.

## Frage

Ist BPreCap auf CIFAR-10 bei Tiefe 100 und 200 in fp32 besser als LB, also trägt der PreCap-Kern bei schwereren Daten etwas bei?

## Eine Änderung gegenüber B1: der Datensatz

| Variante | Form | Rezept |
|---|---|---|
| kL2B = BPreCap | 3 · φ_ρ(f_{λ,α,b}(x)/3), Konstanten wie B1 | Gauß-Init mit Gain, wie B1 |
| LB | M · φ_ρ(βx/M), Konstanten wie B1 | wie B1 |
| kL2 | PreCap-Tiefenfamilie ohne Grenze, wie A/B1 | wie B1 |
| TAT | Leaky-ReLU nach dks, η = 0,9, d100/d200 wie A2 | SUO-Init und Normierung je Beispiel, wie A2/Q3 |

**Daten:** CIFAR-10, Binärversion (`data/cifar10/cifar-10-batches-bin`, von https://www.cs.toronto.edu/~kriz/cifar.html). Die Bilder werden als 3072-Vektor eingelesen und global standardisiert wie MNIST. Die letzten 5000 Trainingsbilder sind die Val-Teilung, also 45 000 Training, 5000 Val und 10 000 Test.

**Konstanten wie B1/B2:** MLP mit Breite 256 ohne Normierung und Residuen, Adam, Batch 128, **1720 Schritte** (etwa 4,9 Epochen), Kalibrierbilder (die ersten 1000 Trainingsbilder), Quantisierung (w8a4_best, w8a4a_best, w8a3_best, w8a3a_best wie B2).

**Lernrate:** Auf CIFAR gibt es keine getunten Werte, deshalb wird **jede Variante je Tiefe neu getunt**.
- Raster 1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4.
- Tuning-Seeds 42000–42003, gleiches Budget.
- Gewählt wird die höchste mediane Val-Genauigkeit am letzten Schritt. Ein Ausfall zählt als 0,1, bei Gleichstand gewinnt die kleinere LR (Regel wie 3a-3/A3).
- Liegt die gewählte LR am Rand des Rasters, wird das berichtet.

## Läufe

- **Tuning:** 4 Varianten × 2 Tiefen × 6 LR × 4 Seeds = 192 Läufe.
- **Bestätigung:** neue Seeds 42084–42095 (12 je Zelle), 4 × 2 × 12 = 96 Läufe.
- **Kontrolle:** kL2B MNIST d100 Seed 42060 (LR von kL2 aus A) muss bitgleich zu B1 sein (fp32, w8a4_best, w8a4a_best, Trainingsverlust). Das prüft, dass der Code-Pfad unverändert ist.
- Zusammen 289 Läufe, etwa 4 h mit 2 Workern. Ein Ausfall zählt als 90 % Testfehler.

## Primärtest

**kL2B − LB bei fp32** (Testfehler in pp), je Tiefe Wilcoxon zweiseitig exakt, Holm über die 2 Zellen. Das Gesamturteil folgt derselben Regel wie in B1/B2. Mit 2 Zellen heißt „besser“: beide Zellen besser.

**Erfolg (für den PreCap-Kern):** Primär „besser als LB“.

## Sekundär (jede Familie Holm über 2 Zellen)

| Nr. | Vergleich | Frage |
|---|---|---|
| S1 | kL2B − LB, w8a4_best | Kostet der PreCap-Kern auf CIFAR bei 4 Bit? |
| S2 | kL2B − kL2, fp32 | Kostet die Grenze auf schwereren Daten fp32? |
| S3 | kL2B − kL2, w8a4_best | Bleibt der 4-Bit-Gewinn der Grenze auf CIFAR? |
| S4 | kL2B − TAT, fp32 | Abstand zur stärksten Referenz |
| S5 | kL2B − TAT, w8a4a_best | Ist BPreCap bei fairer 4-Bit-Quantisierung besser als TAT? |
| S6 | LB − TAT, w8a4a_best | dito für LB |
| S7 | kL2B − LB, w8a3_best | Unterschied bei 3 Bit |
| S8 | kL2 − LB, fp32 | Zählt mehr Nichtlinearität bei der Init (p8)? |

**Beschreibend:**
- gewählte Lernraten (Randlage),
- Mediane aller Metriken, Bereich, Sättigung, Gradientengefälle,
- **Vorhersage p8:** Gilt in fp32 die Ordnung kL2 < kL2B < LB je Tiefe?

## Deutungsregeln (vorab)

| Ergebnis | Deutung |
|---|---|
| Erfolg | Der PreCap-Kern trägt auf schwereren Daten etwas bei. Die zusätzliche Nichtlinearität aus p8 zählt. BPreCap ist dann mehr als „kritisch + begrenzt“, Kandidat für v0.3 (nach Wiederholung) |
| Primär „kein Effekt“ | Auch auf CIFAR bringt der PreCap-Kern nichts Messbares. Der Gewinn bleibt die kritische Begrenzung, LB ist die ehrlichere Empfehlung |
| Primär „schlechter als LB“ | Der PreCap-Kern schadet auf CIFAR. LB ist klar vorzuziehen |
| S3 „besser“ | Die Grenze hilft auch auf CIFAR bei 4 Bit |
| S5/S6 „besser“ | Bei fairer 4-Bit-Quantisierung schlägt die begrenzte kritische Form TAT, ein Hinweis auf einen eigenen Beitrag gegenüber dem Stand der Technik |
| S8 „besser“ und p8-Ordnung gilt | Mehr Nichtlinearität bei der Init nützt auf CIFAR. Die Budget-Rechnung hat Vorhersagekraft |
| LR am Rasterrand | Wird als Einschränkung berichtet. Bei Randlage in mehreren Varianten folgt ein Nachtuning in einer eigenen Studie |
| Kontrolle nicht bitgleich | Abweichung wird gesucht und berichtet, die Tests gelten (Paarung bleibt) |

## Vorab-Erwartung

| Größe | Erwartung |
|---|---|
| Primär „besser als LB“ (Erfolg) | 0,3 |
| Primär „schlechter als LB“ | 0,15 |
| Primär „kein Effekt“ oder „gemischt“ | 0,55 |
| S2 „schlechter als kL2“ | 0,6 |
| S3 „besser als kL2“ | 0,9 |
| S4 „schlechter als TAT“ | 0,75 |
| S5 „besser als TAT“ | 0,35 |
| S6 „besser als TAT“ | 0,4 |
| S8 „besser als LB“ | 0,5 |
| p8-Ordnung gilt in beiden Tiefen | 0,3 |
| Kontrolle bitgleich | 0,95 |

## Neuheit und Literatur (ehrlich)

Siehe `docs/literatur/Recherche_2026-10-30_BPreCap.md`.
- **Bekannt:**
  - kritische Begrenzung (Pennington 2017)
  - Tiefengrenze bei quantisierten Aktivierungen (Blumenfeld 2019)
  - TAT/DKS
  - Nichtlinearität ist in trainierten Netzen oft auf wenige Schichten konzentriert (Ali-Mehmeti-Göpel et al. 2023)
- **Nicht gefunden:** der Vergleich „asymmetrischer, tiefenkalibrierter Kern gegen linearen Kern unter derselben kritischen Grenze“ auf CIFAR, mit Quantisierung nach dem Training.

## Grenzen

- MLP ohne Faltungen, also ein schwaches Modell für CIFAR (erwartet etwa 40–55 % Genauigkeit).
- 1720 Schritte, etwa 5 Epochen: eher untertrainiert. Die LR ist für dieses Budget getunt.
- TAT läuft im eigenen Rezept (orthogonal, Normierung je Beispiel), wie in v0.2. Die anderen Formen laufen mit Gauß-Init, ein Unterschied im Rezept bleibt also.
- Zwei Tiefen, 12 Seeds, ein Datensatz.

## Freeze

Eingefroren werden `b3_run.py`, `b3_analyze.py`, `test_b3_mechanik.py`, `test_b3_analyse.py`, diese Datei, `lr_from_a.csv`, `lr_from_p3a3.csv`, `src/precap/*.py` und die CIFAR-Dateien (`data/cifar10/cifar-10-batches-bin/*.bin`).

Vorab gemachte Tests (30.10., 11:20):

- **Daten:** CIFAR-10-Binärversion, Archiv-MD5 `c32a1d4ab5d03f1284b67883e8d87530` (bekannter Wert). Entpackt nach `data/cifar10/`, SHA-256 der .bin-Dateien in `data/cifar10/SHA256.txt`.
- **Selbsttest:** n_confirm 97, n_tune 192. TAT mit E[f²] = 1,0 bei d100/d200, MLX gleich numpy.
- **Mechanik** (`test_b3_mechanik.py`): MECHANIK OK.
  - kL2B, LB und kL2 gleich B2. TAT (Konstanten, Aktivierung, SUO-Init, Normierung je Beispiel) gleich A2/Q3.
  - CIFAR: Formen 45 000/5000/10 000 × 3072, Testklassen je 1000, standardisiert (Mittel 3e-8, Std 1,0000), erster Datensatz roh gleich gelesen.
  - TAT-Init orthogonal. Netz mit 3072 Eingängen, mit 784 Eingängen gleich B1.
  - Kontrollpfad kL2B (synthetisch) bitgleich zu B1. Alle vier Formen laufen auf CIFAR (60 Schritte).
- **Auswertung an künstlichen Daten** (`test_b3_analyse.py`): AUSWERTUNG OK.
  - „Erfolg“ inklusive p8-Ordnung und LR-Tabelle wird erkannt.
  - „Null“ ergibt 6 von 6 „kein Effekt“.
  - Eine falsche Kontrolle wird erkannt, ein Ausfall gezählt.
  - Der Aufruf mit „.“ findet B1.
- **Rauchtest:** läuft im Startskript (8 Tuning- und 5 Bestätigungsläufe).
