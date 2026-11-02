# B7 – Präregistrierung: Hält der Vorteil von BPreCap über LB bei vierfachem Budget? (CIFAR-10, privat)

Stand: 31.10.2026, nachts, vor dem Start. PreCap, die Tiefenfamilie und Lagma sind Aarons Ideen. BPreCap (= kL2B) ist ein Arbeitsname. Das Design habe ich mit Claude entworfen, mit Aarons Freigabe für die Nacht („freie Hand zu testen und zu entwickeln, exakt begründet, nichts veröffentlichen oder pushen“).

## Ausgangslage

- **B3:** Auf CIFAR-10 ist BPreCap in fp32 besser als LB, bei d100 um −2,95 pp (12/0). Das Budget war kurz: 1720 Schritte, etwa 5 Epochen. Diese Grenze steht offen im Bericht.
- **Literatur:** Nach „Nonlinear Advantage“ (ICML 2023) bremst frühe Linearität vor allem das Training am Anfang. Es ist also möglich, dass LB bei längerem Training aufholt. Dann wäre der B3-Befund nur eine Aussage über frühes Training.
- Die Frage entscheidet, ob BPreCap mehr ist als ein schnellerer Start.

## Frage

Ist BPreCap auf CIFAR-10 (d100) auch nach vierfachem Budget (6880 Schritte, etwa 20 Epochen) in fp32 besser als LB? Und wie verändern sich der Vorsprung und der 4-Bit-Preis mit dem Training?

## Eine Änderung gegenüber B3: das Budget

- **STEPS = 6880 statt 1720.** Adam mit konstanter LR, ohne Plan und ohne frühes Stoppen, wie B3.
- **LR unverändert aus B3 d100** (kL2B 2e-5, LB 2e-5, TAT 5e-5, `lr_from_b3.csv`). Getunt wurde dort auf 1720 Schritte. Das ist eine bewusste Konstante, siehe Grenzen.
- **Zwischenmessungen** nach 1720 und 3440 Schritten: fp32, w8a4_best, w8a4a_best, w8a3_best, Bereich, Verlust. Es sind nur Vorwärtsrechnungen. Der Mechanik-Test zeigt, dass das Training dadurch bitgleich bleibt.
- **Nur d100** und die Formen kL2B, LB und TAT (TAT im eigenen Rezept wie B3).
- **Konstanten wie B3:** Formen, Init, Netz (Breite 256), Batch 128, Daten und Teilung, Kalibrierung, Quantisierung.
- **Seeds:** neu 42132–42143, gepaart.

## Läufe

- 3 Formen × 12 Seeds = 36 Läufe.
- **Kontrolle:** kL2B und LB, CIFAR-10 d100, Seed 42084 (ein B3-Seed). Die Zwischenmessung nach 1720 Schritten muss in allen 7 Kennzahlen bitgleich zu den Endwerten von B3 sein.
- Zusammen 38 Läufe, etwa 1 h mit 2 Workern. Start erst nach dem Ende von B6 und nur am Netzteil. Ein Ausfall zählt als 90 %, ebenso ein fehlender Zwischenwert.

## Primärtest

**kL2B − LB, fp32, nach 6880 Schritten**, Wilcoxon zweiseitig exakt, eine Zelle, α = 0,05.

**Erfolg:** p < 0,05 und Median < 0 („besser“).

## Sekundär (Holm über alle 8)

| Nr. | Vergleich | Frage |
|---|---|---|
| S1 | kL2B − LB, w8a4_best (6880) | 4 Bit symmetrisch |
| S2 | kL2B − LB, w8a4a_best (6880) | 4 Bit asymmetrisch |
| S3 | (kL2B − LB)₆₈₈₀ − (kL2B − LB)₁₇₂₀, fp32, je Seed | Holt LB auf? „schlechter“ (> 0) heißt, der Vorsprung schrumpft |
| S4 | kL2B − TAT, fp32 (6880) | Abstand zu TAT |
| S5 | kL2B − TAT, w8a4a_best (6880) | faire 4 Bit gegen TAT |
| S6 | LB − TAT, w8a4a_best (6880) | dito für LB |
| S7 | 4-Bit-Preis von kL2B (w8a4a − fp32), 6880 minus 1720, je Seed | Wächst der Preis mit dem Training? „schlechter“ (> 0) heißt ja |
| S8 | (kL2B − TAT)₆₈₈₀ − (kL2B − TAT)₁₇₂₀, fp32, je Seed | Ändert sich der Abstand zu TAT? |

**Beschreibend:**
- Mediane aller Metriken zu allen drei Zeitpunkten, Abstände der Formen zu jedem Zeitpunkt.
- Trainingsgewinn je Form (fp32 nach 6880 minus nach 1720) und 4-Bit-Preis je Form und Zeitpunkt.
- Bereich der Aktivierungen und Trainingsverlust (Überanpassung).

## Deutungsregeln (vorab)

| Ergebnis | Deutung |
|---|---|
| Erfolg und S3 „unklar“ | Der Vorteil von BPreCap hält bei längerem Training und ist mehr als ein schnellerer Start |
| Erfolg und S3 „schlechter“ | Der Vorteil hält, schrumpft aber. Wie stark, wird beziffert |
| Primär „unklar“ und S3 „schlechter“ | LB holt auf. Der B3-Befund war vor allem ein Effekt frühen Trainings (passt zu „Nonlinear Advantage“) |
| Primär „schlechter“ | Umkehr: Bei längerem Training ist LB besser |
| S7 „schlechter“ | Der 4-Bit-Preis wächst mit dem Training. Das muss in jede Aussage zu 4 Bit |
| S5 „besser“ | Bei fairen 4 Bit und längerem Training ist BPreCap besser als TAT |
| Kontrolle nicht bitgleich | Abweichung wird gesucht und berichtet, die Tests gelten |

## Vorab-Erwartung

| Größe | Erwartung |
|---|---|
| Primär „besser“ (Erfolg) | 0,6 |
| Primär „unklar“ | 0,3 |
| Primär „schlechter“ | 0,1 |
| S3 „schlechter“ (Vorsprung schrumpft) | 0,45 |
| S4 „schlechter“ (TAT vorn in fp32) | 0,85 |
| S5 „besser“ | 0,1 |
| S7 „schlechter“ (4-Bit-Preis wächst) | 0,5 |
| Kontrolle bitgleich | 0,95 |

## Grenzen

- **LR nicht für das längere Budget getunt.** Eine Form, deren beste LR bei längerem Training kleiner ist, wird benachteiligt.
- Kein LR-Plan und kein frühes Stoppen: Überanpassung geht in den Testfehler ein (für alle Formen gleich behandelt).
- Wie B3: MLP ohne Faltungen, ein Datensatz, nur d100, TAT im eigenen Rezept.
- 20 Epochen sind noch immer kein voll ausgereiztes Training.

## Freeze

Eingefroren werden `b7_run.py`, `b7_analyze.py`, `test_b7_mechanik.py`, `test_b7_analyse.py`, diese Datei, `lr_from_a.csv`, `lr_from_p3a3.csv`, `lr_from_b3.csv`, `src/precap/*.py` und die CIFAR-Dateien.

Vorab gemachte Tests (31.10., nachts):

- **Selbsttest:** n_confirm 38, 6880 Schritte, Zwischenmessungen 1720 und 3440.
- **Mechanik** (`test_b7_mechanik.py`): MECHANIK OK.
  - Formen, Init und Daten sind gleich B3. Budget ist 4 × B3, die Konstanten wie B3.
  - Die LR stammt aus B3 (2e-5, 2e-5, 5e-5). Es sind 38 Läufe, die Seeds neu, die Kontrolle auf dem B3-Seed.
  - **Die Zwischenmessung verändert das Training nicht.** Probe mit 172 Schritten und Zwischenmessung bei 86, für alle drei Formen:
    - Der Zwischenwert ist bitgleich zu einem B3-Lauf mit 86 Schritten.
    - Der Endwert ist bitgleich zu einem B3-Lauf mit 172 Schritten ohne Zwischenmessung.
    - Geprüft wurden alle 7 Kennzahlen, dazu w8a3a und w8a8.
- **Auswertung an künstlichen Daten** (`test_b7_analyse.py`): AUSWERTUNG OK.
  - „Hält“ wird als Erfolg mit S3 „unklar“ erkannt.
  - „Holt auf“ ergibt Primär „unklar“ und S3 „schlechter“.
  - Null mit 20 Wiederholungen: 3 Fehlalarme primär (an der Grenze, erwartet etwa 1), 0 sekundär.
  - Eine falsche Kontrolle wird erkannt, ein Ausfall gezählt (Zwischenwert als 90 %), der relative Pfad findet B3.
- **Rauchtest:** läuft im Startskript (5 Läufe mit 40 Schritten, Zwischenmessungen bei 10 und 20).
