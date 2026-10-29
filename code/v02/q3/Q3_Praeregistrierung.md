# Q3 – Präregistrierung: asymmetrische 4-Bit-Quantisierung, cs gegen TAT (privat)

Stand: 28.10.2026, 02:15, vor dem Start. Start automatisch nach dem Ende von Lagma-1. Entworfen von Claude nach Q2, mit Aarons Freigabe für Folgeversuche in dieser Nacht.

## Frage

In Q2 ist SPreCap-cs bei w8a4_best in allen 4 Zellen klar besser als TAT (−1,4 bis −7,7 pp), obwohl TAT in fp32 besser ist. Q2 quantisiert die Aktivierungen **pro Tensor symmetrisch**. Leaky ReLU (TAT) und SELU sind asymmetrisch, sie verschenken damit einen Teil der 16 Stufen. Hält der Vorsprung von cs, wenn die Quantisierung asymmetrisch ist (affin, mit Nullpunkt, wie in der Praxis üblich)?

## Eine einzige Änderung

Zusätzlich zur symmetrischen Auswertung aus Q2 werden die Aktivierungen auch **asymmetrisch pro Tensor** quantisiert:

- Raster mit 2^b Stufen über [lo, hi], Skala (hi − lo)/(2^b − 1), ganzzahliger Nullpunkt; 0 ist exakt darstellbar.
- Kalibrierkandidaten auf denselben 1000 Kalibrierbildern: max = [min, max]; p999 = [Q 0,05 %, Q 99,95 %]; p9999 = [Q 0,005 %, Q 99,995 %]; mse4 = MSE-optimale Skalierung von [min, max] mit dem Faktor k/100, k = 1…100.
- **w8a4a_best:** Wahl je Lauf über den Fehler auf den Kalibrierbildern, Gleichstand in der Reihenfolge max, p999, p9999, mse4 (wie w8a4_best in Q1b/Q2).
- Gewichte unverändert: int8 pro Ausgangskanal symmetrisch.

Training ist identisch zu Q1b (cs, SELU) und Q2 (TAT): gleiche Seeds, gleiche LR, gleicher Code-Pfad. Code: `q3_run.py`, abgeleitet aus `q2_run.py`. Weil Gewichte nicht gespeichert wurden, werden alle Läufe neu trainiert.

## Varianten und Läufe

- SPreCap-cs, TAT (Rezept wie Q2), SELU. Tiefe 50/100, MNIST/Fashion, Seeds 42010–42024. Zusammen 180 Läufe, kein Tuning.
- LR: cs und SELU aus `lr_from_p3a3.csv` (wie Q1b), TAT aus `lr_tat_from_q2.csv` (Q2-Tuning).
- **Reproduktion:** Jeder Lauf wird gegen Q1b bzw. Q2 geprüft (fp32 und symmetrisches w8a4_best). Bestanden, wenn alle 180 gefunden werden und die größte Abweichung höchstens 0,10 pp ist. Sonst gelten die Tests nur als vorläufig.
- Ausfall zählt als 90 %.
- Geschätzte Dauer etwa 45 min auf 2 Workern.

## Primärtest

cs gegen TAT im Testfehler bei **w8a4a_best**, gepaart je Seed. 4 Zellen, Wilcoxon zweiseitig, Holm über 4, α = 0,05. Gesamturteil wie in A/A2/Q2 (besser: ≥ 2 Zellen besser und keine schlechter; schlechter: umgekehrt; kein Effekt: keine Zelle signifikant; sonst gemischt).

## Sekundär (Holm je Familie, je 4 Tests)

- **S1:** cs gegen SELU, w8a4a_best
- **S2:** TAT asymmetrisch gegen TAT symmetrisch (w8a4a_best − w8a4_best)
- **S3:** cs asymmetrisch gegen cs symmetrisch
- **S4:** cs gegen TAT symmetrisch (Wiederholung des Q2-Primärtests mit neu trainierten Läufen)
- **Beschreibend:** w8a8a_max, w8a4a_max, untere Grenze lo, gewählte Kalibrierung

## Vorab-Erwartung

cs gegen TAT bei w8a4a_best:

| Urteil | Erwartung |
|---|---|
| besser als TAT | 0,45 |
| kein Effekt | 0,20 |
| schlechter als TAT | 0,20 |
| gemischt | 0,15 |

S2 (asymmetrisch hilft TAT): 0,85. S3 (asymmetrisch hilft cs): 0,4, weil cs näher an symmetrisch ist (α ≈ 2, beschränkt auf beiden Seiten).

Begründung: In Q2 war der Vorsprung bei d100 rund 6,5 pp. Asymmetrie gewinnt für Leaky ReLU etwa 1 Bit an nutzbarem Raster (Selbsttest: halber MSE an einer Leaky-Verteilung). Das kann viel, aber vermutlich nicht alles aufholen. Der große Bereich von TAT (bis 18,6) bleibt.

## Bedeutung

- **cs „besser“:** Die Nische „beschränkt und robust bei 4 Bit“ ist unabhängig vom Schema (symmetrisch und asymmetrisch pro Tensor). Das kann so in v0.2.
- **„kein Effekt“ oder „schlechter“:** Der Vorteil aus Q2 hängt am symmetrischen Schema. In v0.2 heißt es dann nur: „bei symmetrischer 4-Bit-Quantisierung“.
- **„gemischt“:** Zellen einzeln berichten.

## Grenzen

- Nur Post-Training-Quantisierung, simuliert, pro Tensor. Kein pro Kanal, kein QAT.
- Adam statt K-FAC für TAT.
- MLPs, Breite 256, 1720 Schritte, MNIST/Fashion.

## Freeze

Eingefroren werden `q3_run.py`, `q3_analyze.py`, `lr_from_p3a3.csv`, `lr_tat_from_q2.csv`, `src/precap/*.py` und diese Datei.

Vorab gemachte Tests:
- **Selbsttest:** 180 Bestätigungsläufe, 0 Tuning. Asymmetrisches Raster ok, 0 exakt darstellbar, asymmetrischer MSE an einer Leaky-Verteilung 0,0053 gegen symmetrisch 0,0108.
- **Auswertung an künstlichen Daten:** Nullfall ergibt „kein Effekt“, eingebauter Effekt −3 pp bei d100 ergibt „besser“ in genau diesen 2 Zellen, eine künstlich verfälschte Reproduktion (+0,2 pp) wird erkannt. S4 auf echten Q1b/Q2-Daten reproduziert das Q2-Urteil „besser als TAT“.
- **Rauchtest** (40 Schritte, MNIST d100, je 1 Lauf cs/TAT/SELU): 3/3 ok.
