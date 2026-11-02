# B9 – Präregistrierung: BPreCap gegen TAT im gleichen Rezept bei längerem Training mit frühem Stoppen (CIFAR-10, d100, privat)

Stand: 31.10.2026, vormittags, vor dem Start. PreCap, die Tiefenfamilie und Lagma sind Aarons Ideen. BPreCap (= kL2B) ist ein Arbeitsname. Das Design habe ich mit Claude entworfen. Aaron hat für heute die Freigabe gegeben: alle sinnvollen Versuche und Recherchen, nichts pushen oder veröffentlichen.

## Ausgangslage

- **B6 (1720 Schritte, gleiches Rezept):** BPreCap_tr ist bei w4a −0,76 pp besser als TAT, in fp32 +1,0 pp schlechter.
- **B7 (6880 Schritte, Gauß-Init gegen TAT-Rezept, kein frühes Stoppen):** TAT passt sich zu stark an. Sein Bereich wächst von 6,7 auf 8,1, und BPreCap ist bei w4a −1,97 pp besser.
  - **Offener Einwand:** Der Vergleich am Ende benachteiligt TAT. Mit frühem Stoppen läge TAT bei w4a etwa gleichauf.
  - Außerdem lief BPreCap dort nicht im gleichen Rezept.
- **B9 schließt beides:** gleiches Rezept, langes Training, Auswertung am besten Validierungspunkt jeder einzelnen Läufe (frühes Stoppen). Das ist der faire Vergleich für eine Aussage „bei 4 Bit mindestens so gut wie TAT“.

## Frage

Ist BPreCap im TAT-Rezept auf CIFAR-10 (d100) bei fairen 4 Bit besser als TAT, wenn beide vier mal so lange trainieren und jeweils am eigenen besten Validierungspunkt gemessen werden?

## Eine Änderung gegenüber B6: Budget 6880 Schritte mit Messpunkten und frühem Stoppen

- **Varianten:**
  - kL2B_tr und LB_tr mit der LR aus B6 (5e-5, dort nicht am Rand)
  - TAT mit 5e-5 aus B3
- **Messpunkte** alle 860 Schritte (860 … 6020) und am Ende (6880). Gemessen werden fp32 (Val und Test), w8a4_best, w8a4a_best, w8a3_best, Bereich und Verlust. Wie in B7 sind es nur Vorwärtsrechnungen, das Training bleibt bitgleich (Mechanik-Test).
- **Frühes Stoppen (in der Auswertung):**
  - Je Lauf zählt der Messpunkt mit dem kleinsten **Validierungsfehler** in fp32, bei Gleichstand der frühere.
  - Test- und 4-Bit-Werte werden an diesem Punkt abgelesen.
  - Die Auswahl nutzt nur Validierungsdaten (5000 aus dem Trainingsteil), die Testdaten nie.
- **Ausfall:** Es zählen nur die Messpunkte vor dem Ausfall. Ohne jeden Messpunkt zählt der Lauf als 90 %. Endwerte eines ausgefallenen Laufs zählen als 90 %.
- **Konstanten wie B6:** Formen, Rezept, Netz, Adam mit konstanter LR, Batch 128, Daten, Kalibrierung, Quantisierung. Der MLX-Cache ist auf 512 MB begrenzt (wie B8, bitgleich geprüft).
- **Seeds:** neu 42156–42167.

## Läufe

- 3 Formen × 12 Seeds = 36.
- **Kontrolle:** kL2B_tr und TAT mit Seed 42120 (B6-Seed). Der Messpunkt 1720 muss in allen 7 Kennzahlen bitgleich zu den Endwerten von B6 sein.
- Zusammen 38 Läufe, etwa 1,5 h. Start nach B8 und nur am Netzteil.

## Primärtest

**kL2B_tr − TAT bei w8a4a_best am Punkt des frühen Stoppens.** Wilcoxon zweiseitig exakt, eine Zelle, α = 0,05.

**Erfolg:** p < 0,05 und Median < 0 („besser“).

## Sekundär (Holm über alle 8)

| Nr. | Vergleich | Frage |
|---|---|---|
| S1 | kL2B_tr − TAT, fp32, frühes Stoppen | Abstand in fp32 |
| S2 | kL2B_tr − TAT, w8a4_best, frühes Stoppen | symmetrische 4 Bit |
| S3 | LB_tr − TAT, w8a4a_best, frühes Stoppen | LB bei fairen 4 Bit |
| S4 | kL2B_tr − LB_tr, fp32, frühes Stoppen | Kern-Befund im Rezept bei langem Training |
| S5 | kL2B_tr − LB_tr, w8a4a_best, frühes Stoppen | Kern bei 4 Bit |
| S6 | kL2B_tr − TAT, w8a4a_best, Ende (6880) | wie B7, ohne frühes Stoppen |
| S7 | kL2B_tr − TAT, fp32, Ende (6880) | dito fp32 |
| S8 | LB_tr − TAT, fp32, frühes Stoppen | LB in fp32 |

**Beschreibend:**
- Verteilung der Stopp-Punkte je Form.
- Verlauf aller Metriken über die 8 Messpunkte.
- Bereich und Verlust (Überanpassung).
- 4-Bit-Preis am Stopp-Punkt.

## Deutungsregeln (vorab)

| Ergebnis | Deutung |
|---|---|
| Erfolg | Im fairen Vergleich (gleiches Rezept, frühes Stoppen, langes Training) ist BPreCap bei 4 Bit besser als TAT. Das ist die Kernaussage für eine v0.3 |
| Primär „unklar“ | Fair verglichen ist BPreCap bei 4 Bit gleichauf mit TAT. Der Vorteil aus B6/B7 hing an kurzem Training bzw. an der Überanpassung von TAT |
| Primär „schlechter“ | Fair verglichen ist TAT bei 4 Bit vorn |
| S6 „besser“, Primär nicht | Der Vorteil aus B7 kam von der Überanpassung von TAT |
| S2 „besser“ | Bei symmetrischen 4 Bit ist BPreCap besser als TAT |
| S4 „besser“ | Der Kern-Befund gilt auch im Rezept bei langem Training |
| Kontrolle nicht bitgleich | Abweichung wird gesucht und berichtet, die Tests gelten |

## Vorab-Erwartung

| Größe | Erwartung |
|---|---|
| Primär „besser“ (Erfolg) | 0,45 |
| Primär „unklar“ | 0,4 |
| Primär „schlechter“ | 0,15 |
| S1 „schlechter“ (TAT in fp32 vorn) | 0,75 |
| S2 „besser“ | 0,85 |
| S4 „besser“ | 0,75 |
| S6 „besser“ | 0,75 |
| Kontrolle bitgleich | 0,95 |

## Grenzen

- LR für das lange Budget nicht neu getunt (aus B6/B3 für 1720 Schritte), kein LR-Plan.
- Frühes Stoppen nur an 8 Punkten (alle 860 Schritte).
- Nur d100, ein Datensatz, MLP ohne Faltungen. TAT ist unsere Nachbildung nach dks.

## Freeze

Eingefroren werden `b9_run.py`, `b9_analyze.py`, `test_b9_mechanik.py`, `test_b9_analyse.py`, diese Datei, `lr_from_a.csv`, `lr_from_p3a3.csv`, `lr_from_b3.csv`, `lr_from_b6.csv`, `src/precap/*.py` und die CIFAR-Dateien.

Vorab gemachte Tests (31.10., vormittags):

- **Selbsttest:** n_confirm 38, 6880 Schritte, 7 Zwischen-Messpunkte.
- **Mechanik** (`test_b9_mechanik.py`): **MECHANIK OK** (31.10., 10:00). Das Startskript führt den Test erneut aus und bricht bei Fehler ab. Geprüft wird:
  - Formen, Init und Daten gleich B6.
  - LR 5e-5 / 5e-5 / 5e-5, 38 Läufe, Seeds neu, Kontrolle auf dem B6-Seed.
  - Zwischenmessung bitgleich zu einem kürzeren B6-Lauf, Endwert bitgleich zu einem gleich langen B6-Lauf ohne Zwischenmessung (alle drei Formen).
- **Auswertung an künstlichen Daten** (`test_b9_analyse.py`): AUSWERTUNG OK.
  - „BPreCap vorn“ (TAT überangepasst) wird als Erfolg erkannt, die Stopp-Punkte werden richtig gewählt.
  - „TAT vorn“ ergibt „schlechter“.
  - Null mit 20 Wiederholungen: 1 Fehlalarm primär, 1 sekundär.
  - Eine falsche Kontrolle wird erkannt. Beim Ausfall nach 3440 wird der frühere Punkt genommen, ohne Messpunkt zählt der Lauf als 90 %. Der relative Pfad findet B6.
  - **Offen genannt:** Meine erste Testerwartung für die Stopp-Punkte von TAT (2580/3440) war falsch abgezählt. Das Minimum der künstlichen Kurve liegt bei Index 3, also Schritt 3440, mit Rauschen auch 4300. Die Auswertung hat richtig gewählt, korrigiert wurde nur die Erwartung im Test.
- **Rauchtest:** läuft im Startskript (5 Läufe mit 40 Schritten, Zwischenwerte bei 10 und 20).
