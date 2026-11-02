# B6 – Präregistrierung: Kommt der TAT-Vorsprung auf CIFAR-10 vom Rezept oder von der Form? (privat)

Stand: 31.10.2026, nachts, vor dem Start. PreCap, die Tiefenfamilie und Lagma sind Aarons Ideen. BPreCap (= kL2B) ist ein Arbeitsname. Das Design habe ich mit Claude entworfen, mit Aarons Freigabe für die Nacht („freie Hand zu testen und zu entwickeln, exakt begründet, nichts veröffentlichen oder pushen“).

## Ausgangslage (B3)

- TAT ist auf CIFAR-10 bei d100 in fp32 um 4,9 pp besser als BPreCap und bei asymmetrischen 4 Bit um 4,1 pp.
- TAT läuft aber in einem eigenen Rezept: orthogonale Init (SUO wie dks) und Normierung je Beispiel. BPreCap und LB laufen mit Gauß-Init und global standardisierten Daten. Diese Einschränkung steht seit v0.2 offen.
- Pennington et al. 2017 zeigen, dass orthogonale Gewichte die Konditionierung tiefer kritischer Netze deutlich verbessern („dynamical isometry“). Das wäre eine Erklärung für den TAT-Vorsprung, die nichts mit der Form zu tun hat.

## Frage

Wird BPreCap auf CIFAR-10 (d100) besser, wenn es dasselbe Rezept wie TAT bekommt? Und wie viel vom Abstand zu TAT bleibt dann?

## Eine Änderung: das Rezept (als Paket)

| Variante | Form | Rezept |
|---|---|---|
| **kL2B_tr** (neu) | wie kL2B (BPreCap, B3) | SUO-orthogonale Init mal Gain, Normierung je Beispiel (wie TAT) |
| **LB_tr** (neu) | wie LB (B3) | dito |
| kL2B | B3 | Gauß-Init, wie B3 |
| LB | B3 | wie B3 |
| TAT | B3 | eigenes Rezept, wie B3 |

- Das Rezept wird als Paket geändert (Init und Eingabe zusammen), so wie TAT es nutzt. Getrennt wird das nicht, das wäre eine eigene Studie.
- Der Gain ist bei kL2B/LB praktisch 1 (E[f²] = 1), die Vorwärts-Varianz ist also gleich (Mechanik-Test: 1,00 nach 100 Schichten).
- **Nur d100:** In B3 lag dort keine LR am Rand, und es ist billiger.
- **LR:** kL2B_tr und LB_tr werden neu getunt (Raster und Regel wie B3: 1e-5 bis 5e-4, Seeds 42000–42003). Die Referenzen bekommen die LR aus B3 (kL2B 2e-5, LB 2e-5, TAT 5e-5).
- **Konstanten wie B3:** Netz, Adam, Batch 128, 1720 Schritte, Daten, Teilung, Kalibrierung, Quantisierung.
- **Seeds:** neu 42120–42131, gepaart.

## Läufe

- Tuning: 2 × 6 × 4 = 48.
- Bestätigung: 5 × 12 = 60, dazu die Kontrolle kL2B MNIST d100 Seed 42060 (bitgleich zu B1).
- Zusammen 109 Läufe, etwa 1 h. Start nach B5, nur am Netzteil. Ein Ausfall zählt als 90 %.

## Primärtest

**kL2B_tr − kL2B, fp32**, Wilcoxon zweiseitig exakt, eine Zelle, α = 0,05. **Erfolg:** „besser“ (p < 0,05, Median < 0).

## Sekundär (Holm über alle 8)

| Nr. | Vergleich | Frage |
|---|---|---|
| S1 | LB_tr − LB, fp32 | Rezept bei LB |
| S2 | kL2B_tr − TAT, fp32 | verbleibender Abstand zu TAT (Form) |
| S3 | kL2B_tr − TAT, w8a4a_best | dito bei fairen 4 Bit |
| S4 | LB_tr − TAT, w8a4a_best | dito für LB |
| S5 | kL2B_tr − LB_tr, fp32 | Gilt der Kern-Befund aus B3 auch im TAT-Rezept? |
| S6 | kL2B_tr − kL2B, w8a4_best | Rezept bei 4 Bit symmetrisch |
| S7 | kL2B_tr − kL2B, w8a4a_best | Rezept bei 4 Bit asymmetrisch |
| S8 | kL2B_tr − TAT, w8a4_best | symmetrische 4 Bit gegen TAT |

**Beschreibend:** Tuning-Kurven, LR-Wahl (Randlage an beiden Enden), Mediane aller Metriken, fp32-Ordnung aller fünf, Vergleich der Referenzen mit B3 d100.

## Deutungsregeln (vorab)

| Ergebnis | Deutung |
|---|---|
| Erfolg und S2 „unklar“ | Der TAT-Vorsprung kam vom Rezept. Mit gleichem Rezept ist BPreCap gleichauf mit TAT |
| Erfolg und S2 „schlechter“ | Das Rezept hilft, ein Teil des Abstands kommt aber von der Form (TAT). Der Rest wird beziffert |
| Primär „unklar“ | Das Rezept bringt BPreCap nichts. Der TAT-Vorsprung kommt von der Form bzw. von dks selbst |
| Primär „schlechter“ | Das TAT-Rezept schadet BPreCap |
| S3/S8 „besser“ | Mit gleichem Rezept ist BPreCap bei 4 Bit besser als TAT |
| S5 „besser“ | Der Kern-Befund aus B3 gilt auch im TAT-Rezept |
| LR am Rand | Wird als Einschränkung berichtet |
| Kontrolle nicht bitgleich | Wird gesucht und berichtet, die Tests gelten |

## Vorab-Erwartung

| Größe | Erwartung |
|---|---|
| Primär „besser“ (Erfolg) | 0,55 |
| Primär „unklar“ | 0,35 |
| Primär „schlechter“ | 0,1 |
| S1 „besser“ | 0,5 |
| S2 „schlechter“ | 0,7 |
| S3 „besser“ | 0,15 |
| S5 „besser“ | 0,8 |
| eine _tr-LR am Rand | 0,2 |
| Kontrolle bitgleich | 0,95 |

## Grenzen

- Wie B3: MLP ohne Faltungen, kurzes Budget, ein Datensatz. Nur d100.
- Init und Eingabe-Normierung werden zusammen geändert. Welcher Teil wirkt, bleibt offen.
- Die Referenzen werden nicht neu getunt (LR aus B3, dort nicht am Rand).

## Freeze

Eingefroren werden `b6_run.py`, `b6_analyze.py`, `test_b6_mechanik.py`, `test_b6_analyse.py`, diese Datei, `lr_from_a.csv`, `lr_from_p3a3.csv`, `lr_from_b3.csv`, `src/precap/*.py` und die CIFAR-Dateien.

Vorab gemachte Tests (31.10., nachts):

- **Mechanik** (`test_b6_mechanik.py`): MECHANIK OK.
  - Die _tr-Formen sind gleich der Grundform.
  - Die Init ist orthogonal mal Gain (g = 1,000003), die Ausgabe-Schicht wie TAT.
  - Die Daten sind wie bei TAT normiert, die Referenzen bekommen Daten und Init wie B3.
  - Die Vor-Aktivierungs-Varianz liegt nach 100 Schichten bei 0,997 (kL2B_tr) bzw. 1,008 (LB_tr).
  - 48 Tuning- und 61 Bestätigungsläufe, Seeds neu, LR der Referenzen aus B3.
  - Die Referenzen laufen bitgleich zu B3, die _tr-Varianten laufen, der Kontrollpfad ist bitgleich zu B1.
- **Auswertung an künstlichen Daten** (`test_b6_analyse.py`): AUSWERTUNG OK.
  - „Rezept hilft“ und „Rezept egal“ werden richtig erkannt.
  - Null mit 20 Wiederholungen: 1 Fehlalarm primär, 1 sekundär.
  - Eine falsche Kontrolle wird erkannt, ein Ausfall gezählt, der relative Pfad findet B1.
- **Rauchtest:** läuft im Startskript (4 Tuning- und 6 Bestätigungsläufe).
