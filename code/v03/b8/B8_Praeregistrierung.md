# B8 – Präregistrierung: Hält der B6-Befund bei Tiefe 200? (BPreCap im TAT-Rezept gegen TAT, CIFAR-10, privat)

Stand: 31.10.2026, vormittags, vor dem Start. PreCap, die Tiefenfamilie und Lagma sind Aarons Ideen. BPreCap (= kL2B) ist ein Arbeitsname. Das Design habe ich mit Claude entworfen. Aaron hat für heute die Freigabe gegeben: alle sinnvollen Versuche und Recherchen, nichts pushen oder veröffentlichen.

## Ausgangslage

- **B6 (d100):** Im TAT-Rezept (SUO-orthogonale Init mal Gain, Normierung je Beispiel) ist BPreCap bei fairen 4 Bit besser als TAT: w8a4a −0,76 pp (10/2), w8a4 −2,59 pp (12/0). In fp32 liegt es +1,0 pp dahinter.
- **B5 (d200, Gauß-Init):** BPreCap ist bei w8a4a +2,66 pp schlechter als TAT, bei w8a4 −1,69 pp besser. In fp32 +9,4 pp.
- Für eine v0.3-Aussage „bei 4 Bit mindestens so gut wie TAT“ fehlt die größere Tiefe im gleichen Rezept.

## Frage

Ist BPreCap im TAT-Rezept auch bei Tiefe 200 auf CIFAR-10 bei fairen (asymmetrischen) 4 Bit besser als TAT?

## Eine Änderung gegenüber B6: Tiefe 200 statt 100

- **Varianten:**
  - kL2B_tr (BPreCap im TAT-Rezept)
  - LB_tr (LB im TAT-Rezept)
  - TAT (eigenes Rezept, wie B3)
- **LR:**
  - kL2B_tr und LB_tr werden neu getunt (Seeds 42000–42003, Regel wie B3).
  - Das Raster ist eine Stufe nach unten verschoben: 5e-6, 1e-5, 2e-5, 5e-5, 1e-4, 2e-4. Grund: In B3/B5 wählten alle Formen bei d200 kleinere Lernraten als bei d100.
  - TAT bekommt 2e-5 aus B3 d200 (dort nicht am Rand).
- **Konstanten wie B6:** Formen, Rezept, Netz (Breite 256), Adam, Batch 128, 1720 Schritte, Daten, Kalibrierung, Quantisierung.
- **Technisch neu (ohne Einfluss auf die Rechnung):** Der MLX-Puffercache ist je Worker auf 512 MB begrenzt. In B7 belegte der Cache etwa 2,5 GB je Worker, und der Mac lagerte in den Swap aus. Der Mechanik-Test zeigt, dass die Ergebnisse mit und ohne Grenze bitgleich sind.
- **Seeds:** neu 42144–42155, gepaart.

## Läufe

- **Tuning:** 2 × 6 × 4 = 48.
- **Bestätigung:** 3 × 12 = 36, dazu die Kontrolle kL2B MNIST d100 Seed 42060 (bitgleich zu B1).
- Zusammen 85 Läufe bei d200, etwa 1,5 h. Start nach B7 und nur am Netzteil. Ein Ausfall zählt als 90 %.

## Primärtest

**kL2B_tr − TAT bei w8a4a_best, d200.** Wilcoxon zweiseitig exakt, eine Zelle, α = 0,05.

**Erfolg:** p < 0,05 und Median < 0 („besser“).

## Sekundär (Holm über alle 8)

| Nr. | Vergleich | Frage |
|---|---|---|
| S1 | kL2B_tr − TAT, fp32 | Abstand in fp32 bei d200 |
| S2 | kL2B_tr − TAT, w8a4_best | symmetrische 4 Bit |
| S3 | LB_tr − TAT, w8a4a_best | dito für LB |
| S4 | kL2B_tr − LB_tr, fp32 | Kern-Befund (B3/B5/B6) bei d200 im Rezept |
| S5 | kL2B_tr − LB_tr, w8a4a_best | Kern bei 4 Bit |
| S6 | LB_tr − TAT, fp32 | LB in fp32 |
| S7 | LB_tr − TAT, w8a4_best | LB symmetrisch |
| S8 | kL2B_tr − LB_tr, w8a4_best | Kern symmetrisch |

**Beschreibend:**
- Tuning-Kurven und LR-Wahl (Randlage an beiden Enden).
- Mediane aller Metriken (auch 3 Bit) mit Bereich und Sättigung.
- fp32-Ordnung.
- Vergleich mit B5 (Gauß-Init, d200) und B6 (d100), nur auf Ebene der Mediane, da die Seeds verschieden sind.

## Deutungsregeln (vorab)

| Ergebnis | Deutung |
|---|---|
| Erfolg | Der B6-Befund gilt auch bei d200: Im gleichen Rezept ist BPreCap bei fairen 4 Bit besser als TAT |
| Primär „unklar“ | Bei d200 gleichauf. Für v0.3 ginge dann höchstens „bei 4 Bit gleichwertig“ |
| Primär „schlechter“ | Bei d200 ist TAT auch bei 4 Bit vorn. Der B6-Befund hängt an der Tiefe |
| S2 „besser“ | Bei symmetrischen 4 Bit ist BPreCap besser als TAT |
| S4 „besser“ | Der Kern-Befund gilt auch bei d200 im TAT-Rezept |
| LR am Rand | Wird als Einschränkung berichtet |
| Kontrolle nicht bitgleich | Abweichung wird gesucht und berichtet, die Tests gelten |

## Vorab-Erwartung

| Größe | Erwartung |
|---|---|
| Primär „besser“ (Erfolg) | 0,55 |
| Primär „unklar“ | 0,3 |
| Primär „schlechter“ | 0,15 |
| S1 „schlechter“ (TAT in fp32 vorn) | 0,8 |
| S2 „besser“ | 0,85 |
| S4 „besser“ | 0,8 |
| eine _tr-LR am Rand | 0,2 |
| Kontrolle bitgleich | 0,95 |

## Grenzen

- Wie B6: Init und Normierung als Paket, kurzes Budget, MLP ohne Faltungen, ein Datensatz.
- TAT wird nicht neu getunt (LR aus B3 d200, dort nicht am Rand).
- TAT ist unsere Nachbildung nach dks (Leaky-ReLU-TAT).

## Freeze

Eingefroren werden `b8_run.py`, `b8_analyze.py`, `test_b8_mechanik.py`, `test_b8_analyse.py`, diese Datei, `lr_from_a.csv`, `lr_from_p3a3.csv`, `lr_from_b3.csv`, `src/precap/*.py` und die CIFAR-Dateien.

Vorab gemachte Tests (31.10., vormittags):

- **Selbsttest:** n_tune 48, n_confirm 37, GPU.
- **Mechanik** (`test_b8_mechanik.py`), erster Durchgang ohne Cache-Prüfung: MECHANIK OK.
  - Formen, Init und Rezept sind gleich B6 bei d200, die Init ist orthogonal mal Gain (g = 1,000004).
  - Die Vor-Aktivierungs-Varianz liegt nach 200 Schichten bei 1,004 (kL2B_tr) bzw. 1,010 (LB_tr).
  - Raster, 48/37 Läufe, Seeds neu, TAT-LR 2e-5.
  - `execute` ist bitgleich zu B6 (kL2B_tr, LB_tr) und B3 (TAT) bei d200, der Kontrollpfad bitgleich zu B1.
  - **Cache-Prüfung** (zweiter Durchgang): getrennte Prozesse mit und ohne Grenze, CIFAR-10 d200 kL2B_tr, 60 Schritte. Die Ergebnisse sind **bitgleich**, der Cache sinkt von 2,51 auf 0,54 GB.
    - Meine erste Bedingung „Cache ≤ genau 512 MiB“ war zu streng, denn MLX gibt den Cache erst bei der nächsten Anforderung frei. Neue Bedingung: höchstens 25 % über der Grenze und weniger als halb so viel wie ohne Grenze.
    - Das Startskript führt den ganzen Test erneut aus, Ergebnis in `meta/mechanik.txt`.
- **Auswertung an künstlichen Daten** (`test_b8_analyse.py`): AUSWERTUNG OK.
  - „Hält“ wird als Erfolg erkannt, „TAT vorn“ als „schlechter“.
  - Null mit 20 Wiederholungen: 0 Fehlalarme primär, 1 sekundär.
  - Eine falsche Kontrolle wird erkannt, ein Ausfall gezählt, der relative Pfad findet B1.
- **Rauchtest:** läuft im Startskript (4 Tuning- und 4 Bestätigungsläufe, d100).
