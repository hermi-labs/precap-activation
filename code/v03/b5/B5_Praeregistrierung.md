# B5 – Präregistrierung: Hält der B3-Befund bei Tiefe 200 nach Nachtuning der Lernrate? (CIFAR-10, privat)

Stand: 31.10.2026, nachts, vor dem Start. PreCap, die Tiefenfamilie und Lagma sind Aarons Ideen. BPreCap (= kL2B) ist ein Arbeitsname. Das Design habe ich mit Claude entworfen, mit Aarons Freigabe für die Nacht („freie Hand zu testen und zu entwickeln, exakt begründet, nichts veröffentlichen oder pushen“). Anlass ist die Vorab-Regel aus B3: „Bei Randlage in mehreren Varianten folgt ein Nachtuning in einer eigenen Studie.“

## Ausgangslage (B3, `studien/b3/B3_Auswertung.md`)

- Auf CIFAR-10 ist BPreCap in fp32 besser als LB: d100 −2,95 pp, d200 −4,79 pp, jeweils 12/12.
- Bei d200 lagen kL2B, LB und kL2 am **unteren Rand** des Rasters (1e-5). TAT lag nicht am Rand (2e-5).
- **Stabilitätskante:** Bei d200 fällt LB schon bei 2e-5 auf 15 % Val-Genauigkeit, kL2B auf 36 %, kL2 bleibt bei 43 %. Es ist also möglich, dass LB bei kleinerer LR deutlich aufholt und der d200-Befund aus B3 ein LR-Artefakt ist.

## Frage

Ist BPreCap bei Tiefe 200 auf CIFAR-10 in fp32 auch dann besser als LB, wenn beide Formen mit kleineren Lernraten nachgetunt sind?

## Eine Änderung gegenüber B3: das LR-Raster nach unten

- **Raster:** 3e-6, 5e-6, 7e-6, 1e-5. Nur d200, getunt werden nur die drei Randformen kL2B, LB und kL2.
- **Tuning-Seeds und Regel wie B3** (42000–42003, höchste mediane Val-Genauigkeit am letzten Schritt, Ausfall zählt 0,1, bei Gleichstand die kleinere LR).
- **Anschluss an B3:** Die Zellen mit 1e-5 werden neu gerechnet und müssen bitgleich zu B3 sein. Über 1e-5 war in B3 bei allen drei Formen jede LR schlechter. Die Wahl über das neue Raster ist deshalb gleich der Wahl über die Vereinigung beider Raster. Als Randlage zählt nur der untere Rand 3e-6.
- **TAT:** nicht am Rand, LR aus B3 (2e-5, `lr_from_b3.csv`), nur Bestätigung.
- **Konstanten wie B3:** Formen, Init, Netz (Breite 256, ohne Normierung), Adam, Batch 128, 1720 Schritte, Daten und Teilung, Kalibrierung, Quantisierung.
- **Seeds:** neu 42108–42119 (12 je Form), gepaart.

## Läufe

- **Tuning:** 3 Formen × 4 LR × 4 Seeds = 48 Läufe (davon 12 Kontrollen gegen B3).
- **Bestätigung:** 4 Formen × 12 Seeds = 48 Läufe.
- **Kontrolle:** kL2B MNIST d100 Seed 42060, bitgleich zu B1.
- Zusammen 97 Läufe, etwa 1,5 h mit 2 Workern. Start erst nach dem Ende von B4 und nur am Netzteil. Ein Ausfall zählt als 90 %.

## Primärtest

**kL2B − LB, fp32, d200**, Wilcoxon zweiseitig exakt, eine Zelle, α = 0,05.

**Erfolg:** p < 0,05 und Median < 0 („besser“).

## Sekundär (Holm über alle 8)

| Nr. | Vergleich | Frage |
|---|---|---|
| S1 | kL2B − LB, w8a4_best | 4 Bit symmetrisch |
| S2 | kL2B − LB, w8a4a_best | 4 Bit asymmetrisch |
| S3 | kL2 − LB, fp32 | p8: mehr Init-Nichtlinearität |
| S4 | kL2B − kL2, fp32 | Preis der Grenze in fp32 |
| S5 | kL2B − TAT, fp32 | Abstand zu TAT |
| S6 | kL2B − TAT, w8a4a_best | faire 4 Bit gegen TAT |
| S7 | LB − TAT, w8a4a_best | dito für LB |
| S8 | kL2B − TAT, w8a4_best | in B3 beschreibend kL2B vorn (69,9 gegen 72,6), jetzt als Test |

**Beschreibend:** Tuning-Kurven, gewählte LR (Randlage unten), Mediane aller Metriken, p8-Ordnung (kL2 < kL2B < LB in fp32), Vergleich der Mediane mit B3 d200.

## Deutungsregeln (vorab)

| Ergebnis | Deutung |
|---|---|
| Erfolg | Der fp32-Vorteil von BPreCap bei d200 ist kein LR-Artefakt. Der B3-Befund gilt in beiden Tiefen |
| Primär „unklar“ | LB holt bei kleinerer LR auf. Der d200-Befund aus B3 hing an der LR. Bei d100 (nicht am Rand) bleibt der B3-Befund |
| Primär „schlechter“ | Umkehr: Bei passender LR ist LB bei d200 besser |
| LR einer Form bei 3e-6 | Randlage unten wird als Einschränkung berichtet, in dieser Nacht kein weiteres Nachtuning |
| S8 „besser“ | Bei symmetrischen 4 Bit und d200 ist BPreCap besser als TAT |
| Kontrolle B1 oder B3 nicht bitgleich | Abweichung wird gesucht und berichtet, die Tests gelten |

## Vorab-Erwartung

| Größe | Erwartung |
|---|---|
| Primär „besser“ (Erfolg) | 0,75 |
| Primär „unklar“ | 0,2 |
| Primär „schlechter“ | 0,05 |
| LB wählt eine LR unter 1e-5 | 0,6 |
| kL2B wählt eine LR unter 1e-5 | 0,4 |
| eine Form wählt 3e-6 | 0,15 |
| S3 „besser“ | 0,85 |
| S6 „schlechter“ | 0,65 |
| S8 „besser“ | 0,35 |
| p8-Ordnung gilt | 0,65 |
| Kontrollen bitgleich (B1 und 12/12 B3) | 0,95 |

## Grenzen

- Wie B3: MLP ohne Faltungen, kurzes Budget (etwa 5 Epochen), ein Datensatz, TAT im eigenen Rezept.
- Nur d200 wird nachgetunt. d100 lag in B3 nicht am Rand.
- TAT wird nicht nachgetunt (nicht am Rand). Ein kleinerer Vorteil von TAT bei feinerem Raster ist möglich.
- Neue Seeds, daher ist B5 zugleich eine Wiederholung von B3 d200 auf anderen Seeds.

## Freeze

Eingefroren werden `b5_run.py`, `b5_analyze.py`, `test_b5_mechanik.py`, `test_b5_analyse.py`, diese Datei, `lr_from_a.csv`, `lr_from_p3a3.csv`, `lr_from_b3.csv`, `src/precap/*.py` und die CIFAR-Dateien.

Vorab gemachte Tests (31.10., nachts):

- **Mechanik** (`test_b5_mechanik.py`): MECHANIK OK.
  - Formen, Gains und TAT gleich B3.
  - Raster, Formen, 48 Tuning- und 49 Bestätigungsläufe, Seeds neu.
  - TAT-LR 2e-5 aus B3, Tuning-Lauf-Namen gleich B3.
  - `execute` auf CIFAR-10 d200 für alle vier Formen bitgleich zu B3, Kontrollpfad bitgleich zu B1.
- **Auswertung an künstlichen Daten** (`test_b5_analyse.py`): AUSWERTUNG OK.
  - „Erfolg“ wird erkannt, ebenso „LB holt auf“ als „unklar“.
  - Null mit 20 Wiederholungen: 2 Fehlalarme primär (bei α = 0,05 erwartet etwa 1), 0 sekundär.
  - Falsche Kontrolle gegen B1 und gegen das B3-Tuning wird erkannt, ein Ausfall gezählt, und der relative Pfad findet B1 und B3.
- **Rauchtest:** läuft im Startskript (6 Tuning- und 5 Bestätigungsläufe, d100).
