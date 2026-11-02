# B7 – Auswertung: Hält der Vorteil von BPreCap über LB bei vierfachem Budget? (CIFAR-10, d100)

Stand: 31.10.2026, 09:50. Die Auswertung folgt strikt `B7_Praeregistrierung.md`. PreCap, die Tiefenfamilie und Lagma sind Aarons Ideen. Lauf und Auswertung habe ich mit Claude gemacht.

## Kurz

- **Erfolg.** BPreCap ist auch nach 6880 Schritten (etwa 20 Epochen) in fp32 besser als LB: −2,14 pp, 12/0, p = 0,0005.
- **LB holt nicht auf.** Der Abstand bleibt über alle drei Messpunkte fast gleich: −2,22 pp (1720), −2,30 (3440), −2,14 (6880). S3 ist „unklar“. Der Vorteil ist also mehr als ein schnellerer Start. Der Einwand aus „Nonlinear Advantage“ trifft hier nicht zu.
- **Unerwartet: TAT verliert bei längerem Training seinen 4-Bit-Vorteil.**
  - TAT passt sich zu stark an die Trainingsdaten an: Der Trainingsverlust fällt auf 0,53, der Testfehler steigt wieder von 49,8 auf 50,4 %.
  - Sein Aktivierungsbereich wächst von 6,7 auf 8,1, und der 4-Bit-Preis steigt von +5,3 auf +9,0 pp.
  - Bei BPreCap bleibt der Bereich fest bei 3,0, der Preis bleibt bei etwa +5 pp (S7 „unklar“).
  - **Ergebnis:** Nach 6880 Schritten ist BPreCap bei fairen 4 Bit besser als TAT (−1,97 pp, 12/0), **auch ohne TAT-Rezept** (Gauß-Init). LB ebenfalls (−1,0 pp, 11/1).
- **Einschränkung bei symmetrischen 4 Bit:** Dort ist LB besser als BPreCap (+0,54 pp, 1/11, Holm 0,037). Asymmetrisch ist es umgekehrt (−0,80 pp, 10/2).

## Integrität

| Prüfung | Ergebnis |
|---|---|
| Läufe | 38 von 38 ok, 0 Ausfälle |
| Hash-Prüfung | 18 von 18 OK |
| Kontrolle gegen B3 | kL2B und LB, Seed 42084: Zwischenmessung nach 1720 Schritten in allen 7 Kennzahlen **bitgleich** zu den Endwerten von B3 |
| Schlaf | Größte Lücke zwischen zwei Läufen 386 s bzw. 417 s. Das entspricht der Laufzeit eines Laufs (6880 Schritte plus Messungen), kein Schlaf |
| Laufzeit | 08:36–09:46 |
| Sicherung | `sicherung/b7_2026-10-31/`, 12 von 12 Dateien OK |

Befehl: `.venv/bin/python -I b7_analyze.py ~/…/studien/b7 ~/…/studien/b3`, Ergebnis in `analysis/b7_result.json`.

**Speicher:** Die Worker lagen bei je etwa 4,1 GB, davon etwa 2,5 GB MLX-Puffercache, und der Mac hat in den Swap ausgelagert. Das kostet Zeit, verändert aber keine Rechnung (die Kontrolle ist bitgleich). Ab B8 ist der Cache begrenzt.

## Primär und sekundär (12 Seeds)

| Test | Vergleich | Median | Siege/Niederlagen | p bzw. Holm | Urteil |
|---|---|---|---|---|---|
| **Primär** | kL2B − LB, fp32 (6880) | **−2,14 pp** | 12/0 | 0,0005 | **besser** |
| S1 | kL2B − LB, w4 (6880) | +0,54 | 1/11 | 0,037 | schlechter |
| S2 | kL2B − LB, w4a (6880) | −0,80 | 10/2 | 0,020 | besser |
| S3 | Abstand kL2B − LB fp32, 6880 minus 1720 | +0,57 | 4/8 | 0,30 | unklar |
| S4 | kL2B − TAT, fp32 (6880) | +2,23 | 0/12 | 0,004 | schlechter |
| S5 | kL2B − TAT, w4a (6880) | **−1,97** | 12/0 | 0,004 | **besser** |
| S6 | LB − TAT, w4a (6880) | −1,00 | 11/1 | 0,005 | besser |
| S7 | 4-Bit-Preis kL2B, 6880 minus 1720 | +0,43 | 3/9 | 0,30 | unklar |
| S8 | Abstand kL2B − TAT fp32, 6880 minus 1720 | −2,82 | 12/0 | 0,004 | besser (Abstand schrumpft) |

Siege und Niederlagen zählen negative bzw. positive Differenzen. Bei S3, S7 und S8 bedeutet „Sieg“ eine Abnahme.

## Verlauf über das Training (Mediane, Testfehler in %)

| Form | Schritt | fp32 | w8a4 | w8a4a | w8a3 | Bereich | 4-Bit-Preis (w4a − fp32) | Trainingsverlust |
|---|---|---|---|---|---|---|---|---|
| kL2B | 1720 | 56,4 | 62,8 | 61,1 | 79,6 | 3,0 | +4,9 | 1,53 |
| kL2B | 3440 | 53,3 | 59,1 | 57,8 | 77,6 | 3,0 | +4,5 | 1,35 |
| kL2B | 6880 | **52,2** | 58,0 | **57,1** | 76,8 | 3,0 | +5,0 | 1,07 |
| LB | 1720 | 58,9 | 62,1 | 61,9 | 74,5 | 2,22 | +2,8 | 1,64 |
| LB | 3440 | 55,6 | 58,6 | 58,5 | 70,9 | 2,22 | +2,9 | 1,46 |
| LB | 6880 | 54,1 | **57,6** | 58,0 | **70,2** | 2,22 | +3,7 | 1,11 |
| TAT | 1720 | 51,7 | 60,1 | 57,0 | 84,2 | 6,65 | +5,3 | 1,29 |
| TAT | 3440 | **49,8** | 60,6 | 57,0 | 84,4 | 7,26 | +7,1 | 1,02 |
| TAT | 6880 | 50,4 | 63,3 | 59,4 | 85,5 | 8,07 | +9,0 | 0,53 |

Gewinn durch das längere Training in fp32 (6880 minus 1720, Median je Seed): kL2B −4,6, LB −4,8, TAT −1,4 pp.

## Deutung nach den Vorab-Regeln

- **„Erfolg und S3 unklar“:** Der Vorteil von BPreCap hält bei längerem Training und ist mehr als ein schnellerer Start.
- **S7 „unklar“:** Der 4-Bit-Preis von BPreCap wächst mit dem Training nicht messbar. Die feste Grenze hält den Bereich bei 3,0.
- **S5 „besser“:** Bei fairen 4 Bit und längerem Training ist BPreCap besser als TAT, hier sogar im eigenen Rezept (Gauß-Init) gegen TAT im TAT-Rezept.
- **Zusatz (nicht vorab als Regel formuliert):**
  - TAT ist zwischen 3440 und 6880 Schritten in fp32 nicht mehr besser geworden, der Testfehler steigt leicht.
  - Der Trainingsverlust von TAT ist halb so groß wie der von BPreCap. Das ist Überanpassung bei konstanter LR ohne frühes Stoppen.
  - Mit frühem Stoppen läge TAT in fp32 bei etwa 49,8 % und bei w4a bei etwa 57,0 %, also gleichauf mit BPreCap nach 6880 Schritten (57,1 %). Das ist eine beschreibende Einordnung, kein Test.

## Abgleich mit der Vorab-Erwartung

| Größe | Erwartung | Ergebnis |
|---|---|---|
| Primär „besser“ | 0,6 | eingetreten |
| Primär „unklar“ | 0,3 | nein |
| Primär „schlechter“ | 0,1 | nein |
| S3 „schlechter“ (Vorsprung schrumpft) | 0,45 | nein, unklar |
| S4 „schlechter“ | 0,85 | ja |
| S5 „besser“ | 0,1 | **ja** |
| S7 „schlechter“ (Preis wächst) | 0,5 | nein, unklar |
| Kontrolle bitgleich | 0,95 | ja |

## Antwort auf die Frage

- Ja, der fp32-Vorteil von BPreCap über LB hält bei vierfachem Budget fast unverändert, etwa −2,1 bis −2,3 pp über den ganzen Verlauf.
- Zusätzlich zeigt sich ein Vorteil der festen Grenze, den kurze Läufe nicht sichtbar machen: Bei längerem Training wächst der Bereich einer unbeschränkten Aktivierung (TAT), die beschränkte Form bleibt bei 4 Bit stabil.

**Vertrauen: 0,85** für „BPreCap bleibt bei längerem Training vor LB in fp32“. **0,6** für „BPreCap ist nach längerem Training bei 4 Bit besser als TAT“: Ein Teil davon ist Überanpassung von TAT bei konstanter LR, frühes Stoppen oder ein LR-Plan könnten das verkleinern.

## Grenzen

- LR nicht für das längere Budget getunt, kein LR-Plan, kein frühes Stoppen. Das trifft TAT sichtbar stärker als die beschränkten Formen.
- Nur d100, ein Datensatz, MLP ohne Faltungen.
- TAT im eigenen Rezept, BPreCap und LB mit Gauß-Init. Den Vergleich im gleichen Rezept bei längerem Training gibt es noch nicht (Idee B9).
