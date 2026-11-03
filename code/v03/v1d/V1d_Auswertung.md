# V1d – Auswertung: Vermeidet ein kürzerer LR-Anlauf BPreCaps 4-Bit-Preis bei Tiefe 200?

Stand: 03.11.2026, 17:50, mit Claude. Ausgewertet strikt nach `V1d_Praeregistrierung.md`. Läufe 15:52–17:50 auf 2 Workern (M1 Pro, Netzteil). PreCap und BPreCap sind Aarons Ideen.

## Kurzfassung

- **Nach Präregistrierung: „auch 1 % Anlauf kostet bei 4 Bit“** (Regel 1, P2 „schlechter“). Kein Erfolg.
  - P2: 1 % gegen 0 % ergibt **+0,63 pp**, 0/12, Holm-p 0,001.
  - P1: 1 % gegen 5 % ergibt **−0,91 pp**, 11/1, Holm-p 0,002. Ein kürzerer Anlauf kostet also weniger, der Preis verschwindet aber nicht.
- **Klare Dosis-Wirkung:** Der 4-Bit-Fehler (w8a4a) steigt mit der Länge des Anlaufs:
  - 0 % 53,63, 1 % 54,27, 2,5 % 54,91, 5 % 55,34;
  - die Rangkorrelation je Seed liegt im Median bei 1,0.
- **fp32:** Nur 2,5 % bringt einen nachweisbaren Gewinn gegen 0 % (S5, −0,28 pp, 10/1). 1 % ist unklar (S3, −0,21 pp, 8/4).
  - Der Tausch ist also stetig: Jeder zusätzliche Anlaufschritt kauft ein wenig fp32 und kostet mehr 4 Bit. Der 4-Bit-Preis wächst schneller als der fp32-Gewinn.
- **Kontrolle:** 2 von 2 bitgleich zu V1c (ohne Anlauf und 5 %, Seed 42192). Es gab 0 Ausfälle und 16 von 16 Hashes OK.

## Integrität

| Prüfung | Ergebnis |
|---|---|
| Hashes (`shasum -a 256 -c meta/frozen_sha256.txt`) | 16 von 16 OK (10 Dateien und 6 CIFAR-Dateien) |
| Zeilen | 26 von 26 ok (2 Kontrollen und 24 Läufe), 0 Fehler in den Logs |
| Kontrolle gegen V1c (cos und w344, Seed 42192) | 2 von 2 bitgleich, alle Mess- und Endwerte |

Befehl: `.venv/bin/python -I studien/v1d/v1d_analyze.py studien/v1d studien/v1c studien/b15`, Ergebnis in `analysis/v1d_result.json`. Hinweis von scipy: Bei S5 gibt es eine Null-Differenz, deshalb nutzt S5 statt des exakten Tests die Normalnäherung.

## Primärtests (w8a4a_best am Punkt des frühen Stoppens, Holm über 2)

| Nr. | Vergleich | Median | W/L | p | Holm-p | Urteil |
|---|---|---|---|---|---|---|
| P1 | 1 % − 5 % | −0,91 pp | 11/1 | 0,0024 | 0,0024 | besser |
| P2 | 1 % − 0 % | +0,63 pp | 0/12 | 0,0005 | 0,0010 | schlechter |

**Gesamturteil** nach der Reihenfolge der Präregistrierung: P2 ist „schlechter“, damit gilt **„auch 1 % Anlauf kostet bei 4 Bit“**.

## Sekundär (Holm über 5)

| Nr. | Vergleich | Größe | Median | W/L | Holm-p | Urteil |
|---|---|---|---|---|---|---|
| S1 | 2,5 % − 5 % | w8a4a | −0,59 | 10/2 | 0,024 | besser |
| S2 | 2,5 % − 0 % | w8a4a | +0,96 | 0/12 | 0,002 | schlechter |
| S3 | 1 % − 0 % | fp32 | −0,21 | 8/4 | 0,27 | unklar |
| S4 | 1 % − 5 % | fp32 | +0,38 | 2/10 | 0,010 | schlechter |
| S5 | 2,5 % − 0 % | fp32 | −0,28 | 10/1 | 0,013 | besser |

## Beschreibend (Mediane über 12 Seeds, Fehler in %, am Stopp-Punkt)

| Anlauf | fp32 | w8a4a | w8a4 (sym.) | w8a3 | 4-Bit-Preis (w8a4a − fp32) | sat_rate | Stopp bei 6020 / 6880 |
|---|---|---|---|---|---|---|---|
| 0 % (B15) | 46,37 | 53,63 | 55,49 | 80,02 | 7,28 | 0,0266 | 3 / 9 |
| 1 % | 46,31 | 54,27 | 56,32 | 82,11 | 8,12 | 0,0226 | 6 / 6 |
| 2,5 % | 46,20 | 54,91 | 56,96 | 82,77 | 8,68 | 0,0209 | 4 / 8 |
| 5 % (V1c) | 45,94 | 55,34 | 57,62 | 83,53 | 9,53 | 0,0191 | 1 / 11 |

- **Gleiches Muster bei allen Quantisierungen:** symmetrisch w8a4 und w8a3 steigen ebenfalls mit der Anlauflänge, w8a3 um 3,5 pp von 0 auf 5 %.
- **Sättigungsanteil:** Er sinkt mit der Anlauflänge, von 0,0266 auf 0,0191. Das passt zu D3b: Mit Anlauf nutzt das Netz die Hülle weniger. Das ist eine Korrelation und kein Nachweis der Ursache.
- **Stopp-Punkte:** Sie unterscheiden sich etwas, frühes Stoppen ist Teil der Messgröße nach der Präregistrierung. Nachgerechnet an den Endwerten (Schritt 6880, w8a4a, nur beschreibend, kein Test): 0/1/2,5/5 % ergeben 53,64/54,35/54,80/55,50 %. 1 % gegen 0 % ergibt +0,85 pp (0/12), 1 % gegen 5 % −0,86 pp (11/1). Der Befund ist also kein Effekt der Stopp-Wahl.

## Vorab-Erwartungen gegen Ergebnis

| Größe | Erwartung | Ergebnis |
|---|---|---|
| Kontrolle 2/2 bitgleich | 0,9 | ja |
| P1 „besser“ | 0,4 | ja |
| P2 „schlechter“ | 0,3 | ja |
| Gesamt „kurzer Anlauf vermeidet den 4-Bit-Preis“ | 0,35 | nein |
| S3 „besser“ | 0,3 | nein (unklar) |
| Rangkorrelation im Median > 0 | 0,7 | ja (1,0) |

## Deutung nach der Vorab-Regel

- **Regel „auch 1 % kostet“:** Der Preis hängt am Anlauf selbst, nicht an seiner Länge. Für 4 Bit bei d200 sollte man ohne Anlauf trainieren oder den Preis offen nennen.
- **Was dazukommt:** Der Preis ist nicht fest, er wächst mit der Länge. Ein kurzer Anlauf kostet weniger als ein langer (P1, S1).
- **Praxisregel für BPreCap bei d200 und LR 5e-5:** Für 4 Bit ohne Anlauf trainieren. Bei dieser LR ist der Anlauf für die Stabilität nicht nötig, B15 trainiert ohne Anlauf normal. Der fp32-Gewinn durch den Anlauf ist klein (höchstens etwa 0,4 pp bei 5 %).
- **Wann ein Anlauf trotzdem nötig ist:** bei größeren LR (d100 mit 2e-4, V1). Dort kostet er wahrscheinlich ebenfalls bei 4 Bit, siehe V1. Bei d100 ist das nicht geprüft.

## Einordnung

- **Für v0.3.1:** Das ist eine ehrliche Präzisierung der Anlauf-Empfehlung, kein neuer Anspruch.
  - Der Anlauf behebt die LR-Grenze. Wer auf 4 Bit zielt und keine große LR braucht, trainiert besser ohne.
  - Man kann es in einem Satz mit Zahlen sagen: 0/1/2,5/5 % ergeben 53,6/54,3/54,9/55,3 % w8a4a.
- **Für v0.4 und V2a:** V2a sollte ohne Anlauf trainieren, wenn die LR das erlaubt. Sonst sollte es den kürzesten stabilen Anlauf wählen und das als Störgröße festhalten.
- **Mögliche Ursache (Hypothese, nicht geprüft):**
  - Mit Anlauf verteilen sich die Werte enger und nutzen den weichen Bereich weniger (D3b, sat_rate).
  - Das Gitter der 4-Bit-Kalibrierung trifft die schmalere Verteilung dann schlechter.
  - Prüfen ließe sich das mit einer Kalibrierung je Schicht statt je Tensor oder mit einer größeren Hülle (M), auf dem neuen Mac.

## Grenzen

- **Rahmen:** ein Datensatz (CIFAR-10), Tiefe 200, eine LR/WD (5e-5/0,3), nur MLPs ohne Normierung, nur BPreCap.
- **Referenzen:** 0 % und 5 % stammen aus B15 und V1c. Die Bitgleichheit ist nur für Seed 42192 geprüft.
- **Größe der Effekte:** Sie sind klein (0,6–1,0 pp), aber einheitlich (0/12).

Sicherung: `../../sicherung/v1d_2026-11-03/`.
