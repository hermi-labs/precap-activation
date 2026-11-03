# V1d – Präregistrierung: Vermeidet ein kürzerer LR-Anlauf BPreCaps 4-Bit-Preis bei Tiefe 200? (privat)

Stand: 03.11.2026, nachmittags, vor dem Start. Die Freeze-Zeit steht in `meta/frozen_at.txt`. PreCap und BPreCap sind Aarons Ideen. Das Design ist mit Claude entworfen und wird in Aarons Abwesenheit gestartet (Leitlinie fürs Alleinarbeiten, Freigabe vom 03.11. bis 18:30).

## Anlass

- **V1c S3:** Mit 5 % Anlauf (344 Schritte) ist BPreCap bei CIFAR-10 d200 in fp32 0,59 pp besser, bei 4 Bit (w8a4a, frühes Stoppen) aber 1,64 pp schlechter als ohne Anlauf (B15), 0/12.
- **V1 bei d100:** dieselbe Richtung, dort nicht signifikant.
- **D3b (2 Seeds, beschreibend):** Mit Anlauf liegen 25–30 % weniger Werte im weichen Bereich der Hülle.
- **Praktische Frage für v0.3.1 und v0.4:** Muss man beim Anlauf wählen zwischen fp32 und 4 Bit, oder reicht ein kurzer Anlauf ohne 4-Bit-Preis?

## Frage

Ist BPreCap bei d200 mit 1 % Anlauf (69 Schritte) bei 4 Bit besser als mit 5 % (V1c), und ist es dabei nicht schlechter als ohne Anlauf (B15)?

## Eine Änderung: Länge des Anlaufs

- **Code wie V1c** (`v1c_run.py`). Die einzige Änderung ist die Anlauflänge je Lauf (`warm`). Danach folgt wie bisher ein Kosinus auf 0 bis Schritt 6880.
- **Konstant:**
  - nur kL2B_tr;
  - CIFAR-10, Tiefe 200, Breite 256, Batch 128, AdamW;
  - LR 5e-5 und WD 0,3. Das war in B15 und V1c die gewählte Kombination. Es gibt kein Tuning, damit nur der Anlauf variiert;
  - Zwischenmessungen alle 860 Schritte, Quantisierung und Kalibrierung wie V1c.
- **Varianten:** Anlauf 69 Schritte (1 %) und 172 Schritte (2,5 %).
- **Referenzen** (dieselben Seeds, gepaart):
  - 0 % aus der B15-Bestätigung;
  - 5 % aus der V1c-Bestätigung.
- **Seeds:** 42192–42203 (12).
- **Kontrolle:** Seed 42192 ohne Anlauf (cos) und mit Anlauf 344 neu gerechnet. Alle Mess- und Endwerte müssen bitgleich zur V1c-Kontrolle bzw. zur V1c-Zeile sein.
- **Läufe:** 2 + 24 = 26, geschätzt 70–80 min auf 2 Workern (M1 Pro), nur am Netzteil.

## Primärtests (Holm über 2, α = 0,05, Wilcoxon zweiseitig, gepaart je Seed)

Messgröße: test_err_w8a4a_best am Punkt des frühen Stoppens (kleinster Val-Fehler fp32, wie V1c). Ein fehlender Lauf zählt als 90 %.

| Nr. | Vergleich | Hypothese |
|---|---|---|
| P1 | Anlauf 1 % − Anlauf 5 % | besser |
| P2 | Anlauf 1 % − kein Anlauf | nicht schlechter (geprüft wird nur „schlechter“) |

**Gesamturteil (in dieser Reihenfolge):**
1. P2 „schlechter“: **auch 1 % Anlauf kostet bei 4 Bit**.
2. Sonst P1 „besser“: **kurzer Anlauf vermeidet den 4-Bit-Preis** (Erfolg).
3. Sonst P1 „schlechter“: 1 % Anlauf ist bei 4 Bit schlechter als 5 %.
4. Sonst: **nicht aufgelöst**.

Ein „unklar“ bei P2 heißt nicht „gleich“. Es heißt nur, dass bei 12 Seeds kein Unterschied gezeigt wurde. So wird es im Bericht formuliert.

## Sekundär (Holm über 5)

| Nr. | Vergleich | Größe |
|---|---|---|
| S1 | 2,5 % − 5 % | w8a4a, frühes Stoppen |
| S2 | 2,5 % − 0 % | w8a4a, frühes Stoppen |
| S3 | 1 % − 0 % | fp32, frühes Stoppen (bleibt der fp32-Gewinn?) |
| S4 | 1 % − 5 % | fp32, frühes Stoppen |
| S5 | 2,5 % − 0 % | fp32, frühes Stoppen |

**Beschreibend:**
- je Anlauflänge die Mediane von fp32, w8a4a, w8a4 und w8a3 am Stopp-Punkt;
- der 4-Bit-Preis (w8a4a − fp32);
- der Sättigungsanteil (`sat_rate`) und die Stopp-Punkte;
- je Seed die Rangkorrelation Anlauflänge → w8a4a über 0/1/2,5/5 % (Median).

## Deutungsregeln (vorab)

| Ergebnis | Deutung |
|---|---|
| Erfolg und S3 „besser“ | Ein kurzer Anlauf holt den fp32-Gewinn ohne 4-Bit-Preis. Empfehlung für BPreCap: 1 % |
| Erfolg und S3 nicht „besser“ | Ein kurzer Anlauf vermeidet den Preis, bringt in fp32 aber nichts Nachweisbares. Empfehlung: kurzer oder kein Anlauf |
| auch 1 % kostet | Der Preis hängt am Anlauf selbst, nicht an seiner Länge. Für 4 Bit bei d200 ohne Anlauf trainieren oder den Preis offen nennen |
| nicht aufgelöst | Bei 12 Seeds nicht entscheidbar. V1c S3 bleibt der Befund, ohne Empfehlung zur Länge |
| Kontrolle nicht bitgleich | Abweichung suchen und berichten. Die Vergleiche gegen B15 und V1c gelten dann nicht, nur 1 % gegen 2,5 % innerhalb von V1d |

## Vorab-Erwartung

| Größe | Erwartung |
|---|---|
| Kontrolle 2/2 bitgleich | 0,9 |
| P1 „besser“ | 0,4 |
| P2 „schlechter“ | 0,3 |
| Gesamt „kurzer Anlauf vermeidet den 4-Bit-Preis“ | 0,35 |
| S3 „besser“ | 0,3 |
| Rangkorrelation im Median > 0 | 0,7 |

## Grenzen

- Ein Datensatz (CIFAR-10), eine Tiefe (200), eine LR/WD, nur MLPs ohne Normierung.
- Die Referenzen 0 % und 5 % stammen aus B15 und V1c. Die Bitgleichheit ist nur für Seed 42192 geprüft, für die übrigen Seeds folgt sie aus demselben Code und denselben Konstanten.
- Mit 12 Seeds sind Unterschiede unter etwa 0,5 pp kaum nachweisbar.
- SELU läuft nicht mit. Die Studie sagt nur etwas über BPreCap.

## Freeze und Tests

- **Eingefroren:**
  - `v1d_run.py`, `v1d_analyze.py`, beide Tests, diese Datei, `src/precap/*.py`, CIFAR-10-Dateien;
  - `../v1c/v1c_run.py` und `../b15/b15_run.py` werden gegen ihren Freeze geprüft.
- **Selbsttest:** 26 Läufe, Anlauf 69/172, HP 5e-5/0,3.
- **Mechanik** (`test_v1d_mechanik.py`):
  - Die Laufliste stimmt.
  - Der LR-Plan je Länge hat 0 am Start, die Spitze am Ende des Anlaufs und fällt danach auf 0.
  - Kurzläufe mit 40 Schritten sind bitgleich zu V1c, für cos und für den Anlauf-Pfad.
- **Auswertung an künstlichen Daten** (`test_v1d_analyse.py`):
  - lineare Dosis-Wirkung;
  - nur 5 % kostet;
  - jeder Anlauf kostet;
  - Null (10×);
  - Kontrolle verfehlt;
  - Ausfall;
  - fp32-Gewinn.
- **Rauchtest:** im Startskript, 40 Schritte, Anlauf 5/10, 3 Zeilen ok.
