# V2a-1 – Präregistrierung: Ist BPreCap im CNN ohne Normierung bei 4 Bit besser als SELU? (privat, M1 Pro)

Stand: 04.11.2026, 13:35, vor dem Start. PreCap, die Tiefenfamilie und die Hülle sind Aarons Ideen. Das Design habe ich mit Claude entworfen. Aaron hat am 04.11. mittags „V2a gegen SELU“ freigegeben und mir den Mac bis 18:45 überlassen. Grundlage sind die Piloten in `studien/v2a_entwurf/V2a_Entwurf.md` (Pilot 1–4). Die große V2a-Studie (L = 32 und 50, AdamW, Kosinus, Tuning) bleibt für den neuen Mac. V2a-1 ist die kleine, auf dem M1 Pro machbare Stufe davor.

## Frage

Ist BPreCap in einem CNN mit L = 32 Faltungsschichten auf CIFAR-10 bei 4 Bit (PTQ, w8a4a_max wie SCULPT) besser als SELU? Das Netz hat keine Normierung, keine Residuen und eine Delta-orthogonale Init.

## Aufbau (exakt wie Pilot 4, Phase A)

- **Netz:** Faltung 3×3, 3 → 32, dann 31 × (32 → 32), volle Auflösung, globales Mittel, Linear 10. Delta-orthogonale Init, keine Normierung, keine Residuen.
- **Formen:**
  - BPreCap mit den Konstanten für L = 32 (`kL2B_d32.json`: λ 1,2292, α 1,9526, b 0,6419, χ1 = 1,0625, M = 3);
  - SELU.
- **Daten:** CIFAR-10, Standardisierung über den Trainingssatz, die letzten 5000 Trainingsbilder als Val, Test-Satz für die Tests.
- **Training:** Adam mit konstanter LR, 1720 Schritte, Batch 128.
  - **LR je Form aus den Piloten:** BPreCap 1e-4, SELU 3e-5. Gewählt am Val-Satz mit Seed 42000 (Pilot 1–3). Das ist eine Wahl auf wenigen Daten, sie wird hier offen genannt.
- **Seeds:** 42330–42341 (12, neu, per Suche in `studien/` geprüft).

## Quantisierung (PTQ, simuliert)

- **Gewichte:** int8 je Ausgangskanal, symmetrisch.
- **Aktivierungen:** Eingang jeder Faltung und der Linearschicht, je Tensor.
  - w8a8 symmetrisch (max);
  - w8a4a_max asymmetrisch (min/max);
  - w8a4a_p999 und w8a4a_p9999 (Quantile);
  - w8a4a_best: die Kalibrierung mit dem kleinsten Fehler auf den Kalibrierbildern.
- **Kalibrierung:** die ersten 1000 Trainingsbilder, Quantile auf einer festen Stichprobe von 2 Mio. Werten je Tensor.

## Primärtest

**BPreCap − SELU, Test-Fehler w8a4a_max.** Wilcoxon zweiseitig, gepaart über 12 Seeds, α = 0,05. Ein Ausfall oder eine fehlende Zeile zählt als 90 %.

**Erfolg:** p < 0,05 und Median < 0 („besser“).

## Sekundär (Holm über 3)

| Nr. | Vergleich BPreCap − SELU |
|---|---|
| S1 | w8a4a_best |
| S2 | fp32 |
| S3 | w8a8 |

**Beschreibend:**
- Bereich je Form (Median über die Schichten, unten und oben);
- 4-Bit-Preis;
- gewählte Kalibrierung;
- Val-Fehler.

## Deutungsregeln (vorab)

| Ergebnis | Deutung |
|---|---|
| Erfolg | Der 4-Bit-Vorteil gegen SELU trägt vom MLP (B12, B15, V1c) ins CNN ohne Normierung, bei L = 32 und mit einfachem Training |
| Erfolg und S2 „schlechter“ | Der Vorteil kostet fp32-Güte, der Preis wird genannt |
| Primär „unklar“ | Im CNN ist SELU bei 4 Bit nicht klar schlechter. Ehrlich festhalten. Die große V2a-Studie muss das mit Tuning klären |
| Primär „schlechter“ | SELU ist im CNN bei 4 Bit vorn. Ins Preprint als negatives Ergebnis |

## Vorab-Erwartung

| Größe | Erwartung |
|---|---|
| Primär „besser“ (Erfolg) | 0,75 |
| Primär „unklar“ | 0,2 |
| Primär „schlechter“ | 0,05 |
| S1 „besser“ | 0,65 |
| S2 „unklar“ | 0,6 |
| S2 „besser“ | 0,25 |
| S3 „unklar“ | 0,8 |

Die Erwartung für den Primärtest liegt hier höher als für die große V2a-Studie (0,55). Der Grund: Hier gibt es nur L = 32, und dort war der Pilot-Abstand groß (−8 bis −13 pp). Außerdem fehlen WD und Kosinus, die den SELU-Bereich bremsen könnten.

## Grenzen

- **Ein Netz:** L = 32, 32 Kanäle, kein Pooling, keine Augmentierung, nur CIFAR-10, nur PTQ.
- **Einfaches Regime:** Adam mit konstanter LR, 1720 Schritte, kein WD, kein Tuning in der Studie. Die LR ist aus einem Pilot-Seed übernommen.
- **TAT ist nicht dabei.** Gegen TAT ist der Abstand nach Pilot 4 (−44 pp) so groß, dass ein Test nichts prüfen würde.
- **Kontrolle nur über 20 Schritte:** Die Gleichheit mit dem Pilot-Code ist bei 20 Schritten für alle Messgrößen geprüft. Eine Kontrolle über die volle Länge (1720 Schritte gegen die Pilot-Werte) läuft aus Zeitgründen nicht vor dem Start.

## Freeze und Tests

Eingefroren werden: `v2a1_run.py`, `v2a1_analyze.py`, `test_v2a1_mechanik.py`, `test_v2a1_analyse.py`, `kL2B_d32.json`, diese Datei, `launch_v2a1.sh`, `../v2a_entwurf/v2a_pilot4.py`, `v2a_pilot.py`, `v2a_probe.py` (Vergleichscode), `v2a_pilot4_A_M1Pro.json` und die CIFAR-Dateien.

Vorab gemachte Tests (04.11., 13:20–13:33):
- **Selbsttest:** E f² = 1,0000 für BPreCap, Delta-Init orthogonal, 24 Läufe, Seeds neu, beide Worker rechnen beide Formen.
- **Mechanik:** MECHANIK OK.
  - Die Daten sind gleich wie im Pilot.
  - Bei 20 Schritten sind fp32, w8a8, w8a4a_max und w8a4a_p9999 für beide Formen bitgleich zum Pilot-Code.
  - p999 ändert die übrigen Kalibrierwerte nicht.
- **Auswertung an künstlichen Daten:** AUSWERTUNG OK. Erfolg und „schlechter“ werden erkannt, bei Null gibt es 0 von 20 Fehlalarmen, Ausfall und fehlende Zeilen zählen 90 %.
- **Rauchtest:** im Startskript, 20 Schritte, 4 Zeilen.
