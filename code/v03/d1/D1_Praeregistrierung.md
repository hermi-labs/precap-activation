# D1 – Präregistrierung: Diagnose, warum TAT bei 4 Bit mehr zahlt als BPreCap (CIFAR-10, d100, privat, beschreibend)

Stand: 31.10.2026, nachts, vor dem Start. PreCap, die Tiefenfamilie und Lagma sind Aarons Ideen. Das Design habe ich mit Claude entworfen. Aaron hat Zwischentests zwischen den B-Studien am 31.10. erlaubt, wenn sie exakt begründet sind. Nichts veröffentlichen.

## Warum dieser Test (Begründung)

- **B10b** hat die bisherige Deutung widerlegt: TAT mit Gewichtsabfall hat denselben oder einen kleineren Bereich als BPreCap (3,1 bzw. 0,36 gegen 3,0). Trotzdem zahlt es bei 4 Bit +7,3 bis +7,8 pp, BPreCap nur +3,0.
- Leaky ReLU ist positiv homogen, die Quantisierung je Tensor ist skalierungsinvariant. Der absolute Bereich kann es also nicht sein.
- Für v0.3 brauchen wir eine ehrliche Aussage, **warum** die feste Hülle bei 4 Bit hilft. Sonst bleibt nur die Beobachtung. Zwei Erklärungen sind offen:
  - **E1 (Form):** Die beschränkte, teils gesättigte Verteilung wird je Schicht genauer quantisiert (höheres Signal-Rausch-Verhältnis je Schicht).
  - **E2 (Fortpflanzung):** Der Rundungsfehler einer Schicht wird bei BPreCap über die folgenden Schichten gedämpft (gesättigte Einheiten haben eine Ableitung nahe 0). Bei TAT wird er weitergereicht.
- D1 misst beides direkt an trainierten Netzen. Es gibt **keinen Hypothesentest**, nur Beschreibung mit vorab festgelegten Kennzahlen und Lesarten. n = 2 Seeds.

## Netze (eine Änderung: keine, nur Nachtraining und Messung)

Nachtraining exakt wie in B10 bzw. B10b, CIFAR-10, d100, AdamW, Kosinus, 6880 Schritte, Seeds **42168 und 42169**:

| Name | Form | LR / WD | Quelle der Vergleichszeile |
|---|---|---|---|
| bp | kL2B_tr | 1e-4 / 0,3 | B10 |
| lb | LB_tr | 1e-4 / 0,3 | B10 |
| ta | TAT | 1e-4 / 0,3 | B10b (TAT a) |
| t0 | TAT | 2e-4 / 0 | B10 |

- Code: `studien/b10/b10_run.py`, **unverändert** als Modul geladen (eingefroren in B10). Eigener Code nur für die Trainingsschleife ohne Zwischenmessungen und für die Messungen.
- **Prüfung der Nachbildung (Pflicht):** Am Ende müssen `val_err_fp32`, `test_err_fp32` und `test_err_w8a4a_best` (mit derselben Wahlregel) **bitgleich** zu den Zeilen in B10 bzw. B10b sein. Sonst gilt der Lauf als ungültig und wird nicht ausgewertet.
- Gemessen wird am **Ende** (Schritt 6880), weil nur dort die Prüfung möglich ist. Die Studien selbst werten am Stopp-Punkt aus; das ist eine Grenze.

## Kennzahlen (je Netz)

Auf den 1000 Kalibrierbildern bzw. auf 2000 Testbildern (die ersten 2000), asymmetrische 4-Bit-Quantisierung der Eingänge der Linearschichten mit der je Lauf gewählten Kalibrierung (Wahlregel wie B10):

1. **Form je Schicht** (Eingänge der Schichten 2–100): Kurtosis, Max/99,99-%-Quantil, Anteil der Werte über 95 % des Betragsmaximums.
2. **Lokaler Rundungsfehler je Schicht:** relatives MSE ε_k = E[(q(a_k) − a_k)²] / E[a_k²] (fp32-Eingänge, nur diese Schicht gerundet). Median über die Schichten. → misst E1.
3. **Fortpflanzung:** Für k ∈ {2, 10, 25, 50, 75, 99} wird **nur** der Eingang von Schicht k gerundet, alle anderen bleiben fp32. Gemessen:
   - relativer Fehler der Logits r_k = ‖Δlogits‖ / ‖logits‖ (Frobenius über 2000 Testbilder),
   - Verstärkung G_k = r_k / √ε_k,
   - Anteil geänderter Vorhersagen.
   → misst E2.
4. **Rausch-Verstärkung:** Gauß-Rauschen mit Standardabweichung 0,01 · RMS(a_k) am Eingang von Schicht k (fester Seed), relative Abweichung der Logits geteilt durch 0,01. Gleiche k. → zweite Messung von E2, unabhängig von der Rundung.
5. **Alle Schichten gerundet:** r_alle und Testfehler (Kontrolle gegen `test_err_w8a4a_best` auf denselben 2000 Bildern).

## Lesart (vorab)

| Ergebnis | Lesart |
|---|---|
| ε_k (Median) bei bp und lb mindestens 2-mal kleiner als bei ta, G_k ähnlich (Faktor < 1,5) | E1 trägt: die Form der Verteilung |
| ε_k ähnlich (Faktor < 1,5), G_k bei bp und lb mindestens 2-mal kleiner | E2 trägt: Dämpfung über die Tiefe |
| beide Faktoren ≥ 2 | beide tragen |
| keiner ≥ 1,5 | keine der beiden Erklärungen. Offen berichten |

Die Faktoren werden je Seed berechnet und nur dann als Befund gelesen, wenn beide Seeds dieselbe Lesart ergeben. Sonst „uneinheitlich“.

## Vorab-Erwartung

| Größe | Erwartung |
|---|---|
| Nachbildung bitgleich (alle 8 Läufe) | 0,85 |
| E1 trägt (allein oder mit E2) | 0,5 |
| E2 trägt (allein oder mit E1) | 0,5 |
| keine der beiden | 0,2 |

## Grenzen

- Beschreibend, n = 2 Seeds, keine Statistik.
- Gemessen am Ende, nicht am Stopp-Punkt.
- Nur CIFAR-10, d100, asymmetrisch 4 Bit.
- G_k mischt Fortpflanzung und die Empfindlichkeit der Logits. Die Rausch-Verstärkung (4) dient als Gegenprobe.

## Ablauf

1. Selbsttest `--selfcheck`: Quantisierungs-SNR an synthetischen Gleich- und Normalverteilungen gegen den theoretischen Wert. Messfunktionen an einem untrainierten Netz.
2. Freeze: `d1_diag.py`, diese Datei, `studien/b10/b10_run.py` (Hash muss dem B10-Freeze entsprechen).
3. Start erst nach dem Ende von B14 (GPU frei), 2 Worker in `screen` mit `caffeinate -i`, Logs in `logs/`.

## Vorab gemachte Tests (31.10., 23:45)

- **Selbsttest** (`d1_diag.py --selfcheck`): ok. Rundungsfehler der Gleichverteilung 0,004441 gegen theoretisch 0,004444. `fwd_mod` mit Identität ist bitgleich zum normalen Vorwärts. 101 Schichteingänge, 8 Läufe, alle Vergleichszeilen in B10/B10b gefunden.
- **Rauchtest der Messung** (`test/rauch_messung.py`, untrainierte Netze, echte CIFAR-10-Daten): alle Kennzahlen endlich, 6–9 s je Netz. Rundungs- und Rausch-Verstärkung stimmen überein (z. B. BPreCap Schicht 2: 2,49 gegen 2,76; TAT: 0,84 gegen 0,96).
- **Auswertung an künstlichen Daten:** E1, E2, beide, keine, uneinheitlich und ungültig werden richtig erkannt (AUSWERTUNGSTEST OK).
- **Auswertung:** `python3 -I d1_auswerten.py`, eingefroren mit dem Start.
- **Start:** automatisch nach dem Ende von B14 über `launch_d1.sh` (prüft Netzteil, andere Worker, Hash von `b10_run.py` gegen den B10-Freeze).

## Änderung vor dem Freeze (01.11., 02:00): SELU aufgenommen

- **Was:** zusätzlich **se** = SELU_tr (LR 2e-4, WD 0,3) aus B12, Seeds 42168 und 42169. Nachtraining mit `studien/b12/b12_run.py` (unverändert geladen, Hash gegen den B12-Freeze geprüft). Bitgleich-Prüfung gegen die B12-Zeilen wie oben. Zusammen 10 Läufe, etwa 25–30 min.
- **Warum:** B12 (fertig 01:58) zeigt, dass SELU im gleichen Rezept bei 4 Bit nur +2,5 pp zahlt, obwohl es oben unbeschränkt ist. Zusammen mit B10b spricht das gegen „die obere Schranke ist der Grund“. SELU in D1 trennt die Frage „Schranke“ von der Frage „Form der Funktion (glatt/sättigend gegen stückweise linear)“.
- **Offen genannt:** Diese Erweiterung ist nach dem Ergebnis von B12 entschieden. Die Vergleiche mit SELU sind deshalb **nur beschreibend**. Die ursprüngliche Lesart (ta gegen bp und lb) bleibt unverändert die vorab festgelegte.
- **Beschreibend zusätzlich:** Faktoren ta gegen se mit derselben Rechnung.
- Selbsttest nach der Änderung: ok (10 Läufe, alle Vergleichszeilen gefunden).

## Zweite Änderung vor dem Freeze (01.11., 03:50): harter linearer Clip aufgenommen

- **Was:** zusätzlich **lh** = LBh_tr (LR 5e-5, WD 0,3) aus B13, Seeds 42168 und 42169, Nachtraining mit `studien/b13/b13_run.py` (unverändert, Hash gegen den B13-Freeze geprüft). Zusammen 12 Läufe, etwa 30–35 min.
- **Warum:** B13 zeigt beim linearen Kern einen großen Unterschied zwischen weicher und harter Hülle (4-Bit-Preis +2,3 gegen +7,7 pp) bei gleicher fp32-Güte. **lh gegen lb** ist der sauberste Vergleich für E1/E2: gleicher Kern, gleiche Kritikalität, nur die Form der Grenze unterscheidet sich.
- **Offen genannt:** nach dem Ergebnis von B13 entschieden, deshalb **nur beschreibend**. Die vorab festgelegte Lesart (ta gegen bp und lb) bleibt unverändert.
