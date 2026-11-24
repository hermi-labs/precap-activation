# V2a2 (V2a groß) – Präregistrierung: Ist BPreCap im tiefen CNN ohne Normierung bei 4 Bit besser als SELU und als TAT?

Stand: 16.11.2026, vor dem Start (privat, M1 Pro). PreCap, die Tiefenfamilie und die Hülle sind Aarons Ideen. Das Design habe ich mit Claude entworfen. Grundlage sind der Entwurf `../v2a_entwurf/V2a_Praeregistrierung_Entwurf.md` mit Pilot 1–4 und das Ergebnis von V2a-1 (`../v2a1/V2a1_Auswertung.md`).

**Freigaben Aarons:**
- 15.11., 19:52: „okay passt, so machen wir das“. Damit freigegeben: V2a groß auf dem M1 Pro, 3440 Schritte, Raumnetz pausiert.
- 16.11. morgens: beide Vergleiche als Primärtests („wäre nicht gegen beides testen sinnvoll?“).

Eingefroren wird mit dem Start (`meta/frozen_sha256.txt`).

## Frage

Ist BPreCap in einem CNN mit L = 50 Faltungsschichten auf CIFAR-10 bei 4-Bit-Aktivierungen (PTQ, jede Form mit ihrer besten Kalibrierung, w8a4a_best) besser als SELU und besser als TAT?
- Das Netz hat keine Normierung, keine Residuen und eine Delta-orthogonale Init.
- Alle Formen trainieren mit AdamW und Kosinus-Plan, mit je Form getunter LR und WD.

## Änderungen gegenüber V2a-1

V2a-1 war die kleine Vorstufe: L = 32, nur gegen SELU, Adam mit konstanter LR, 1720 Schritte, kein Tuning. Ergebnis: −7,91 pp bei w8a4a_max (12/0) und −6,17 pp bei w8a4a_best.

V2a2 ist die im Entwurf geplante große Studie. Sie ändert gezielt das, was V2a-1 offen ließ:

| Punkt | V2a-1 | V2a2 |
|---|---|---|
| Tiefe | L = 32 | L = 50 |
| Formen | BPreCap, SELU | BPreCap, SELU, TAT |
| Regime | Adam, konstante LR, 1720 Schritte | AdamW mit entkoppeltem WD, Kosinus auf 0 ohne Anlauf, 3440 Schritte (wie B10) |
| LR/WD | aus einem Pilot-Seed | je Form getunt (2 Tuning-Seeds) |
| Messpunkt | Ende | frühes Stoppen am kleinsten Val-Fehler, dazu Ende (beschreibend) |
| Kalibrierer | max, p999, p9999 | max, p999, p9999, p99999, mse4 |
| TAT mit CLE | – | zusätzlich gemessen (Sekundärtest S5) |

## Abweichungen vom Entwurf (vor dem Freeze entschieden, nicht nach Ergebnissen)

- **Nur L = 50, keine Zelle L = 32.** L = 32 deckt V2a-1 ab. Zwei Zellen würden die Laufzeit auf dem M1 Pro verdoppeln (etwa 60 h).
- **Kein tanh als vierte Form.** Das kostet etwa 8 h mehr. Die Frage „Grenze oder Tiefenkalibrierung“ bleibt für eine eigene Studie (V2c). Das ist unten als Grenze genannt.
- **Kein ACIQ.** Den MSE-optimalen Bereich deckt mse4 ab, und zwar asymmetrisch, mit Gitter 1–100 % von (min, max) und dem kleinsten MSE auf der Stichprobe.
- **Keine Normierung je Beispiel.** Das ist gleich wie in Pilot 1–4 und V2a-1, damit die Kontrolle gegen den Pilot-Code möglich bleibt. Es gilt für alle Formen gleich.
- **TAT zusätzlich mit Cross-Layer Equalization (CLE)**, nachgetragen nach der Recherche vom 16.11. (`docs/literatur/Recherche_2026-11-16_tief.md`):
  - CLE stammt aus Nagel et al. 2019, DFQ ([arXiv 1906.04721](https://arxiv.org/abs/1906.04721)). Es setzt f(sx) = s·f(x) für s > 0 voraus. TATs Leaky ReLU erfüllt das, SELU und BPreCap nicht.
  - Ohne CLE wäre der Vergleich mit TAT zu unseren Gunsten verzerrt.
  - Deshalb gibt es den Sekundärtest S5 gegen TAT nach CLE. Die Primärtests bleiben wie mit Aaron besprochen.
- **Primär w8a4a_best statt w8a4a_max.** Mit TAT als Primärvergleich wäre min/max ein Test mit feststehendem Ausgang: Im Pilot liegt TAT bei w8a4a_max bei 89 %. Die beste Kalibrierung je Form ist der strengere und fairere Vergleich. w8a4a_max wird Sekundärtest.

## Aufbau

- **Netz:**
  - Faltung 3×3 (3 → 32), dann 49 × (32 → 32), volle Auflösung 32×32, Padding 1.
  - Globales Mittel, danach Linear 32 → 10.
  - Delta-orthogonale Init: Nur der mittlere Abgriff ist orthogonal, die Biases sind 0. Linear-Init Gauß mit 1/√32.
  - Für alle Formen gleich. Code wie Pilot 4 und V2a-1.
- **Formen:**
  - **BPreCap:** Konstanten `kL2B_d50.json` (λ 1,7524, α 1,6813, b 0,4647, χ1 = 1,04, Hülle M = 3, ρ = 0,8).
  - **TAT:** Leaky ReLU mit `v2a_tat_params.json`, Schlüssel 50 (Steigung negativ 0,4305, Skala 1,2989). E f² = 1 ist im Selbsttest geprüft.
  - **SELU.**
- **Daten:**
  - CIFAR-10, Standardisierung über den Trainingssatz (ein Mittelwert, eine Standardabweichung).
  - Die letzten 5000 Trainingsbilder sind der Val-Satz, der Test-Satz (10 000 Bilder) dient den Tests.
  - Keine Augmentierung.
- **Training:**
  - Batch 128, 3440 Schritte, AdamW (MLX: w ← w(1 − lr·wd)), Kosinus von der LR-Spitze auf 0 ohne Anlauf.
  - Datenreihenfolge aus `default_rng(seed + 777)`.
- **Frühes Stoppen:**
  - 8 Messpunkte (alle 430 Schritte), Val-Fehler fp32.
  - Gewählt wird der Punkt mit dem kleinsten Val-Fehler, bei Gleichstand der frühere.
  - Die Tests nutzen das Netz an diesem Punkt. Das Ende wird zusätzlich gemessen.

## Quantisierung (PTQ, simuliert, wie V2a-1)

- **Gewichte:** int8 je Ausgangskanal, symmetrisch.
- **Aktivierungen:** Eingang jeder Faltung und der Linearschicht, je Tensor.
  - w8a8: symmetrisch (max).
  - w8a4a_{max, p999, p9999, p99999, mse4}: asymmetrisch.
  - w8a4a_best: die Kalibrierung mit dem kleinsten Fehler auf den Kalibrierbildern. Bei Gleichstand gilt die Reihenfolge max, p999, p9999, p99999, mse4.
- **Kalibrierung:**
  - die ersten 1000 Trainingsbilder;
  - Quantile und mse4 auf einer festen Stichprobe von 2 Mio. Werten je Tensor, min/max auf allen Werten.

## Tuning

- **Raster je Form**, um die beste LR aus Pilot 4 (L = 50) gelegt:
  - BPreCap: LR {1e-5, 3e-5, 1e-4}
  - SELU: LR {1e-5, 3e-5, 1e-4}
  - TAT: LR {3e-5, 1e-4, 3e-4}
  - WD für alle {0; 3,0}
- **Seeds:** 42000 und 42001, die Pilot-Seeds, nur fürs Tuning.
- **Umfang:** 36 Läufe.
- **Kriterium wie B10:**
  - die höchste mediane Val-Genauigkeit (fp32) am Ende über beide Seeds;
  - ein Ausfall zählt 0,1;
  - bei Gleichstand gilt die kleinere LR, dann der kleinere WD.
  - Die Wahl erfolgt automatisch im Code (`tune/hp_sel.json`).
- **Anlauf-Regel:** Bricht eine Form bei jeder Einstellung ein, stoppt die Studie vor der Bestätigung. Dann folgt eine neue Version mit linearem Anlauf über 5 % für alle Formen.

## Bestätigung

- **Umfang:** 3 Formen × 12 Seeds, 42350–42361, also 36 Läufe.
- **Seeds:** neu, per Suche in `studien/` geprüft. Vorgesehen seit dem Entwurf vom 04.11.
- **Reihenfolge:** je Seed alle drei Formen, die Reihenfolge rotiert.
- **Soll:** 36 Tuning-Zeilen und 36 Bestätigungszeilen. Ein Worker, ein Gerät (M1 Pro).

## Primärtests (Holm über 2)

- **P1:** BPreCap − SELU, Test-Fehler w8a4a_best am Punkt des frühen Stoppens (pp).
- **P2:** BPreCap − TAT, ebenso.

Wilcoxon-Vorzeichen-Rang-Test, gepaart über 12 Seeds, zweiseitig. Ein Ausfall oder eine fehlende Zeile zählt 90 %.

**Urteil je Test:**
- Holm-p < 0,05 und Median < 0: „besser“.
- Holm-p < 0,05 und Median > 0: „schlechter“.
- Sonst: „unklar“.

**Gesamt:**
- „Erfolg“, wenn beide Tests „besser“ ergeben.
- „Teilweise“, wenn einer „besser“ und der andere „unklar“ ergibt.
- Sonst nach den Einzelurteilen.

## Sekundärtests (Holm über 5, frühes Stoppen)

| Nr. | Vergleich |
|---|---|
| S1 | BPreCap − SELU, fp32 |
| S2 | BPreCap − TAT, fp32 |
| S3 | BPreCap − SELU, w8a4a_max |
| S4 | BPreCap − TAT, w8a4a_max |
| S5 | BPreCap w8a4a_best − TAT nach CLE, w8a4a_best |

**CLE (nur TAT, nach dem Training, am Punkt des frühen Stoppens):**
- Jedes Paar aufeinanderfolgender Schichten wird angeglichen: Faltung k mit Faltung k + 1, die letzte Faltung mit Linear nach dem globalen Mittel.
- Skalierung je Kanal i: s_i = √(r1_i / r2_i). Ausgangskanal i (Gewicht und Bias) wird durch s_i geteilt, Eingangskanal i der nächsten Schicht mit s_i multipliziert.
- 30 Durchgänge.
- In fp32 ist die Funktion gleich; das prüft der Mechanik-Test.
- Danach folgen dieselbe Kalibrierung und dieselbe Wahl w8a4a_best wie für alle Formen.
- Kein Bias-Ausgleich und keine Gewichts-Rundungsoptimierung (AdaRound): Das wären weitere PTQ-Stufen, die BPreCap und SELU ebenso zustünden.

**Beschreibend (ohne Test):**
- alle Messgrößen am Ende;
- 4-Bit-Preis je Form (w8a4a_best − fp32);
- Wahl der Kalibrierung;
- Bereich der Aktivierungen (unten, oben, |max|), am frühen Stopp und am Ende;
- gewählte LR/WD und ob sie am Rand liegen;
- Punkt des frühen Stoppens;
- Val-Verlauf;
- Ausfälle.

## Deutungsregeln (vorab)

| Ergebnis | Deutung |
|---|---|
| Erfolg | Der 4-Bit-Vorteil trägt vom MLP (B10) und von V2a-1 ins tiefe CNN ohne Normierung, mit getuntem Regime, gegen beide Referenzen |
| Teilweise, besser als TAT, gegen SELU unklar | Gegen SELU hält der Vorteil mit Tuning und WD nicht sicher. So wird es ehrlich ins Preprint geschrieben |
| Teilweise, besser als SELU, gegen TAT unklar | TAT holt mit guter Kalibrierung auf. Der Vorteil gegen TAT hängt dann an der Kalibrierung (S4 zeigt den Fall min/max) |
| Ein „schlechter“ | Negatives Ergebnis, kommt ins Preprint |
| S1/S2 „schlechter“ bei Erfolg | Der Vorteil kostet fp32-Güte. Der Preis wird in der Zusammenfassung genannt |
| Bereich von SELU/TAT bleibt klein | Der Mechanismus aus V2a-1 fehlt hier. Ein Vorteil muss dann anders erklärt werden |

## Vorab-Wahrscheinlichkeiten

| Aussage | Wahrscheinlichkeit |
|---|---|
| P1 (gegen SELU) „besser“ | 0,65 |
| P2 (gegen TAT) „besser“ | 0,8 |
| Gesamt „Erfolg“ | 0,55 |
| Irgendein Primärtest „schlechter“ | 0,05 |
| S1 fp32 gegen SELU „unklar“ | 0,55 |
| S2 fp32 gegen TAT „unklar“ | 0,5 |
| S3 (min/max gegen SELU) „besser“ | 0,75 |
| S4 (min/max gegen TAT) „besser“ | 0,95 |
| S5 (gegen TAT mit CLE) „besser“ | 0,6 |
| Mindestens eine Wahl am Rand des Rasters | 0,6 |

Begründung:
- V2a-1 (L = 32) zeigte gegen SELU −6,2 pp bei w8a4a_best, 12/0.
- Pilot 4 (L = 50, Adam) zeigte gegen SELU mit p9999 etwa −5 pp und gegen TAT etwa −30 pp.
- Unsicher ist, ob WD und der Kosinus-Plan den Bereich von SELU und TAT bremsen, und ob mse4 TAT stark zurückholt.

## Kontrolle und Tests vor dem Start

- **Selbsttest (`--selfcheck`):**
  - E f² = 1 für alle drei Formen;
  - Delta-Init orthogonal;
  - 36 Tuning- und 36 Bestätigungsläufe;
  - Seeds neu;
  - die Reihenfolge rotiert.
- **Mechanik (`test_v2a2_mechanik.py`):**
  - Die Daten sind gleich wie im Pilot.
  - Der Kontrollpfad (konstante LR, WD 0) ist bei 20 Schritten, L = 50, für alle drei Formen bitgleich zum Pilot-Code: fp32, w8a8, w8a4a_max und w8a4a_p9999 auf 500 Val-Bildern, dazu die Kalibrierwerte.
  - mse4 ist nie schlechter als min/max.
  - Das frühe Stoppen lädt die richtigen Parameter zurück.
- **Volle Kontrolle (`--full`):** BPreCap, LR 3e-5, Seed 42000, 1720 Schritte, gegen `v2a_pilot4_D_M1Pro.json`. Das Ergebnis steht in `meta/mechanik.txt` (Lauf vor dem Freeze in `meta_vorab/`).
- **Auswertung an künstlichen Daten (`test_v2a2_analyse.py`):**
  - Ein Erfolg gegen beide, ein Teilerfolg und „schlechter“ werden erkannt.
  - Ohne Effekt gibt es höchstens 3 von 20 Fehlalarmen je Test.
  - Ausfall und fehlende Zeile zählen 90 %.
- **CLE (im Mechanik-Test):** Die fp32-Ausgabe ist gleich (relativer Unterschied < 1e-4), der fp32-Testfehler gleich (±0,1 pp). Die Gewichtsbereiche je Kanalpaar sind angeglichen.
- **Rauchtest:** 20 Schritte, 6 Tuning-Zeilen, Auswahl, 3 Bestätigungszeilen.

## Laufzeit

Pilot 4 (L = 50, 1720 Schritte, M1 Pro, allein) brauchte für BPreCap 14,2 min und für TAT und SELU je etwa 10 min.

| Phase | Schätzung |
|---|---|
| Tuning | etwa 14 h |
| Bestätigung mit PTQ (TAT zusätzlich mit CLE) | etwa 18 h |
| Gesamt, ein Worker | etwa 32 h |

Zwei Worker bringen auf dem M1 Pro nichts, weil die GPU schon mit einem Lauf ausgelastet ist (V2a-1 mit 2 Workern: 1495 s je BPreCap-Lauf einschließlich Messung, Pilot 4 allein: 546 s Training für denselben Lauf).

## Grenzen

- **Ein CNN-Typ:** Faltungsstapel ohne Pooling, Normierung und Residuen, eine Breite (32), eine Tiefe (50). Das ist nicht die Praxis (ResNet mit BatchNorm), das wäre V2b.
- **Ein Datensatz:** nur CIFAR-10, keine Augmentierung, niedrige absolute Güte (Pilot etwa 44 % fp32-Fehler).
- **Nur PTQ:** keine QAT, keine Rotation (etwa eine Online-Hadamard nach der Nichtlinearität wie QuaRot), kein BRECQ/QDrop/AdaRound. Eine Obergrenze der PTQ ist nicht gemessen.
  - Die Recherche nennt Rotation als gefährlichste Gutachterfrage. Eine Rotation könnte Ausreißer bei SELU und TAT so verteilen, dass der Vorteil der Grenze schrumpft. Das prüft V2a2 nicht.
- **CLE nur für TAT:** BPreCap und SELU sind nicht positiv homogen, CLE ist dort nicht anwendbar. Das ist ein echter Nachteil fester Grenzen und kommt so in die Deutung.
- **χ1 nur im Grenzfall unendlicher Breite gesetzt:** Nach neuerer Theorie verschiebt sich der Rand des Chaos bei endlicher Breite. χ1 ist bei 32 Kanälen nicht gemessen.
- **TAT nachgebildet:** Die Parameter sind über die C-Abbildung hergeleitet, bei d100/d200 bis 1e-9 gleich wie `dks`. Ohne Normierung je Beispiel ist das TAT-Rezept nicht vollständig.
- **Kein tanh:** Ob ein Vorteil von der Grenze oder von der Tiefenkalibrierung kommt, trennt V2a2 nicht.
- **Kleines Tuning:** 3 LR × 2 WD, 2 Seeds, je Form um die Pilot-LR gelegt. Eine Wahl am Rand wird genannt.
- **Gerät:** alles auf dem M1 Pro. Wird die Studie unterbrochen und anderswo fortgesetzt, wird das im Bericht genannt.
