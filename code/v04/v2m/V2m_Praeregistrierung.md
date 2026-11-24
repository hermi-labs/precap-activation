# V2m – Präregistrierung: Eingriff an der Form (frühe Sättigung der Hülle)

Stand: 23.11.2026, 06:44 (Uhrzeit mit `date` geprüft). Freigabe Aaron 23.11. morgens: „Ja, V2m starten, wobei wir am Schluss schauen, ob wir es in v0.4 mit reinnehmen.“ PreCap, BPreCap, die Hülle und das Kalibrierungsrezept sind Aarons Ideen. Das Design habe ich mit Claude entworfen, aus dem Entwurf `studien/v2m_entwurf/V2m_Entwurf.md` (22.11.). Privat, nichts davon wird veröffentlicht.

## Frage

V2k (präregistriert, 9/9 Seeds) zeigt: BPreCaps 4-Bit-Vorteil gegenüber ctanh sitzt im Abschneideteil des Fehlers. Ein Abschneidefehler gegebener Größe kommt bei BPreCap schwächer an den Logits an (ḡ^C, q = 0,77, 9/9). Ungeprüft ist, woran das liegt. Eine naheliegende Lesart ist, dass BPreCap spät sättigt: Die Hülle ist bis 80 % der Grenze die Identität.

**Wird die Weitergabe des Abschneidefehlers stärker, wenn BPreCap früher sättigt, bei sonst gleichem Rezept?**

## Die eine Änderung

Das Knie der Hülle φ_ρ geht von ρ = 0,8 auf ρ = 0,4, die Grenze bleibt M = 3. Die Konstanten werden für die begrenzte Funktion neu gelöst, wieder mit E f = 0, E f² = 1, χ1 = 1,04. Gelöst wurde mit dem unveränderten Löser `math/p4_bounded_depth.py`, nur dessen `RHO` gesetzt (`v2m_konstanten.py` → `kL2B_d50_rho04.json`). Gegenprobe: Mit ρ = 0,8 findet der Löser die V2a2-Konstanten exakt wieder (Abweichung 0).

| ρ | λ | α | b | χ1 | Bereich | Anteil der Eingaben jenseits des Knies bei der Init |
|---|---|---|---|---|---|---|
| 0,8 (BPreCap, V2a2) | 1,75242 | 1,68131 | 0,46473 | 1,04 | [−2,289; 3,000] | 1,7 % |
| 0,4 (V2m) | 1,80508 | 1,65194 | 0,46350 | 1,04 | [−2,192; 2,972] | 25,0 % (Selbsttest 25,2 %) |

Alles andere ist wie in V2a2 und V2f:
- L = 50, 32 Kanäle, Delta-orthogonal, ohne Normierung und Residuen;
- AdamW (MLX 0.32.4, ohne Bias-Korrektur), Kosinus auf 0, 3440 Schritte, frühes Stoppen an 8 Punkten;
- PTQ wie V2a2 (w8 je Kanal; a4 asymmetrisch, beste von fünf Kalibrierungen auf 1000 Trainingsbildern);
- die Messung ist `v2k_run.measure`, eingefroren, bei mse4 an 1000 Testbildern.
- Der Code ruft die eingefrorenen Funktionen auf (`v2e_run.run_keep`, `v2k_run.measure`, `v2k_run.summary`). Neu ist nur die Formfunktion. Sie ist Zeichen für Zeichen der Rechenweg aus `v2a2_run.make_act`, mit ρ und (λ, α, b) als Parametern.

**Lernrate:** 3e-5, WD 0, ohne neues Tuning. Das Tuning in V2a2 hat diesen Wert für BPreCap gewählt, und dieselbe Lernrate haben auch ctanh und alle tanh-Varianten getroffen. Das ist eine Grenze, siehe unten.

**Seeds:** 42353–42361 (9), dieselben wie in V2f und V2k. Damit sind Startgewichte und Batch-Reihenfolge je Seed dieselben wie bei der Referenz.

**Referenzen (nicht neu gerechnet):**
- BPreCap ρ 0,8 und ctanh: V2k-Zeilen bei mse4 (`studien/v2k/workers/w0.csv`);
- BPreCap-Testfehler: V2a2-Zeilen (`studien/v2a2/workers/w0.csv`), stage confirm.

**Soll:** 9 Zeilen in `workers/w0.csv`, Gewichte am frühen Stopp in `gewichte/`, Messungen in `mess/`.

## Kontrolle vor dem Start (im Selbsttest, muss erfüllt sein)

- Die Formfunktion mit ρ = 0,8 und den V2a2-Konstanten ist auf 400 001 Punkten in [−40, 40] bitgleich zu `v2a2_run.make_act('bprecap')`.
- 120 Trainingsschritte, Seed 42353, mit beiden Funktionen: alle Gewichte und die Val-Kurve bitgleich. So ist gezeigt, dass der neue Code außer der Form nichts ändert. (Im Entwurf war ein voller Kontrolllauf über 3440 Schritte vorgesehen, etwa 41 min. Die bitgleiche Form auf dem ganzen Gitter und die bitgleichen 120 Schritte zeigen dasselbe in kürzerer Zeit.)
- Die Variante liegt bei E f = 0, E f² = 1, χ1 = 1,04, jeweils ±0,002, und unterscheidet sich sichtbar vom Original (größte Abweichung > 0,01).
- Die Messmechanik läuft an einem untrainierten Netz der Variante, alle Kennzahlen endlich.
- Vorab am 23.11. um 06:35 einmal komplett gelaufen: `"alle_ok": true`, bitgleich in Form und Training, Abweichung höchstens 0,253, Dauer 6:42 min.

## Primärtests (mse4, 9 Paare, exakter Wilcoxon, zweiseitig, Holm über 2)

- **E1:** x_s = log(ḡ^C(ρ 0,4) / ḡ^C(ρ 0,8)). Gefragt ist, ob ein Abschneidefehler gleicher Größe bei früher Sättigung stärker an den Logits ankommt.
- **E2:** x_s = log(D^C_all(ρ 0,4) / D^C_all(ρ 0,8)). Gefragt ist, ob Abschneiden in allen Schichten zugleich die Logits stärker stört.

**Entscheidungsregel**, je Test:
- „bestätigt“: Holm-p < 0,05, Median > 0 und mindestens 7 von 9 Paaren positiv;
- „widerlegt“: Holm-p < 0,05, Median < 0 und mindestens 7 von 9 negativ;
- sonst „unklar“.

Ein gültiges Paar braucht eine Zeile mit Status ok, endliche Kennzahlen, D^C > 0 und ḡ^C > 0. Bei weniger als 7 gültigen Paaren lautet das Urteil „unklar (zu wenige Paare)“.

## Sekundär (zweiseitig, Holm über 4)

- **S1:** 4-Bit-Preis (w8a4a_best − fp32), ρ 0,4 − ρ 0,8.
- **S2:** Testfehler w8a4a_best, ρ 0,4 − ρ 0,8.
- **S3:** fp32-Testfehler, ρ 0,4 − ρ 0,8. Zeigt, ob die Form auch das Lernen ändert.
- **S4:** log(ḡ^C(ρ 0,4) / ḡ^C(ctanh)). Liegt die Variante noch unter ctanh?

Ein Ausfall der Variante zählt in S2 und S3 als Fehler 0,9; in S1 und S4 fällt das Paar weg.

**Urteil:** Bei Holm-p < 0,05 „Unterschied“ mit Richtung, sonst „kein klarer Unterschied“.

## Beschreibend

- Mediane aller Kennzahlen je Form (ρ 0,4, ρ 0,8, ctanh).
- Lage der Variante zwischen BPreCap und ctanh: t = (log v − log b) / (log c − log b) für ḡ^C, D^C, ḡ^Q und D^Q. t = 0 heißt wie BPreCap, t = 1 heißt wie ctanh.
- log-Quotienten gegenüber ρ 0,8 für r̄^C, r̄^R, ḡ^R, D^R und den Anteil abgeschnittener Werte (frac_clip).
- Differenzen für κ, ν und die Superposition.

## Deutung

| Ausgang | Deutung |
|---|---|
| E1 und E2 bestätigt | Die späte Sättigung der Hülle trägt den Vorteil beim Abschneiden mit. Die Lesart aus v0.4 wird vom Ausblick zum gemessenen Befund, für den Teil, den die Hülle ausmacht. |
| beide unklar | Das Knie der Hülle ist nicht klar der Hebel. Der Vorteil sitzt eher im Kern (tanh im Exponenten, Asymmetrie) oder in der Verteilung nach dem Training. |
| einer oder beide widerlegt | Frühere Sättigung schwächt die Weitergabe sogar. Die Lesart „spät sättigen hilft“ ist für die Hülle falsch und wird in v0.4 gestrichen bzw. umgeschrieben. |
| nur einer bestätigt | Teilbefund, wird so berichtet. |

Ob V2m in v0.4 kommt, entscheidet Aaron nach der Auswertung. Auch ein unklares oder widerlegendes Ergebnis wird berichtet.

## Vorab-Wahrscheinlichkeiten

- E1 bestätigt: 0,45; widerlegt: 0,10.
  - Das Knie ändert nur, was jenseits von 1,2 liegt.
  - Der Kern sättigt über s·tanh(bx/s) selbst, und die negative Seite sättigt über die Exponentialfunktion früh.
  - Die Nichtlinearität ändert sich kaum (1 − a₁²: 0,0171 gegen 0,0176).
- E2 bestätigt: 0,40; widerlegt: 0,10.
- S3 (fp32 anders): 0,30.
- Alle 9 Läufe ok: 0,92.

## Grenzen

- Eine Variante (ρ = 0,4), kein Verlauf über ρ und keine spätere Sättigung (ρ = 0,95).
- Ohne Tuning. Bräuchte ρ = 0,4 eine andere Lernrate, wäre der Vergleich verzerrt. S3 würde das anzeigen.
- Die Paare vergleichen trainierte Netze. Ein Unterschied kann aus der Form selbst kommen oder aus anderen Gewichten nach dem Training. V2m trennt das nicht.
- Nur mse4 wird gemessen. Für die V2k-Hauptaussage war mse4 die Primärkalibrierung; max wird hier zugunsten der Laufzeit weggelassen.
- Nur die Hülle wird verändert. Ein Nullergebnis schließt die Form als Ursache nicht aus.
- Ein Rechner (M1 Pro, MLX 0.32.4), ein Datensatz.

## Aufwand und Ablauf

Je Seed etwa 41 min Training mit PTQ (wie BPreCap in V2f) plus etwa 7 min Messung, mit einem Worker zusammen etwa 7,3 h.

Start über `launch_v2m.sh` mit diesem Ablauf:
1. Prüfungen: Netzteil, keine anderen Worker, Ergebnisordner noch nicht vorhanden.
2. Selbsttest komplett mit `alle_ok`.
3. Freeze.
4. Hash-Prüfung von V2a2, V2e, V2f, V2k.
5. Auswertungstest.
6. Rauchtest: 1 Lauf mit 20 Schritten und 50 Bildern, Soll 1 ok-Zeile.
7. Ein Worker `v2mw0` mit caffeinate und dem Netzteil-Wächter `v2mwache`.

Die Läufe sind je Seed wiederaufnehmbar. Ausgewertet wird mit `v2m_analyze.py` (Python mit scipy, absoluter Pfad).
