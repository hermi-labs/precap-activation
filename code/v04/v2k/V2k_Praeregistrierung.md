# V2k – Präregistrierung: Abschneiden oder Runden? Kreuzterme des 4-Bit-Fehlers in den V2f-Netzen

Stand: 21.11.2026, 18:47 (mit `date` geprüft). Die Präregistrierung entstand vor jeder Messung mit `v2k_run.py`. PreCap, BPreCap, die Hülle und das Kalibrierungsrezept sind Aarons Ideen. Das Design habe ich mit Claude entworfen. Freigabe: Aaron am 21.11. abends („mach beides hast den mac bis morgen“, Kreuzterm-Messung und V2h). Gestartet wird nach V2h über `launch_v2k.sh`. Eingefroren wird erst, nachdem der Selbsttest einmal komplett gelaufen ist (Regel seit dem V2g-Abbruch).

## Anlass

V2f (präregistriert, 9 Seeds) hat bestätigt:
- BPreCap rundet bei gleicher Kalibrierung etwa 10 % gröber als ctanh.
- Bei der abschneidenden Kalibrierung mse4 erreicht ein Rundungsfehler gegebener Größe die Logits bei BPreCap aber etwa 16 % schwächer (ḡ^Q, q = 0,84, 9/9).
- Bei max kehrt sich das um.
- Die Kohärenz (Superposition) unterscheidet sich nicht.

Meine Deutung war: **BPreCap verträgt das Abschneiden der Ränder besser.** Gemessen ist das bisher nicht, denn der 4-Bit-Fehler wurde nur als Ganzes betrachtet. V2k zerlegt ihn in zwei Teile:
- **Abschneiden:** e_C = clip(h) − h
- **Runden:** e_R = Q(h) − clip(h)

Damit gilt e_Q = e_C + e_R exakt. Die Grenzen von clip sind die tatsächlichen Gittergrenzen von `qa_asym`, und es gilt Q(clip(h)) = Q(h).

Zusätzlich misst V2k die Kreuzterme zwischen den Schichten. Vorbild sind Tarawneh 2026 und Chen et al. 2026 (Recherche vom 21.11., `docs/literatur/reports/Quantisierungsfehler über die Tiefe.md`): Heben sich die Fehler einzelner Schichten auf, und wie stark ist der nichtlineare Rest?

## Material

- Die 18 gespeicherten V2f-Netze: BPreCap und ctanh, Seeds 42353–42361, Gewichte am frühen Stopp (`../v2f/gewichte/`). Sie sind bitgleich zu V2a2 bzw. V2d nachgerechnet.
- **Kein Training.**
- Gemessen wird an denselben 1000 Testbildern (Teile zu 250), mit derselben Kalibrierung (`R.calib`, Seed = Lauf-Seed), wie in V2f.
- Nur Aktivierungen werden gerundet, Gewichte bleiben fp32. Eingänge i = 0 … 50: 0 ist das Bild, 50 der Eingang der Linearschicht.
- Kalibrierungen: **mse4 (primär)** und max (beschreibend).

## Messgrößen (je Netz und Kalibrierung, `v2k_run.measure`)

**Fehleranteile und Weitergabe:**
- r_i^K = ‖e_K‖ / ‖h‖ am Eingang i im sauberen Pfad, für K ∈ {Q, C, R}.
- d_i^K = ‖Δz‖ / ‖z‖, wenn nur am Eingang i der Teil K eingesetzt wird. Dabei ist z die Logits-Matrix über alle Bilder.
- ḡ^K ist das geometrische Mittel von d_i^K / r_i^K über i = 1 … 49. Schichten ohne Abschneiden (r_i^C = 0) fallen bei ḡ^C heraus. Ihre Zahl wird berichtet.

**Gesamtwirkung:**
- D^K = ‖Δz‖ / ‖z‖, wenn an allen Eingängen K eingesetzt wird (im jeweils gestörten Pfad).

**Kreuzterme:** δ_i ist die Logit-Abweichung, wenn nur Eingang i gerundet wird (Vektor über alle Bilder und Logits).
- **κ** = (‖Σδ_i‖² − Σ‖δ_i‖²) / Σ‖δ_i‖² ist der lineare Kreuzterm. κ < 0 heißt, die Einzelfehler heben sich teilweise auf.
- **ν** = ‖δ_all − Σδ_i‖ / ‖δ_all‖ ist der nichtlineare Rest.
- Die Kosinus-Matrix der δ_i wird gespeichert, ebenso die δ_i selbst (`mess/*_delta.npz`).

**Abgleich:** d_i^Q und D^Q müssen die V2f-Werte desselben Netzes reproduzieren, mit relativer Abweichung < 10⁻³ (`v2f_abw_dQ`, `v2f_abw_D`).

## Primärtests (Kalibrierung mse4, q_s = BPreCap / ctanh je Seed, 9 Paare, exakter Wilcoxon, einseitig, Holm über 2)

| Test | Größe je Seed | vorhergesagte Richtung |
|---|---|---|
| **K1 Abschneiden** | log q(D^C) | < 0: BPreCap verliert durch das Abschneiden allein weniger |
| **K2 Abschneiden vor Runden** | log q(D^C) − log q(D^R) | < 0: BPreCaps Vorteil ist beim Abschneiden größer als beim Runden |

**Entscheidungsregel je Test:**
- **„bestätigt“**, wenn p_Holm < 0,05 und mindestens 7 von 9 Werten in Richtung liegen.
- **„nicht bestätigt“** sonst. Der zweiseitige p-Wert wird berichtet.
- **„unklar (Kontrolle)“**, wenn weniger als 9 gültige Paare vorliegen. Gültig heißt:
  - Beide Netze haben bei beiden Kalibrierungen den Status ok, und der Abgleich mit V2f liegt unter 10⁻³.
  - Bei mse4 sind alle Kennzahlen endlich, und D^C, D^R > 0.

**Gesamtdeutung (vorab):**

| K1 | K2 | Deutung |
|---|---|---|
| bestätigt | bestätigt | Der Vorteil sitzt im Abschneiden. Die V2f-Deutung ist gestützt. |
| bestätigt | nicht | BPreCap verliert durch das Abschneiden weniger, aber nicht klar mehr als beim Runden. |
| nicht | bestätigt | Relativ größerer Vorteil beim Abschneiden, absolut nicht klar kleiner. |
| nicht | nicht | Die Abschneide-Deutung aus V2f ist nicht bestätigt. Im v0.4-Entwurf wird sie abgeschwächt. |

## Sekundärtests (mse4, zweiseitig, Holm über 4)

| Nr. | Größe je Seed | Frage |
|---|---|---|
| S1 | log q(ḡ^R) | Erreicht ein reiner Rundungsfehler gegebener Größe die Logits unterschiedlich stark? |
| S2 | log q(ḡ^C) | dasselbe für einen reinen Abschneidefehler |
| S3 | κ(BPreCap) − κ(ctanh) | Heben sich die Einzelfehler unterschiedlich stark auf? |
| S4 | ν(BPreCap) − ν(ctanh) | Ist der nichtlineare Rest verschieden? |

Urteil: p_Holm < 0,05 heißt „Unterschied (BPreCap kleiner/größer)“, sonst „kein klarer Unterschied“.

**Beschreibend:**
- K1, K2 und S1–S4 bei max, ohne Korrektur (wenn bei max nichts abgeschnitten wird: „nicht berechenbar“)
- Mediane aller Kennzahlen je Form und Kalibrierung
- Energie-Anteil des Abschneidens (D^C)² / ((D^C)² + (D^R)²)
- Profil von log q(g_i^Q) und log q(r_i^C) in fünf Blöcken (1–10, 11–20, 21–30, 31–40, 41–49), also wo im Netz der Unterschied entsteht
- Kosinus-Blockmatrix je Form
- κ im Vergleich zu Tarawneh 2026 (Median c = −0,05 in Plain-CNNs)

## Vorab-Wahrscheinlichkeiten

| Aussage | p |
|---|---|
| alle 36 Messungen ok, Abgleich mit V2f < 10⁻³ | 0,85 |
| K1 bestätigt | 0,6 |
| K2 bestätigt | 0,5 |
| S1 Unterschied | 0,35 |
| S3 Unterschied | 0,2 |
| κ bei beiden Formen zwischen −0,2 und +0,2 (Median) | 0,7 |
| bei max wird in den meisten inneren Schichten nichts abgeschnitten | 0,6 |

Begründung: ḡ^Q war bei mse4 klar kleiner und bei max größer. Das spricht für das Abschneiden. Es kann aber auch sein, dass mse4 bei BPreCap anders rundet, etwa wegen eines anderen Gitterabstands, und das Runden den Unterschied trägt. Dann wäre K1 nicht bestätigt und S1 zeigte einen Unterschied. sup ≈ 1 in V2f spricht dafür, dass κ klein ist.

## Grenzen

- Messstudie, kein Eingriff. Die Zerlegung zeigt, welcher Teil des Fehlers wirkt, nicht warum die Form das bewirkt.
- e_C und e_R sind nicht unabhängig, denn Runden an den Gitterrändern hängt am Abschneiden. Die Energie-Summe (r^C)² + (r^R)² ist nur ungefähr (r^Q)². Der Selbsttest berichtet den Quotienten.
- 9 Seeds, ein Netz (CNN, L = 50, ohne Normierung), ein Datensatz, 1000 Bilder, Aktivierungen je Tensor, Gewichte fp32, M1 Pro.
- Die Netze und die Hypothese stammen aus V2f. Unabhängig neu sind nur die Zerlegung und die Kreuzterme.

## Ablauf

1. `launch_v2k.sh` wartet, bis `v2hw0` beendet ist, prüft Netzteil und andere Worker.
2. **Selbsttest vor dem Freeze**, komplett, Fehler nach `logs/selfcheck_err.txt`. Er prüft:
   - Zerlegung exakt
   - d^Q und D^Q gleich `v2e_run.measure` auf 50 Bildern
   - κ direkt nachgerechnet
   - alles endlich
3. Dann Freeze, Hash-Prüfung von V2e und V2f, Auswertungstest, Rauchtest (50 Bilder) und Start in screen `v2kw0` mit caffeinate und dem Wächter `v2kwache`.
4. **Soll:** 36 Zeilen (18 Netze × 2 Kalibrierungen). Dauer etwa 3 h (geschätzt aus der V2f-Messzeit × 1,5).
5. Ausgewertet wird mit `v2k_analyze.py <V2k-Ordner absolut>` (scipy-venv), mit Hash-Prüfung vorher und nachher. Der Bericht kommt nach `V2k_Auswertung.md`.
