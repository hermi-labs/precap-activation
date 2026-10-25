# V18 – Faire Lernrate, flache bis mitteltiefe Netze (vor Start eingefroren, 25.10.2026)

## Ziel
Prüfen, ob der PreCap-Vorteil in flachen bis mitteltiefen Residualnetzen (Tiefe 8/16/32) echt ist oder nur daher kommt, dass Lernrate 0,001 zufällig PreCap begünstigt. Zweitens: prüfen, ob der Vorsprung dort größer ist, wo PreCap mehr große Voraktivierungen hat (Erklärung "kleine Werte = alle Aktivierungen fast gleich", Taylor: PreCap ≈ x+0,51x², SiLU ≈ x/2+x²/4, GELU ≈ x/2+0,40x²).

## Design
- Aufgaben: radial_ripple, ridge_mix, smooth_bowl, corner_kink, angled_kink (Formeln unverändert aus V3/V14/V17; highfreq_grid entfällt, weil dort keine Aktivierung lernt).
- Tiefen 8, 16, 32; Residualskala 1/sqrt(D) (wie V14/V16).
- Aktivierungen precap_base (c=1, s=1,5, a=1, gamma=0,01), SiLU, GELU.
- Adam-Lernraten 3e-4, 1e-3, 3e-3, konstant, 3000 Schritte, Batch 128, Breite 64, INIT_K 0,5.
- Seeds 9900–9929 (disjunkt zu V1b–V17, auch Testseeds +10000). Tuning-Seeds 9900–9909, Bestätigungs-Seeds 9910–9929.
- 5 × 3 × 3 × 3 × 30 = 4050 Runs, 32400 Checkpoint-Zeilen (0/100/200/400/800/1200/2000/3000).
- Zeilen werden pro Run erst am Ende geschrieben; ein Neustart überspringt fertige Runs (keine Duplikate).

## LR-Auswahl (fair, getrennt)
Je Aufgabe × Tiefe × Aktivierung wird die Lernrate mit dem kleinsten Median von log10(Test-MSE) bei Schritt 3000 auf den 10 Tuning-Seeds gewählt. Nicht-endliche Runs zählen als MSE 1e6.

## Konfirmatorisch
Auf den 20 Bestätigungs-Seeds, jede Aktivierung mit ihrer eigenen gewählten Lernrate: precap_base gegen SiLU und gegen GELU je Aufgabe × Tiefe, gepaarter zweiseitiger Wilcoxon-Test auf log10(MSE_PreCap/MSE_Ref). 30 Tests, Holm, alpha 0,05.

H_A (je Referenz getrennt): Vorteil bestätigt, wenn mindestens 8 von 15 Zellen "PreCap besser" und höchstens 1 "PreCap schlechter".

## Mechanismus (vorab festgelegtes Muster)
Über die 15 Zellen: Spearman zwischen log10(ratio gegen SiLU) und dem Median-Anteil |pre|>0,5 von PreCap (Bestätigungs-Seeds, gewählte LR, Schritt 3000). Vorhersage rho < 0, einseitig p < 0,05. Wird als Muster gewertet, nicht als Kausalbeweis.

## Explorativ
Vergleich bei fester LR 1e-3 (Anschluss an V14–V17), Schritt 1200, LR-Empfindlichkeit je Aktivierung, gewählte Lernraten.

## Geltungsbereich
2D-Synthetik, res_inv-MLP, Adam ohne Schedule und ohne Weight Decay.
