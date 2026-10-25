# V16 Tiefenstudie (Entwurf, vor Start eingefroren)

## Frage
Warum verschwindet der PreCap-Vorteil bei Tiefe 64 und 128 (frueher: meist gleichauf mit SiLU/GELU)? Mit der neuen Erkenntnis aus V14: Vorsprung bei Tiefe 8-32 gross bei glatten Funktionen, bei 1200 Schritten gemessen.

## Hypothesen zur Erklaerung (konkurrierend)
- H1 Untertraining: Bei Tiefe >=64 sind nach 1200 Schritten alle Aktivierungen noch nicht trainiert; bei 3000 Schritten taucht der Unterschied wieder auf. Vorhersage: Verhaeltnis base/SiLU entfernt sich von 1 zwischen Schritt 1200 und 3000.
- H2 Residual-Skalierung: Mit Verzweigung post/sqrt(Tiefe) wird die Nichtlinearitaet pro Schicht klein; Voraktivierungen bleiben im fast linearen Bereich, PreCap-Form (Exponential, Cap) wird kaum genutzt. Vorhersage: frac_abs_pre_gt_half, pre_rms und post_rms_scaled sinken mit Tiefe; cap_fraction nahe 0; Verhaeltnis naehert sich 1.
- H3 Optimierung/Seed-Rauschen: grosse Streuung einzelner Seeds (wie V15) verdeckt Effekte; Vorhersage: weite KIs, keine Aequivalenz, Vorzeichen wechselt je Aufgabe.

## Design
6 V3-Aufgaben x Tiefen 32 (Anker), 64, 128 x 4 Aktivierungen (precap_base, precap_s225, silu, gelu) x 20 frische Seeds 9700-9719 = 1440 Runs, 11520 Checkpoint-Zeilen. 3000 Schritte, Checkpoints 0,100,200,400,800,1200,2000,3000. Konstante Lernrate: Schritt 1200 entspricht exakt einem 1200-Schritt-Lauf, damit direkt mit V14 (Tiefe 32) vergleichbar. Gleiche Netze/Formeln wie V3/V14 (Selfcheck und verify_vs_v3: Abweichung 0.0). Aenderungen gegenueber V14: kein Vollnetz-Forward pro Schritt (hmax_peak nur am Checkpoint, ca. 5x schneller), neue Spalten pre_rms, post_rms_scaled, frac_abs_pre_gt_half.

## Konfirmatorisch
Schritt 3000, Tiefen 64 und 128, 6 Aufgaben, precap_base gegen SiLU und gegen GELU = 24 Tests, gepaarter Wilcoxon (zweiseitig) auf log10(MSE-Verhaeltnis), Mittel entfaellt (pro Seed ein Wert), Holm ueber 24. Urteil je Zelle: PreCap besser/schlechter (Holm<0.05), aequivalent (90%-Bootstrap-KI des Median-Verhaeltnisses innerhalb 0.9-1.1), sonst unklar. Aequivalenz zaehlt nur, wenn nicht signifikant.

## Explorativ
Verhaeltnis-Lernkurve je Tiefe (Median ueber Aufgaben), alle Vergleiche zu Schritt 1200 und 3000 inkl. s225, Mechanismus-Mediane je Aktivierung/Tiefe/Schritt, Spearman zwischen Anteil |pre|>0.5 (precap_base, Schritt 3000) und |log10-Verhaeltnis| ueber Zellen.

## Lesart vorab
- H1 gestuetzt: Mehrzahl der Tiefe-64/128-Zellen wechselt zwischen Schritt 1200 und 3000 von aequivalent/unklar zu besser.
- H2 gestuetzt: Mechanismusgroessen fallen mit Tiefe und Verhaeltnis liegt in aequivalent-Zellen bei 1.
- H3 gestuetzt: Mehrheit unklar mit weiten KIs.
- Mehrere Hypothesen koennen gleichzeitig teilweise gelten; keine Kausalaussage aus Korrelation.

## Laufzeit und Schlafschutz
Messung: 3000 Schritte dauern ca. 49 s (Tiefe 32), 168 s (Tiefe 128) pro Run bei Last 4, Tiefe 64 ca. 95 s. Gesamt ca. 150000 Worker-Sekunden: 6 Worker ca. 7 h, 7 Worker ca. 6 h. Launcher laeuft komplett unter caffeinate, Deckel offen, Netzteil. Start: ./launch_v16.sh [Worker].
