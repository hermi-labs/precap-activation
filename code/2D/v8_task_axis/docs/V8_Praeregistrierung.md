# V8 Präregistrierung — Aufgabenachse

Frage: Ist s225 gegenüber base für radiale/isotrope Zielfunktionen besser, aber nicht für ridge-/anisotrope Zielfunktionen?

Design: radial_ripple, radial_highfreq, radial_bessel versus ridge_mix, ridge_highfreq, ridge_curved; Tiefen 16/32; base, s225, SiLU, GELU; Seeds 3000–3019; 960 Runs und 5760 Checkpoints.

Primär: d=log10(MSE_s225/MSE_base). Pro Aufgaben×Tiefe-Zelle werden Median d, Siege und einseitiger Wilcoxon-Test berichtet. Gruppenentscheidung: s225-Hinweis in einer Gruppe, wenn mindestens 4/6 Zellen negativen Median haben und der Median der sechs Zell-Mediane negativ ist. Ein gruppenübergreifender Kontrast gilt als unterstützt, wenn radial mindestens 4/6 negativ und ridge höchstens 2/6 negativ ist. Dies ist eine Replikations-/Richtungsstudie; p-Werte je Zelle sind deskriptiv nach Bonferroni p<0.0083.

Integrität: eindeutige 960 Runs; 5760 Checkpoints; keine NaN; Formel-Selfcheck 0. Keine nachträgliche Änderung von Funktionen, Seeds oder Kriterien.
