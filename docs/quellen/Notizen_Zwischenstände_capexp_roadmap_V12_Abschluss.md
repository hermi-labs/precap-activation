# CapExp v12 — Abschluss

- Lauf: 1620/1620 eindeutige Runs, 0 bad-Runs, Seeds 200–209, drei Regime (base, high_lr, large_init), Tiefen 4/8/16, `plain` und `res_inv`.
- Manifestregel: Optimierungsempfindlichkeit nicht gestützt — in `plain` erreicht PreCap in keinem Regime in mindestens 2 von 3 Aufgaben d<0 gegen beide Referenzen (base 1/3, high_lr 0/3, large_init 0/3). Einzige Ausnahme: `abs_wave` im Basisregime.
- `res_inv`: PreCap ist in 8 von 9 Aufgabe-Regime-Zellen gegen SiLU und GELU besser; Seed-Ebene 81/90 und 83/90 (p < 1e-15).
- `plain`: Nachteil wächst mit Tiefe (Median d gegen SiLU: +0,13 / +0,29 / +0,42 bei Tiefe 4/8/16) und mit größerer Initialisierung bzw. Lernrate, vor allem auf `sin3_quad`.
- Stabilität: hmax_peak ist bei PreCap kleiner (Verhältnis 0,36–0,59); grad_final ist in `plain` aber 3–5× größer, in `res_inv` dagegen deutlich kleiner (0,03–0,12).
- Interpretation (Hypothese, kausal nicht isoliert): Die Plain-Schwäche ist strukturell und nicht durch die getesteten Init-/LR-Regime behebbar; der Residualpfad ist Teil dessen, was den Cap-Vorteil trägt.

