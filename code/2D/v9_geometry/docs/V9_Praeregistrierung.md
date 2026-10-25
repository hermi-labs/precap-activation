# V9 Praeregistrierung — Geometrie-Kontinuum

Frage: Waechst der Nachteil von s225 gegenueber base (d=log10 MSE_s225/MSE_base, negativ = s225 besser) kontinuierlich mit der Anisotropie der Zielfunktion?

Design: f_a = sin(3(1-a)(x^2+y^2)+4a x)+0.4 cos(2.5(1-a) sqrt(x^2+y^2+0.05)+3a y), a in {0,.25,.5,.75,1}; a=0 radial, a=1 ridge. Tiefen 16, 32. Varianten base und s225. Seeds 4000-4019 (neu). 400 Runs, 2400 Checkpoint-Zeilen. Rest wie V8.

Geometrie-Kennzahl: Anisotropie A=(l1-l2)/(l1+l2) der Eigenwerte der Gradientenkovarianz E[grad f grad f^T] ueber 4096 feste Punkte in [-1,1]^2 (Seed 7).

Primaer: s225 wird als geometrieabhaengig unterstuetzt, wenn ALLE drei gelten: (1) Spearman-rho zwischen A und Median-d ueber die 10 Zellen >= 0.6; (2) seed-gemittelter Endpunktkontrast d(a=0) < d(a=1), einseitiger gepaarter Wilcoxon p<0.05; (3) Median d bei a=0 (seed-gemittelt) < 0. Sonst: nicht bestaetigt. Zellweise Siege/p werden berichtet, sind aber nicht entscheidend.

Integritaet: 400 eindeutige Runs, 2400 Zeilen, keine NaN, Formel-Selfcheck 0. Keine nachtraegliche Aenderung.
