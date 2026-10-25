# CapExp 2D v1 Pilot — Manifest

- Aktivierungen: PreCap, SiLU, GELU.
- Architektur: residual (`res_inv`-artig), Breite 64.
- Aufgaben: `radial_ripple`, `ridge_mix`, `corner_kink`.
- Tiefen: 4, 8, 16.
- Seeds: 300–304.
- Gesamtumfang: 135 Runs.
- Primärmetrik: gepaarter Log-MSE-Quotient d = log10(MSE_PreCap / MSE_Referenz).
- PASS-Kriterium: mindestens 2 von 3 Aufgaben mit negativem Median d gegen beide Referenzen, keine höhere bad-Rate.
- Nach PASS: größerer 2D-Holdout mit frischen Seeds und fixierten Zellen.
