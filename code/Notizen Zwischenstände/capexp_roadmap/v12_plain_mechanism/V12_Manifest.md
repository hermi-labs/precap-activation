# CapExp v12 — Mechanismusstudie zur Plain-Schwäche

Vor dem Start festgeschrieben am 20.10.2026, nach Abschluss von v11 confirm.

## Ausgangspunkt

v11 zeigte: PreCap bleibt dem hybriden Kandidaten überlegen. Die Cap-Familie wirkt in `res_inv` deutlich besser als SiLU/GELU, während sie bei `plain`, insbesondere auf `sin3_quad`, oft verliert. V12 ist deshalb keine weitere Aktivierungssuche, sondern eine Ursachenstudie.

## Frage

Ist die Plain-Schwäche von PreCap durch empfindliche Optimierung (Initialisierung/Lernrate) erklärt, und verschwindet sie mit Residualpfaden oder bleibt sie unabhängig vom Trainingsregime bestehen?

## Raster: 1.620 Runs

- Aktivierungen: PreCap, SiLU, GELU; PreCap-Parameter unverändert.
- Aufgaben: `sin3_quad`, `sin_mix`, `abs_wave`.
- Tiefen: 4, 8, 16.
- Architekturen: `plain`, `res_inv`.
- Regime: base (0,5; 0,001), high_lr (0,5; 0,003), large_init (1,0; 0,001); jeweils (init_k; lr). Das doppelte Extremregime large_init_high_lr entfällt bewusst, weil es die zentrale Optimierungsfrage kaum zusätzlich identifiziert, aber ein Drittel mehr Rechenbudget kostet.
- Seeds: 200–209, frisch und unabhängig von v10/v11.
- Je 3000 Adam-Schritte, gleiche Daten- und Modellpipeline wie v11.

## Messung und Entscheidung

Primär: gepaarter d = log10(MSE_PreCap / MSE_Referenz), innerhalb gleicher Aufgabe, Tiefe, Architektur, Regime und Seed. d<0 heißt PreCap besser. Zusätzlich: hmax_peak, grad_final, cap_final und bad-Rate.

Die Ursache „Optimierungsempfindlichkeit“ wird gestützt, wenn PreCap in `plain` in mindestens 2 von 3 Aufgaben bei einem der vier Regime median d<0 gegen beide Referenzen erreicht und die Stabilitätsmetriken zugleich günstiger werden. Bleibt PreCap in `plain` unter allen Regimen schlechter, während `res_inv` robust bleibt, stützt das die Erklärung „Residualpfad ist strukturell nötig“.
