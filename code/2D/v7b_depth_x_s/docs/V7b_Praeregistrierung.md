# v7b — Präregistrierung: Tiefe × Sättigungsbreite

**Status:** Eingefrorenes Paket, noch nicht gestartet.  
**Frage:** Ist die in v7a bei radial_ripple Tiefe 16 beobachtete Verbesserung durch s=2.25 systematisch tiefenabhängig?

## Design
- Primär: radial_ripple; Replikation: ridge_mix.
- Tiefen: 8, 16, 32, 64.
- Varianten: precap_base (c=1,s=1.5), precap_s225 (c=1,s=2.25), SiLU, GELU.
- Neue Seeds: 2000–2019; unabhängig von v6/v7a.
- 1200 Schritte, identische Architektur/Optimierung/Datenprotokoll wie v7a.
- Umfang: 8 Zellen × 4 Varianten × 20 Seeds = 640 Runs; 3840 Checkpoint-Zeilen.

## Primäre Auswertung
Metrik d=log10(MSE_s225/MSE_base), negativ begünstigt s225.
- Pro radial-Zelle: Median d, Siege/20, einseitiger gepaarter Wilcoxon-Test.
- Eine Zelle ist ein bestätigter s225-Vorteil bei Median d<0, mindestens 13/20 Siegen und p<0.0125 (Bonferroni für vier Tiefen).
- Primäre Tiefe×s-Schlussfolgerung: mindestens zwei benachbarte radial-Tiefen bestätigt ODER ein vorab klarer Kontrast: d16 bestätigt und d64 nicht bestätigt, mit Effektgrößenunterschied >=0.15 log10.
- ridge_mix berichtet dieselben Tests als unabhängige Replikation, ohne die primäre Entscheidung zu ersetzen.
- SiLU/GELU dienen als Kontext; die direkte Entscheidung bleibt s225 vs base.

## Integrität
640/640 Runs, 3840/3840 Checkpoints, eindeutige Schlüssel, keine NaN, Formel-Selfcheck=0. Keine Änderung an Seeds, Varianten, Architektur oder Kriterien nach Start.
