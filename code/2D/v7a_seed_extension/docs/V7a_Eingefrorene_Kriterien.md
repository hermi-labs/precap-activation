# v7a — Seed-Erweiterung, eingefrorene Kriterien (21.10.2026, vor Start)

Zweck: Bestätigen, ob in V6 beobachtete Hinweise (c125, s225) bei unabhängigen Seeds Bestand haben, und die Varianz bei Tiefe 64 quantifizieren.

## Design
- Zellen: radial_ripple d64, ridge_mix d64 (Mechanismus); radial_ripple d16 (Anker).
- Varianten: precap_base (c=1, s=1.5), precap_c125 (c=1.25), precap_s225 (s=2.25), SiLU, GELU.
- Seeds 1000–1019 (20, unabhängig von V6-Seeds 900–904), 1200 Schritte, Checkpoints 0/100/300/500/800/1200.
- 300 Runs, 1800 Checkpoint-Zeilen. Formel, Architektur, Optimierer unverändert gegenüber V6.

## Primäre Kriterien (je Variante, einseitig gepaart gegen precap_base, d = log10(MSE_Variante/MSE_base))
- Tiefe-64-Zelle bestanden: Median d < 0 UND Wilcoxon einseitig p < 0.025 (Bonferroni für 2 Varianten) UND >= 13/20 Seeds besser.
- Anker (radial d16) bestanden: Median d <= +0.05 (nicht klar schlechter).
- Bewertung: ROBUST = beide Tiefe-64-Zellen und Anker bestanden; AUFGABENABHÄNGIG = eine Zelle und Anker; sonst KEIN BELASTBARER EFFEKT.

## Referenzvergleich der Basis
base gegen SiLU bzw. GELU je Zelle: Median d < 0 UND >= 13/20 Seeds.

## Varianzmaße (Bericht, kein Entscheidungskriterium)
Ausreißerrate (Test-MSE > 5x Median der Variante in der Zelle), Variationskoeffizient, 10%-getrimmter Mittelwert.

## Regeln
- V6-Seeds werden nicht in die Primärentscheidung eingerechnet; ein gepoolter Bericht ist rein explorativ.
- Keine nachträglichen Änderungen an Kriterien, Varianten oder Seeds.
- Integrität: 300/300 Runs, 1800/1800 Zeilen, eindeutige Schlüssel, keine NaN, Formel-Selfcheck = 0.
- Auswertung: python analysis/analyze_v7a.py
