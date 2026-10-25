# 3D-V1 – Präregistrierung (eingefroren vor dem Start, 25.10.2026)

## Fragen
1. Hält ein PreCap- oder CapExp-Vorteil bei 3D-Eingaben, wenn jede Aktivierung ihre eigene Lernrate bekommt?
2. Ist ein Vorteil mehr als reine Skalierung? Dafür dient SiLU×2 als Kontrolle: Es hat am Ursprung dieselbe Steigung 1 wie PreCap und CapExp.
3. Ändert Pre-LayerNorm das Ergebnis? Mit LN liegen etwa 30 % statt 3 % der Voraktivierungen über 0,5.

## Hintergrund
V18: Der flache PreCap-Vorteil verschwand mit LR-Tuning. SiLU/GELU wählten den Rasterrand 3e-3. PreCap sättigt durch das tanh bei etwa 3,08 (Steigung bei x = 3 nur 0,18). Das reine CapExp behält die Steigung e oberhalb von c = 1.

## Design
- Aufgaben (Eingabe in [-1,1]^3, 4096 Train, 1024 Test): smooth_bowl3d, gyroid, torus_sdf, box_sdf, spheres_union.
- Residual-MLP Breite 64, Tiefen 8 und 16, h = skip + act(W·LN?(h))/sqrt(D); norm ∈ {none, layernorm}.
- Aktivierungen: precap_base (c = 1, s = 1,5), capexp_c1 (c = 1), silu, silu2x. GELU wurde aus Zeitgründen entfernt; in V18 verhielt es sich wie SiLU.
- Adam LR ∈ {1e-3, 3e-3, 1e-2}, 3000 Schritte, Batch 128, 8 Checkpoints.
- Seeds 30000–30029: 30000–30009 Tuning, 30010–30029 Bestätigung.
- 5 × 2 × 2 × 4 × 3 × 30 = 7200 Runs, 57600 Zeilen.

## LR-Auswahl
Je Aufgabe × Tiefe × Norm × Aktivierung die LR mit dem kleinsten Median log10-Test-MSE auf den Tuning-Seeds (Schritt 3000). Nicht-endliche Runs zählen als MSE 1e6.

## Konfirmatorisch
- Kandidaten precap_base und capexp_c1, jeweils gegen silu und silu2x, je Aufgabe × Tiefe × Norm.
- Gepaarter Wilcoxon-Test der log10-MSE auf den 20 Bestätigungs-Seeds: 80 Tests, Holm, α = 0,05.
- Vorteil je (Kandidat, Norm, Referenz) bestätigt bei mindestens 6 von 10 Zellen signifikant besser und höchstens 1 schlechter.
- Hauptfrage: der Vergleich gegen silu2x.

## Explorativ
- Wechselwirkung LayerNorm × Aktivierung (gepaarte Differenz der Differenzen je Seed).
- Anteil |pre| > 0,5, cap_fraction, Rasterrand-Zählung (LR = 1e-2 gewählt).
- Instabilität von CapExp (nicht-endliche Runs).

## Erwartung (vor dem Lauf)
- Kein klarer Vorteil von PreCap gegen silu2x: etwa 0,7.
- CapExp besser als PreCap: etwa 0,35; Risiko von Instabilität bei LR 1e-2.
