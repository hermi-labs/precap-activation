# Depth-64-Diagnose aus v4

## Zeitliches Muster

### radial_ripple
- Schritt 0 vs silu: log10-Testquotient +0.009, Gradienten-Verhältnis nan, Peak-Verhältnis nan.
- Schritt 100 vs silu: log10-Testquotient -0.017, Gradienten-Verhältnis 1.256, Peak-Verhältnis 1.457.
- Schritt 500 vs silu: log10-Testquotient -0.150, Gradienten-Verhältnis 1.765, Peak-Verhältnis 1.062.
- Schritt 1500 vs silu: log10-Testquotient -0.250, Gradienten-Verhältnis 3.210, Peak-Verhältnis 0.668.
- Schritt 3000 vs silu: log10-Testquotient +0.086, Gradienten-Verhältnis 1.794, Peak-Verhältnis 0.668.
- Schritt 0 vs gelu: log10-Testquotient -0.001, Gradienten-Verhältnis nan, Peak-Verhältnis nan.
- Schritt 100 vs gelu: log10-Testquotient +0.069, Gradienten-Verhältnis 0.791, Peak-Verhältnis 0.338.
- Schritt 500 vs gelu: log10-Testquotient +0.143, Gradienten-Verhältnis 1.560, Peak-Verhältnis 0.763.
- Schritt 1500 vs gelu: log10-Testquotient -0.519, Gradienten-Verhältnis 2.152, Peak-Verhältnis 0.596.
- Schritt 3000 vs gelu: log10-Testquotient +0.224, Gradienten-Verhältnis 2.083, Peak-Verhältnis 0.596.

### ridge_mix
- Schritt 0 vs silu: log10-Testquotient -0.001, Gradienten-Verhältnis nan, Peak-Verhältnis nan.
- Schritt 100 vs silu: log10-Testquotient -0.015, Gradienten-Verhältnis 1.187, Peak-Verhältnis 1.387.
- Schritt 500 vs silu: log10-Testquotient +0.037, Gradienten-Verhältnis 1.025, Peak-Verhältnis 0.631.
- Schritt 1500 vs silu: log10-Testquotient -0.280, Gradienten-Verhältnis 0.886, Peak-Verhältnis 0.493.
- Schritt 3000 vs silu: log10-Testquotient -0.145, Gradienten-Verhältnis 0.935, Peak-Verhältnis 0.478.
- Schritt 0 vs gelu: log10-Testquotient -0.007, Gradienten-Verhältnis nan, Peak-Verhältnis nan.
- Schritt 100 vs gelu: log10-Testquotient -0.004, Gradienten-Verhältnis 0.943, Peak-Verhältnis 0.830.
- Schritt 500 vs gelu: log10-Testquotient +0.233, Gradienten-Verhältnis 0.779, Peak-Verhältnis 1.051.
- Schritt 1500 vs gelu: log10-Testquotient -0.013, Gradienten-Verhältnis 1.661, Peak-Verhältnis 0.881.
- Schritt 3000 vs gelu: log10-Testquotient +0.129, Gradienten-Verhältnis 1.119, Peak-Verhältnis 0.826.

## Vorläufige Lesart

- `radial_ripple` bei 64 zeigt ein auffälliges Muster: Zwischen Schritt 1500 und 3000 kippt der Testvorteil zurück, obwohl die Peaks klein bleiben.
- `ridge_mix` bei 64 ist gemischt und referenzabhängig: gegen SiLU bleibt PreCap positiv, gegen GELU nicht.
- Damit ist die nächste Studie am besten als kleine, worker-parallele Tiefe-64-Diagnose mit dichteren Checkpoints angelegt.
