# CapExp v4 — Mechanismusauswertung (deskriptiv)

Negative log10-Testquotienten begünstigen PreCap. Gradienten- und Peak-Verhältnisse unter 1 bedeuten geringere Werte bei PreCap.

## Endpunkt bei Schritt 3000

### radial_ripple, Tiefe 16
- Gegen silu: log10(Test-MSE-Quotient) -0.268; Gradienten-Verhältnis 0.565; Peak-Verhältnis 0.624; Cap-Anteil 0.0098.
- Gegen gelu: log10(Test-MSE-Quotient) -0.205; Gradienten-Verhältnis 1.232; Peak-Verhältnis 0.644; Cap-Anteil 0.0098.

### angled_kink, Tiefe 32
- Gegen silu: log10(Test-MSE-Quotient) -0.135; Gradienten-Verhältnis 1.794; Peak-Verhältnis 0.548; Cap-Anteil 0.0060.
- Gegen gelu: log10(Test-MSE-Quotient) -0.212; Gradienten-Verhältnis 1.020; Peak-Verhältnis 1.087; Cap-Anteil 0.0060.

### radial_ripple, Tiefe 64
- Gegen silu: log10(Test-MSE-Quotient) +0.086; Gradienten-Verhältnis 1.794; Peak-Verhältnis 0.668; Cap-Anteil 0.0052.
- Gegen gelu: log10(Test-MSE-Quotient) +0.224; Gradienten-Verhältnis 2.083; Peak-Verhältnis 0.596; Cap-Anteil 0.0052.

### ridge_mix, Tiefe 64
- Gegen silu: log10(Test-MSE-Quotient) -0.145; Gradienten-Verhältnis 0.935; Peak-Verhältnis 0.478; Cap-Anteil 0.0168.
- Gegen gelu: log10(Test-MSE-Quotient) +0.129; Gradienten-Verhältnis 1.119; Peak-Verhältnis 0.826; Cap-Anteil 0.0168.

### highfreq_grid, Tiefe 32
- Gegen silu: log10(Test-MSE-Quotient) +0.012; Gradienten-Verhältnis 1.683; Peak-Verhältnis 0.384; Cap-Anteil 0.0085.
- Gegen gelu: log10(Test-MSE-Quotient) +0.011; Gradienten-Verhältnis 1.856; Peak-Verhältnis 0.626; Cap-Anteil 0.0085.

## Zeitliche Diagnose

Die Datei `paired_temporal_effects.csv` enthält dieselben gepaarten Kennzahlen bei Schritt 0, 100, 500, 1500 und 3000. Sie erlaubt die Prüfung, ob Unterschiede in Gradient oder Peak vor dem Testfehler-Effekt sichtbar werden.

## Vorsichtige Hypothesenbewertung

Diese Studie ist diagnostisch und deskriptiv. Sie kann zeitliche und layerweise Muster zeigen, aber keine kausale Wirkung von Aktivierungs- oder Gradientenunterschieden beweisen.
