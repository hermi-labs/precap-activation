# v5 — Per-Seed-Analyse und "Warum"

Interpretation: Ein negativer Unterschied bedeutet kleineren Test-MSE von PreCap. Ein Quotient unter 1 bedeutet ebenfalls besseren Test-MSE von PreCap. Alle Vergleiche sind innerhalb desselben Seeds gepaart.

## Radial ripple, Tiefe 64 — Umschwung je Seed

### Gegen silu
- Seed 800: 800: Δ=-0.14, Grad×=2.10, Peak×=0.57; 1200: Δ=-0.0227, Grad×=0.98, Peak×=0.54; 1500: Δ=-0.00528, Grad×=0.72, Peak×=0.54; 2200: Δ=+0.0133, Grad×=4.63, Peak×=0.54; 3000: Δ=-0.000419, Grad×=0.85, Peak×=0.54.
- Seed 801: 800: Δ=-0.00598, Grad×=2.52, Peak×=0.46; 1200: Δ=-0.00457, Grad×=0.31, Peak×=0.46; 1500: Δ=+0.00353, Grad×=1.49, Peak×=0.46; 2200: Δ=+6.38e-05, Grad×=1.12, Peak×=0.46; 3000: Δ=+0.000383, Grad×=2.96, Peak×=0.46.
- Seed 802: 800: Δ=-0.00456, Grad×=1.66, Peak×=1.05; 1200: Δ=-0.00773, Grad×=2.83, Peak×=1.05; 1500: Δ=+0.00474, Grad×=1.22, Peak×=1.05; 2200: Δ=-0.000173, Grad×=0.46, Peak×=1.05; 3000: Δ=-0.00948, Grad×=0.19, Peak×=1.05.
- Seed 803: 800: Δ=-0.0231, Grad×=5.32, Peak×=0.33; 1200: Δ=-0.0226, Grad×=3.18, Peak×=0.33; 1500: Δ=-0.0039, Grad×=0.48, Peak×=0.33; 2200: Δ=+0.00024, Grad×=1.46, Peak×=0.33; 3000: Δ=+0.000496, Grad×=3.98, Peak×=0.33.
- Seed 804: 800: Δ=-0.00509, Grad×=1.05, Peak×=0.39; 1200: Δ=-0.00649, Grad×=0.56, Peak×=0.39; 1500: Δ=+0.000728, Grad×=2.31, Peak×=0.39; 2200: Δ=+8.51e-05, Grad×=2.00, Peak×=0.39; 3000: Δ=+0.0044, Grad×=0.94, Peak×=0.39.

### Gegen gelu
- Seed 800: 800: Δ=+0.00177, Grad×=2.18, Peak×=1.69; 1200: Δ=-0.00174, Grad×=1.21, Peak×=1.69; 1500: Δ=+2.79e-05, Grad×=1.41, Peak×=1.69; 2200: Δ=+0.0144, Grad×=1.73, Peak×=1.67; 3000: Δ=-0.00177, Grad×=1.70, Peak×=1.62.
- Seed 801: 800: Δ=-0.0031, Grad×=2.02, Peak×=0.68; 1200: Δ=-0.00125, Grad×=1.52, Peak×=0.68; 1500: Δ=+0.000643, Grad×=1.46, Peak×=0.68; 2200: Δ=-0.000335, Grad×=3.07, Peak×=0.68; 3000: Δ=-0.000413, Grad×=0.78, Peak×=0.68.
- Seed 802: 800: Δ=-0.0143, Grad×=1.23, Peak×=0.48; 1200: Δ=-0.00824, Grad×=1.57, Peak×=0.48; 1500: Δ=+0.000585, Grad×=0.89, Peak×=0.48; 2200: Δ=-0.00011, Grad×=0.83, Peak×=0.48; 3000: Δ=-0.00193, Grad×=0.47, Peak×=0.48.
- Seed 803: 800: Δ=-0.00571, Grad×=3.93, Peak×=0.87; 1200: Δ=-0.00488, Grad×=4.09, Peak×=0.87; 1500: Δ=-0.00142, Grad×=1.29, Peak×=0.87; 2200: Δ=-0.000413, Grad×=0.79, Peak×=0.87; 3000: Δ=+0.000266, Grad×=5.11, Peak×=0.86.
- Seed 804: 800: Δ=-0.00713, Grad×=4.43, Peak×=0.90; 1200: Δ=-0.00137, Grad×=0.39, Peak×=0.90; 1500: Δ=+0.00187, Grad×=3.38, Peak×=0.90; 2200: Δ=-0.00131, Grad×=3.54, Peak×=0.90; 3000: Δ=+0.00524, Grad×=1.82, Peak×=0.90.

## Endpunkte nach Seed

### radial_ripple, Tiefe 64
- Gegen silu: PreCap gewinnt 2/5 Seeds; medianer absoluter Δ-Test-MSE=+0.000382801; medianer MSE-Quotient=3.360.
- Gegen gelu: PreCap gewinnt 3/5 Seeds; medianer absoluter Δ-Test-MSE=-0.00041263; medianer MSE-Quotient=0.547.

### ridge_mix, Tiefe 64
- Gegen silu: PreCap gewinnt 3/5 Seeds; medianer absoluter Δ-Test-MSE=-0.000799689; medianer MSE-Quotient=0.207.
- Gegen gelu: PreCap gewinnt 2/5 Seeds; medianer absoluter Δ-Test-MSE=+0.0001197; medianer MSE-Quotient=1.188.

### radial_ripple, Tiefe 16
- Gegen silu: PreCap gewinnt 3/5 Seeds; medianer absoluter Δ-Test-MSE=-0.000537503; medianer MSE-Quotient=0.552.
- Gegen gelu: PreCap gewinnt 5/5 Seeds; medianer absoluter Δ-Test-MSE=-0.000709775; medianer MSE-Quotient=0.164.

### highfreq_grid, Tiefe 32
- Gegen silu: PreCap gewinnt 1/5 Seeds; medianer absoluter Δ-Test-MSE=+0.000562501; medianer MSE-Quotient=1.067.
- Gegen gelu: PreCap gewinnt 1/5 Seeds; medianer absoluter Δ-Test-MSE=+0.000487645; medianer MSE-Quotient=1.061.

## Was die Daten bisher tragen

- PreCap ist nicht "schlecht". Es besitzt einen wiederkehrenden deutlichen Früh- bis Mittellernvorteil auf radial_ripple und funktioniert stabil beim Tiefe-16-Anker.
- Das Problem ist vielmehr die Konsistenz des späten Endpunkts in bestimmten Tiefe-64-/Referenz-Paaren; dieser Umschwung ist seedabhängig und nicht durch einen einzigen universellen Gradientenmarker erklärt.
- Ein niedrigerer laufender Aktivierungspeak ist häufig vorhanden, aber weder notwendig noch hinreichend für besseren End-Test-MSE.

## Grenzen

- Die gespeicherte globale Gradientennorm stammt nur vom letzten Minibatch eines Zeitpunkts. Sie ist eine Diagnose, kein stabiler Schätzer eines ganzen Trainingsfensters.
- Die Daten zeigen Korrelationen und zeitliche Muster, aber keine kausale Mechanismusbestätigung.
