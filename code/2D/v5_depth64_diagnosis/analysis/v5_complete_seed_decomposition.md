# v5 — Vollständige Seed-Zerlegung

Alle Aussagen beruhen auf gepaarten Runs mit identischem Seed. Negative Differenzen begünstigen PreCap. Der Generalisierungsabstand ist Test-MSE minus Train-MSE: kleiner ist hier nicht automatisch besser, sondern muss zusammen mit absolutem Testfehler gelesen werden.

## Klassifikation der Trainingspfade

### highfreq_grid, Tiefe 32, gegen gelu
- Später Vorsprung verloren: 2/5; Kein stabiler Vorsprung: 2/5; Vorsprung gehalten/erholt: 1/5.
- Seed 800: Später Vorsprung verloren; End-Δ Test=+0.000314826, End-Δ Train=+0.000251211, End-Δ Gap=+6.36154e-05.
- Seed 801: Kein stabiler Vorsprung; End-Δ Test=+0.000991695, End-Δ Train=+0.000882648, End-Δ Gap=+0.000109047.
- Seed 802: Kein stabiler Vorsprung; End-Δ Test=+0.000487645, End-Δ Train=+0.000976296, End-Δ Gap=-0.000488651.
- Seed 803: Später Vorsprung verloren; End-Δ Test=+0.000618934, End-Δ Train=+0.000565114, End-Δ Gap=+5.38193e-05.
- Seed 804: Vorsprung gehalten/erholt; End-Δ Test=-0.000119278, End-Δ Train=+0.00010687, End-Δ Gap=-0.000226148.

### highfreq_grid, Tiefe 32, gegen silu
- Später Vorsprung verloren: 2/5; Kein stabiler Vorsprung: 2/5; Vorsprung gehalten/erholt: 1/5.
- Seed 800: Später Vorsprung verloren; End-Δ Test=+9.88338e-05, End-Δ Train=+2.02795e-05, End-Δ Gap=+7.85543e-05.
- Seed 801: Kein stabiler Vorsprung; End-Δ Test=+0.000645973, End-Δ Train=+0.000517025, End-Δ Gap=+0.000128948.
- Seed 802: Kein stabiler Vorsprung; End-Δ Test=+0.000562501, End-Δ Train=+0.00117898, End-Δ Gap=-0.000616477.
- Seed 803: Später Vorsprung verloren; End-Δ Test=+0.000568725, End-Δ Train=+0.000462891, End-Δ Gap=+0.000105835.
- Seed 804: Vorsprung gehalten/erholt; End-Δ Test=-0.00014543, End-Δ Train=+0.000146204, End-Δ Gap=-0.000291633.

### radial_ripple, Tiefe 16, gegen gelu
- Vorsprung gehalten/erholt: 5/5.
- Seed 800: Vorsprung gehalten/erholt; End-Δ Test=-0.00497974, End-Δ Train=-0.00398656, End-Δ Gap=-0.000993181.
- Seed 801: Vorsprung gehalten/erholt; End-Δ Test=-0.00129663, End-Δ Train=-0.00163393, End-Δ Gap=+0.000337297.
- Seed 802: Vorsprung gehalten/erholt; End-Δ Test=-0.000553029, End-Δ Train=-0.000915883, End-Δ Gap=+0.000362854.
- Seed 803: Vorsprung gehalten/erholt; End-Δ Test=-0.000709775, End-Δ Train=-0.000372191, End-Δ Gap=-0.000337585.
- Seed 804: Vorsprung gehalten/erholt; End-Δ Test=-0.000379825, End-Δ Train=-0.00057666, End-Δ Gap=+0.000196834.

### radial_ripple, Tiefe 16, gegen silu
- Vorsprung gehalten/erholt: 3/5; Später Vorsprung verloren: 2/5.
- Seed 800: Später Vorsprung verloren; End-Δ Test=+6.1253e-05, End-Δ Train=-1.60651e-05, End-Δ Gap=+7.73181e-05.
- Seed 801: Vorsprung gehalten/erholt; End-Δ Test=-0.0018416, End-Δ Train=-0.00220746, End-Δ Gap=+0.000365867.
- Seed 802: Vorsprung gehalten/erholt; End-Δ Test=-0.00205559, End-Δ Train=-0.00190787, End-Δ Gap=-0.000147722.
- Seed 803: Vorsprung gehalten/erholt; End-Δ Test=-0.000537503, End-Δ Train=-0.000310948, End-Δ Gap=-0.000226555.
- Seed 804: Später Vorsprung verloren; End-Δ Test=+6.01555e-05, End-Δ Train=+3.98341e-05, End-Δ Gap=+2.03214e-05.

### radial_ripple, Tiefe 64, gegen gelu
- Vorsprung gehalten/erholt: 3/5; Später Vorsprung verloren: 2/5.
- Seed 800: Vorsprung gehalten/erholt; End-Δ Test=-0.00176822, End-Δ Train=-0.00156171, End-Δ Gap=-0.000206514.
- Seed 801: Vorsprung gehalten/erholt; End-Δ Test=-0.00041263, End-Δ Train=-0.000528262, End-Δ Gap=+0.000115632.
- Seed 802: Vorsprung gehalten/erholt; End-Δ Test=-0.001932, End-Δ Train=-0.00167243, End-Δ Gap=-0.000259573.
- Seed 803: Später Vorsprung verloren; End-Δ Test=+0.000265854, End-Δ Train=+7.30286e-05, End-Δ Gap=+0.000192825.
- Seed 804: Später Vorsprung verloren; End-Δ Test=+0.00524027, End-Δ Train=+0.00582849, End-Δ Gap=-0.000588215.

### radial_ripple, Tiefe 64, gegen silu
- Später Vorsprung verloren: 3/5; Vorsprung gehalten/erholt: 2/5.
- Seed 800: Vorsprung gehalten/erholt; End-Δ Test=-0.000418883, End-Δ Train=-0.000253221, End-Δ Gap=-0.000165662.
- Seed 801: Später Vorsprung verloren; End-Δ Test=+0.000382801, End-Δ Train=+0.000330467, End-Δ Gap=+5.23338e-05.
- Seed 802: Vorsprung gehalten/erholt; End-Δ Test=-0.00947786, End-Δ Train=-0.00872, End-Δ Gap=-0.000757854.
- Seed 803: Später Vorsprung verloren; End-Δ Test=+0.000496357, End-Δ Train=+0.000219698, End-Δ Gap=+0.000276659.
- Seed 804: Später Vorsprung verloren; End-Δ Test=+0.00440139, End-Δ Train=+0.0050375, End-Δ Gap=-0.000636111.

### ridge_mix, Tiefe 64, gegen gelu
- Später Vorsprung verloren: 3/5; Vorsprung gehalten/erholt: 2/5.
- Seed 800: Später Vorsprung verloren; End-Δ Test=+0.000841573, End-Δ Train=+0.000878406, End-Δ Gap=-3.68338e-05.
- Seed 801: Vorsprung gehalten/erholt; End-Δ Test=-0.000339612, End-Δ Train=-0.000366167, End-Δ Gap=+2.65551e-05.
- Seed 802: Vorsprung gehalten/erholt; End-Δ Test=-0.00222321, End-Δ Train=-0.00199232, End-Δ Gap=-0.000230893.
- Seed 803: Später Vorsprung verloren; End-Δ Test=+0.0010291, End-Δ Train=+0.00101297, End-Δ Gap=+1.61238e-05.
- Seed 804: Später Vorsprung verloren; End-Δ Test=+0.0001197, End-Δ Train=+1.67378e-05, End-Δ Gap=+0.000102962.

### ridge_mix, Tiefe 64, gegen silu
- Vorsprung gehalten/erholt: 3/5; Später Vorsprung verloren: 2/5.
- Seed 800: Später Vorsprung verloren; End-Δ Test=+0.000695327, End-Δ Train=+0.000738173, End-Δ Gap=-4.2847e-05.
- Seed 801: Vorsprung gehalten/erholt; End-Δ Test=-0.000799689, End-Δ Train=-0.000837817, End-Δ Gap=+3.81279e-05.
- Seed 802: Vorsprung gehalten/erholt; End-Δ Test=-0.00131953, End-Δ Train=-0.00118585, End-Δ Gap=-0.000133674.
- Seed 803: Vorsprung gehalten/erholt; End-Δ Test=-0.00817858, End-Δ Train=-0.00893886, End-Δ Gap=+0.000760286.
- Seed 804: Später Vorsprung verloren; End-Δ Test=+0.000317868, End-Δ Train=+0.000209187, End-Δ Gap=+0.000108682.

## Richtige Lesart der Formel

- PreCap ist keine starre Regelung, die immer "zu hart" oder immer "zu weich" ist. Ihre CapExp-Komponente wird oberhalb des Caps linear statt exponentiell; dadurch verhindert sie exponentielle Verstärkung, schneidet aber nicht hart ab.
- Der `tanh`-Teil begrenzt den Eingang vor CapExp weich. Zusammen ergibt das eine glatte, aktivitätsbegrenzende Form mit weiterhin vorhandener Steigung.
- Ob das passend ist, hängt mit Aufgabe, Tiefe, Initialisierung, Optimierungspfad und Referenzaktivierung zusammen. Die Daten stützen deshalb ein Zusammenkommen mehrerer Faktoren statt eines einzelnen Fehlers.

## Was v5 leisten kann und was nicht

- v5 zeigt, ob sich Test- und Trainingsfehler innerhalb eines Seeds unterschiedlich bewegen und ob der späte Wechsel von PreCap selbst oder von der Referenz kommt.
- v5 beweist keinen einzelnen kausalen Mechanismus, weil die Gradientenspeicherung nur punktuell erfolgt und keine gezielte Ablation der Formelparameter durchgeführt wurde.
