# CapExp 2D-Pilot — Manifestentwurf vor Start

## Ziel

Prüfen, ob der 1D-Befund — PreCap besser als SiLU/GELU in residualer Architektur — auf eine kontrollierte 2D-Regressionsfamilie übertragbar ist. Dies ist ein Pilot, keine Generalisierungsbehauptung.

## Kandidaten

- PreCap: eingefrorene 1D-Referenz (c=1,0; a=1,25; gamma=0,01; soft_s=1,5; Knie=0,75).
- Referenzen: SiLU und GELU.
- Keine gewichteten oder hybriden Varianten.

## Design vor Start

- Architekturen: primär residuale 2D-MLPs; `plain` nur als klar getrennte negative Kontrollbedingung, nicht als pooled Haupteffekt.
- Gleiche Breite, Parameterbudget und Trainingsschritte je Aktivierung.
- Aufgabenfamilien: mindestens eine glatte 2D-Funktion, eine oszillatorisch-gemischte Funktion und eine Funktion mit Knick/hoher lokaler Krümmung.
- Tiefen: 4, 8, 16, 32; frische Seeds; pro Zelle ausreichende Seed-Zahl vorab festlegen.
- Metriken: Test-MSE, gepaarter Log-Quotient d=log10(MSE_PreCap/MSE_Referenz), Seed-Siege, bad-Rate, hmax_peak, Gradientennormen, Cap-Anteil und Laufzeit.

## Gate für den Pilot

Vor dem Start numerisch ausfüllen:

1. Kein PreCap-bedingter Anstieg der bad-Rate gegenüber beiden Referenzen.
2. Mindestens zwei von drei 2D-Aufgaben zeigen bei `res_inv` einen Median d<0 gegen beide Referenzen.
3. In Tiefe 16/32 erfüllt PreCap mindestens eine Stabilitätsbedingung (niedrigerer hmax_peak oder weniger Ausfälle) ohne Test-MSE-Verschlechterung über Faktor 1,5.

## Nach dem Pilot

- PASS: unabhängiger 2D-Holdout mit neuen Seeds und fixierten Zellen.
- FAIL: 2D nicht aufblasen; den 1D-Befund sauber mit seiner Architekturgrenze berichten.
