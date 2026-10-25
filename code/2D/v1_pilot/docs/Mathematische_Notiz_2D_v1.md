# Mathematische Notiz — CapExp 2D v1 Pilot

## Ziel

Der 2D-Pilot prüft, ob der in 1D beobachtete Vorteil von PreCap in residualen Architekturen auf einfache 2D-Regressionsaufgaben übertragbar ist. Getestet werden nur PreCap, SiLU und GELU, damit die Frage klar bleibt.

## Vergleichsgröße

Für jede identische Zelle (Aufgabe, Tiefe, Seed) wird

\[
d = \log_{10}\!\left(\frac{\mathrm{MSE}_{\mathrm{PreCap}}}{\mathrm{MSE}_{\mathrm{Ref}}}\right)
\]

berechnet. Hier bedeutet \(d<0\), dass PreCap besser ist. Die logarithmische Form ist passend, weil MSE-Werte typischerweise über Größenordnungen streuen.

## Modellidee

Das Netz ist residual aufgebaut:

\[
h_{\ell+1} = h_{\ell} + \phi(W_{\ell} h_{\ell} + b_{\ell})
\]

mit \(\phi\) als Aktivierung. Für PreCap wird eine gekappte SiLU-artige Aktivierung verwendet, die große Aktivierungen weich zurückdrängt. Dadurch soll bei größerer Tiefe die Aktivierungsamplitude kontrolliert werden, ohne kleine Signale abzuschneiden.

## 2D-Aufgaben

- `radial_ripple`: radial oszillierende glatte Struktur.
- `ridge_mix`: gemischte periodische und quadratische Krümmung.
- `corner_kink`: nichtglatte Ecke/Knick-Struktur.

Diese drei Aufgaben decken unterschiedliche lokale Geometrien ab: glatte Krümmung, anisotrope Ridges und stückweise Glattheit.

## Pilot-Gedanke

Wenn PreCap wirklich über Aktivierungskontrolle und Residualdynamik hilft, dann sollte der Effekt in 2D vor allem dort sichtbar werden, wo die Aufgabe gleichzeitig Krümmung und größere Tiefe verlangt. Ein negatives Ergebnis ist ebenfalls informativ: Dann ist der 1D-Befund enger an seine ursprüngliche Aufgabenfamilie gebunden als erhofft.

## Bewertungsregel

Der Pilot gilt als erfolgreich, wenn PreCap bei mindestens zwei der drei 2D-Aufgaben einen negativen Median von \(d\) gegen SiLU und GELU zeigt und dabei keine erhöhte Ausfallrate verursacht. Zusätzlich werden `hmax_peak`, Endgradient und Laufzeit dokumentiert.
