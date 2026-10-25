# Fahrplan nach V8

1. V9 — Geometrie-Mechanismus gezielt testen: Faktorielle Aufgabenfamilien mit kontrollierter Radialität/Anisotropie statt weiterer willkürlicher Funktionen. Parametrische Interpolation zwischen radialer und ridge-artiger Frequenzstruktur, d16/d32, base vs s225, mindestens 20 neue Seeds.
2. Primärmetrik V9: kontinuierliche Geometrie-Kennzahl (radiale Energie bzw. Richtungsanisotropie) gegen Effekt d; nicht nur Gruppenlabels. Vorab Regression bzw. Rangkorrelation festlegen.
3. Snapshot-Mechanismus parallel: Aktivierungs- und Gradientenverteilungen für radial_bessel und ridge_highfreq, weil sie in V8 die klarsten Gegensätze liefern.
4. Keine adaptive Regel und keine globale Umstellung auf s225 vor V9. Base bleibt Standard; s225 ist eine gezielte Option für bestätigte radiale Aufgaben.
