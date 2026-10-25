# V11 Praeregistrierung (Bestaetigungsstudie)

Ziel: Bestaetigen, dass der V10-Befund (s225 hilft bei gemischter Geometrie a=0.25, nicht bei a=0.75) auf neuen Seeds und einer unabhaengigen Aufgabenfamilie gilt, und den V10-Mechanismus (niedrigere Pre-Activation-p99) pruefen.

## Design
- Aktivierung: PreCap, u=z+0.01 z^2, v=s*tanh(u/s), CapExp mit c=1; base s=1.5, s225 s=2.25. Residual h=skip+phi/sqrt(L).
- Aufgaben: geo_a025 (V10-Anker), rot_a025, rot_a075 (rotierte Ridge-Richtung, andere Frequenzen/Amplitude).
- Tiefen 16, 32. Seeds 6000-6019 (Test-Seed +10000). Kein Overlap mit V10.
- 240 Runs, 1680 Checkpoint-Zeilen, Checkpoints 0,50,100,200,400,800,1200. Width 64, Batch 128, Adam lr 1e-3, 1200 Schritte.
- hmax_peak = laufendes Maximum von max|f(x)| (Ausgabespitze), nicht Hidden-Maximum.

## Hypothesen (fest, vor Datenerhebung)
d = log10(test_MSE_s225 / test_MSE_base) bei Step 1200, pro Seed gepaart. Erfolgszellen: geo_a025 und rot_a025 (je Tiefe 16, 32 = 4 Zellen). Gegenzellen: rot_a075 (2 Zellen).
- H1: In allen 4 Erfolgszellen d<0 (einseitiger Wilcoxon, Holm ueber 4 Tests, alpha 0.05) und Median d<0.
- H2: In den Gegenzellen Median d >= -0.05 (kein relevanter Vorteil).
- H3 (primaerer Mechanismus): Delta pre-activation p99 (s225 minus base, Step 1200, gesamt ueber Schichten) <0 in mindestens 3 von 4 Erfolgszellen (einseitiger Wilcoxon, Holm ueber 4) und Median-Delta >0 in beiden Gegenzellen.
- Sekundaer: Ausgabespitze (hmax_peak) gleiche Logik; zentrierte Spearman-Korrelation d vs Delta-Metrik (in V10 nicht gezeigt, nur explorativ); Kappungsanteil, Gradienten-Norm, Schicht-p99 (erste/mitte/letzte).

## Entscheidungslogik
- H1 und H2 erfuellt: s225 als Spezialvariante fuer gemischte Geometrie bestaetigt (kein globaler Standard).
- H3 zusaetzlich erfuellt: p99-Mechanismus als Kandidat gestuetzt; Kausalitaet bleibt unbewiesen (nur Korrelation auf Zellebene).
- H1 nicht erfuellt: V10-Befund nicht bestaetigt; base bleibt Standard.
- H1 erfuellt, aber rot_a025 nicht: Befund ist an die geo-Familie gebunden.
- Keine Aenderungen an Code, Hypothesen oder Parametern nach Start. Abweichungen werden als explorativ markiert.

## Bekannte Grenzen
Die Hypothesenwahl (p99 statt Ausgabespitze) folgt aus V10-Daten, dem Quell-Datensatz. Seed-Korrelation in V10 mit falschem Vorzeichen; keine Aussage auf Seed-Ebene.
