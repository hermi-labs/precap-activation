# V13 Praeregistrierung (ENTWURF, nicht eingefroren)

## Anlass
V12 zeigte: s225 hilft ohne Drehung und bei niedriger Frequenz, verliert aber bei hoher Frequenz (k=1.5) den Vorteil; bei Tiefe 32 kehrt er sich um (Verhaeltnis 1.50). Dieser Befund war explorativ (alle 6 Holm-Kontraste n.s.). Der Anker bei Tiefe 32 schwankte zudem zwischen V11 (0.98) und V12 (0.70). V13 untersucht gezielt die Frequenzachse mit mehr Seeds.

## Frage
Nimmt der Vorteil von precap_s225 gegenueber precap_base monoton mit der Frequenz k ab?

## Design
- Aktivierungen: precap_base (s=1.5), precap_s225 (s=2.25). s300 entfaellt (V12: keine konsistente Dosiswirkung).
- Aufgaben: rot_ang53_f{070,100,125,150,200}_a025, d.h. k = 0.7, 1.0, 1.25, 1.5, 2.0 (Drehung 53 Grad, a=0.25 fix). Funktion identisch zu V12; k = f/100.
- Tiefen: 16 und 32. Breite 64, 1200 Schritte, Adam lr 1e-3, Batch 128, gleiche Checkpoints wie V12.
- Seeds: 8000-8029 (30 neu; Test-Seeds +10000). Disjunkt zu V10 (5000/6000) und V12 (7000).
- Umfang: 5 x 2 x 30 x 2 = 600 Runs, 4200 Checkpoint-Zeilen. Geschaetzte Dauer ca. 3-3.5 h mit 6 Workern.

## Metrik
d = log10(test_MSE_s225 / test_MSE_base) bei Schritt 1200, pro Seed. d<0 heisst s225 besser.

## Konfirmatorisch (2 Tests, Holm)
Fuer jede Tiefe und jeden Seed wird die Steigung von d ueber log10(k) (5 Punkte, lineare Regression) berechnet. Test: Wilcoxon einseitig, Steigung > 0 (Vorteil nimmt mit k ab). Familie = 2 Tiefen, Holm bei alpha 0.05.

## Explorativ
- Zellmediane, Seeds besser, einseitige Tests je Zelle.
- Paarkontraste k=1.5 und k=2.0 gegen k=1.0 je Tiefe (zweiseitig, ohne Korrektur).
- Interpolierte Kreuzungsfrequenz (Median d = 0) je Tiefe.
- Spearman Zellmedian d gegen k je Tiefe.

## Interpretation vorab
- Beide Steigungstests signifikant: frequenzabhaengige Grenze gestuetzt; s225 nur unterhalb der Kreuzungsfrequenz empfehlen.
- Nur eine Tiefe signifikant: tiefenabhaengig, weiter explorativ.
- Keiner signifikant: kein belastbarer Frequenzeffekt; base bleibt Standard.
- Keine Aussage ueber Mechanismen (V12: p99-Zusammenhang null).

## Eingefrorene Dateien (nach Freigabe)
run_v13_frequency.py, analyze_v13.py, launch_v13.sh, status.py, docs/V13_Praeregistrierung.md
