# V18 – Ausführliche Auswertung (Lernraten-fairer Vergleich in flachen Netzen)

Stand: 25.10.2026, 20:25 Uhr. Daten: 4050/4050 Runs, 32400/32400 Zeilen, 0 nicht-endlich. Integrität ok, eingefrorene Hashes ok.

## Betriebsvorfall (dokumentiert)
Um 15:01 und 15:12 Uhr liefen zwei Starter gleichzeitig, also 14 Worker. Sie haben dieselben Runs doppelt gerechnet. Alle 14416 doppelten Zeilen waren bitgenau identisch. Um 18:41 Uhr wurde der zweite Satz beendet, der erste um 18:46 Uhr sauber neu gestartet. Der Satz von 18:46 Uhr stoppte gegen 18:51 Uhr, wahrscheinlich weil das Terminal-Fenster geschlossen wurde; um 19:14 Uhr wurde neu gestartet (PyCharm). Nach Laufende wurden die Rohdaten nach meta/workers_raw_with_duplicates gesichert, die identischen Duplikate entfernt (genau 32400 Zeilen übrig) und Integrität sowie Analyse mit dem unveränderten eingefrorenen Code neu ausgeführt. Die automatische Analyse mit Duplikaten (meta/analysis_autorun_with_duplicates) kam zum gleichen Urteil.

## Konfirmatorisch (LR je Aufgabe × Tiefe × Aktivierung auf Tuning-Seeds gewählt)
- Gewählte LR: SiLU und GELU in 15/15 Zellen 3e-3 (oberer Rand des Rasters). PreCap 9× 3e-3, 6× 1e-3 (vor allem Tiefe 32).
- 30 Tests, Holm: 0× PreCap besser, 2× PreCap schlechter (smooth_bowl Tiefe 16 vs SiLU, Ratio 2,74; corner_kink Tiefe 32 vs GELU, Ratio 2,55), 28× unklar.
- H_A (mindestens 8/15 besser, höchstens 1 schlechter) für SiLU und GELU: nicht bestätigt.
- Tendenz ohne Signifikanz: Tiefe 8 meist Ratio < 1 (0,5–1,0), Tiefe 16/32 gemischt.

## Explorativ: feste LR 1e-3 (altes Protokoll aus V14/V16/V17)
- PreCap ist hier fast überall klar besser: Ratio 0,03–0,7, oft 18–20 von 20 Seeds. Der frühere Befund repliziert also exakt.
- Median log10-MSE über alle Aufgaben (LR-Sensitivität): bei 1e-3 ist PreCap um 0,3–0,6 besser; SiLU und GELU holen bei 3e-3 fast vollständig auf.

## Interpretation
Der flache PreCap-Vorteil aus V14/V16 war zum größten Teil ein Lernraten-Effekt. PreCap hat am Ursprung Steigung 1, SiLU und GELU nur 0,5 (Taylor: PreCap ≈ x + 0,51x², SiLU ≈ x/2 + x²/4). Bei gleicher LR lernt PreCap dadurch effektiv schneller. Bekommen SiLU und GELU ihre eigene, höhere LR, verschwindet der Vorteil. Das passt zur Taylor-Überlegung, dass PreCap bei kleinen Werten fast ein umskaliertes SiLU ist.

## Mechanismus (präregistriert)
Spearman(Anteil |pre| > 0,5; log-Ratio vs SiLU) = −0,52, einseitig p = 0,023, n = 15: gestützt. Wo mehr Voraktivierungen im Bereich liegen, in dem sich PreCap von SiLU unterscheidet, schneidet PreCap relativ besser ab. Einschränkung: Der Anteil sinkt mit der Tiefe, die Korrelation ist also teilweise mit der Tiefe vermischt.

## Einschränkungen
- 3e-3 ist für SiLU/GELU der Rand des Rasters; mit 1e-2 könnten sie noch besser werden. Das Ergebnis ist also eher noch günstig für PreCap.
- 20 Bestätigungs-Seeds pro Zelle, 30 Tests mit Holm: kleine Effekte (Faktor < 1,5) sind kaum nachweisbar.

## Folgerungen
1. Kein belegbarer allgemeiner Vorteil, auch nicht flach, wenn die Lernrate fair gewählt wird.
2. Fairer Vergleich braucht künftig immer LR-Tuning je Aktivierung und ein Raster bis mindestens 1e-2.
3. Naheliegender nächster Test: SiLU × 2 (gleiche Steigung am Ursprung wie PreCap) als Referenz. Wenn das gleich gut ist, ist PreCaps Vorteil reine Skalierung.
4. Für 3D-V1: LR-Raster auf 1e-3/3e-3/1e-2 verschieben und die LayerNorm-Frage in den Mittelpunkt stellen. Mit LayerNorm liegen 30 % statt 3 % der Werte über 0,5, also genau in PreCaps eigenem Bereich.

Konfidenz: Der flache Vorteil war überwiegend ein LR-Effekt: etwa 0,85.
