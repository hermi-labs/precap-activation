# CapExp — aktualisierter Fahrplan nach v11

## Stand heute

1. **v10B Holdout abgeschlossen:** Die PreCap-Grundidee wurde auf frischen Seeds geprüft; Gate A bestanden.
2. **v11 mini abgeschlossen:** 918/918 valide Runs. Gewichtete PreCap- und Hybrid-Reparaturen wurden als Kandidaten gescreent.
3. **v11 confirm abgeschlossen:** 2400/2400 Runs. Hybrid Gate verfehlt die vorab fixierten Kriterien gegen PreCap; PreCap bleibt beste getestete Aktivierung.
4. **v12 läuft:** Mechanismusstudie zur beobachteten Plain-Schwäche.

## Reihenfolge der Ideen

Die **Aktivierungsfamilie** (PreCap, gewichtete und hybride Varianten) wurde konzeptionell und in v11 **vor** einer etwaigen Erweiterung auf 2D-Netze untersucht. Die 2D-Netze waren nicht Teil der v10B/v11-Entscheidung. Die korrekte Reihenfolge ist daher: erst die Familie im kontrollierten 1D-Protokoll bereinigen; danach — mit PreCap als fixierter Referenz — eine neue 2D-Studie planen. Das vermeidet, Variantenwahl und räumliche Skalierung gleichzeitig zu vermischen.

## Warum v12

v11 trennt zwei Befunde: Gegen Standardaktivierungen hilft die Cap-Familie besonders bei `res_inv` und tiefen Netzen; gegen PreCap liefert das Hybrid-Gate keinen robusten Zusatznutzen. Unklar blieb, ob PreCap bei `plain` wegen einer falschen Optimierungsregion verliert oder weil die Residualarchitektur Teil des Mechanismus ist. V12 ändert deshalb ausschließlich Initialisierung/Lernrate und Architektur, nicht die Aktivierungsfamilie.

## v12-Protokoll

- 1.620 Runs: PreCap, SiLU, GELU; drei Aufgaben; Tiefen 4/8/16; `plain` und `res_inv`; drei Regime (Basis, hohe Lernrate, große Initialisierung); frische Seeds 200–209.
- Primäranalyse: gepaarter Log-MSE-Quotient gegen SiLU/GELU; sekundär gegen PreCap-Referenzlogik, Hmax-Peak, Gradient, Cap-Anteil und Ausfälle.
- Entscheidung: Erholt sich PreCap in `plain` in einem Regime, ist die Schwäche überwiegend Optimierungsabhängigkeit. Bleibt sie dort trotz Regimewechsel, während `res_inv` robust bleibt, ist der Residualpfad strukturell relevant.

## Danach

1. V12 abschließen und die Mechanismuserklärung mit Tabellen und Abbildungen festhalten.
2. PreCap als finale 1D-Referenz einfrieren, inklusive Version, Hyperparameter und Evidenzgrenzen.
3. Erst dann 2D-Protokoll präregistrieren: Aufgabenfamilie, Parametergleichheit, Tiefenskalierung, Stabilitätsmetriken und unabhängiger Holdout.
4. 2D-Pilot mit PreCap, SiLU und GELU; nur bei klarer Replikation die vollständige 2D-Studie starten.
5. Ergebnisbericht/Manuskript aus v10B–v12 plus 2D-Evidenz konsolidieren.
