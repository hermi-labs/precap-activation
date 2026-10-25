# V17 – Residual-Skalierung und Tiefe (vor Start eingefroren)

## Ziel
Kausaler Test von V16-H2: Verschwindet der PreCap-Vorteil in tiefen Netzen, weil die Residualskalierung `1/sqrt(D)` Aktivierungsbeitrage und Nichtlinearitaetsnutzung absenkt?

## Design
Aufgaben radial_ripple, ridge_mix, smooth_bowl, corner_kink; Tiefen 64/128; Aktivierungen precap_base, SiLU, GELU; Skalen inv_sqrt=1/sqrt(D), inv_quart=D^-1/4, constant=1; 20 neue Seeds 9800-9819; 3000 Schritte, Checkpoints 0/100/200/400/800/1200/2000/3000. Total 4*2*3*3*20=1440 Runs, 11520 Zeilen. Architektur/Initialisierung/Adam LR=0.001 wie V16. s225 absichtlich ausgeschlossen: V16 zeigt einen aufgabenspezifischen statt allgemeinen Effekt.

## Sicherheits- und Interpretationsregeln
Konstante Skala kann instabil sein; jeder Run schreibt Checkpoints, und nicht-endliche Werte sollen als Fehler behandelt werden. Zuerst Pilot mit exakt zwei Seeds (9800,9801) in separatem Pilotordner; Vollrun nur bei keinen NaN/Inf und vollstaendigen Pilotdaten. Vergleich nur innerhalb derselben Tiefe/Skala/Seed, daher ist die geaenderte Netzfunktion kein unfairer Aktivierungsvergleich.

## Konfirmatorisch
Bei Schritt 3000: precap_base gegen SiLU und gegen GELU fuer jede Aufgabe*Tiefe*Skala = 48 Wilcoxon-Tests, Holm ueber 48. Primare H2-Vorhersage: PreCap-Vorteil ist unter inv_sqrt am kleinsten und nimmt bei inv_quart/constant zu, zusammen mit post_rms_scaled, |pre|>0.5 und cap_fraction. Diese Vorhersage wird nur als Muster gewertet, nicht als einzelner signifikanter Test.

## Entscheidung
H2 gestuetzt, falls mindestens 3/4 Aufgaben bei mindestens einer tiefen Stufe einen kleineren ratio(base/ref) unter inv_quart oder constant als unter inv_sqrt zeigen, ohne dass die Verbesserung ausschliesslich auf instabile/fehlende Runs beruht. H2 nicht gestuetzt, falls die Verhaeltnisse trotz klar groesserer Aktivierungsamplitude nicht systematisch sinken. Alle Resultate bleiben 2D-res_inv-spezifisch.
