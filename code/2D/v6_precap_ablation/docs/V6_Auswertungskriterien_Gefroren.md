# V6 — eingefrorene Auswertungskriterien

Stand: 20.10.2026, vor Abschluss der laufenden Ablation.

## Forschungsfrage
Erklärt die Tiefe-64-Schwäche der bisherigen PreCap-Basisformel überwiegend (a) die Stärke bzw. Schwelle der Begrenzung (Parameter c oder s) oder (b) Seed-/Pfadvariabilität?

## Gültige Vergleichseinheit
Jeder Variantenvergleich erfolgt gepaart, innerhalb derselben Zelle (Aufgabe × Tiefe × Seed). Primäre Zielgröße ist die finale Test-MSE; kleinere Werte sind besser.

## Primäres Erfolgskriterium
Eine Variante (precap_c075, precap_c125, precap_s075 oder precap_s225) ist in einer der beiden Tiefe-64-Mechanismuszellen nur dann ein belastbarer Kandidat gegenüber precap_base, wenn sie:
1. eine niedrigere mittlere bzw. mediane finale Test-MSE als precap_base erreicht und
2. mindestens 4 von 5 gepaarten Seeds gegen precap_base gewinnt.

Ein einzelner günstiger Seed, ein Effekt nur in einer Referenz oder ein ungerichteter Mittelwert genügt nicht.

## Zellen und Rollen
- Mechanismus: radial_ripple, Tiefe 64.
- Mechanismus: ridge_mix, Tiefe 64.
- Anker: radial_ripple, Tiefe 16. Der Anker muss mindestens stabil bleiben; eine Verbesserung bei Tiefe 64 ist nicht überzeugend, wenn sie den bekannten Vorteil bei Tiefe 16 klar verschlechtert.
- Referenzen: SiLU und GELU werden als Kontext berichtet, aber der direkte Ablationsentscheid ist Variante gegen precap_base.

## Mechanistische Deutung
- c075 oder s075 konsistent besser: Hinweis, dass die Basis zu schwach begrenzt (Unterbegrenzung).
- c125 oder s225 konsistent besser: Hinweis, dass die Basis zu stark begrenzt (Überbegrenzung).
- Kein konsistenter Sieger bei großer Seed-Streuung: Variabilität dominiert; c und s sind in diesem Bereich kein klarer Stellhebel.
- Unterschiedliche Sieger je Aufgabe: Parameterwirkung ist aufgabenabhängig; keine globale Neukalibrierung behaupten.

## Zusätzliche Prüfung
- Checkpoint-Verläufe: Entsteht ein Vorteil früh, hält er sich oder wird er später verloren?
- Streuung: Seed-weise Differenzen, Median und Mittelwert berichten; Ausreißer offen kennzeichnen.
- Integrität: 105/105 vollständige Runs, 630/630 Checkpoint-Zeilen, eindeutige Paare, keine bad-Runs, Formel-Selfcheck = 0.
- Keine neue Parameterauswahl während der Auswertung. Falls eine Variante die Kriterien erfüllt, folgt erst danach ein separat eingefrorenes Bestätigungsexperiment mit mehr Seeds.

## Grenzen
Mit fünf Seeds liefert V6 einen gezielten Hinweis, keinen endgültigen Nachweis. Eine adaptive oder eingabewertabhängige Wahl von c/s ist erst sinnvoll, wenn V6 einen robusten, reproduzierbaren und nicht nur seed-spezifischen Effekt zeigt.
