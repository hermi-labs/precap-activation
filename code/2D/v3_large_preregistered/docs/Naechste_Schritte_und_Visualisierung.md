# CapExp — Nächste Schritte, Visualisierung und Machbarkeit

**Stand:** Dieses Memo hält Zeitplan, Bewertungsgrundlagen, Visualisierungsideen und technische Machbarkeit fest. Es ist bewusst getrennt von der eingefrorenen Studienlogik des laufenden v3-Experiments.

## Aktueller Ablaufplan

1. **v3 sauber fertiglaufen lassen.** Während des Laufs werden keine Änderungen an Formel, Aufgaben, Seeds, Tiefen, Optimierer oder Gate vorgenommen.
2. **Integrität prüfen.** Pflichtpunkte: 900/900 Runs, vollständige gepaarte Zellen, keine erhöhte PreCap-bad-rate, keine beschädigten Ergebnisdateien.
3. **Eingefrorene Auswertung ausführen.** Kernentscheidungen nur auf den drei Legacy-Core-Aufgaben; neue Aufgaben getrennt als Generalisierungsbefund berichten.
4. **V2 visualisieren.** Zuerst die bereits gesicherte Geschichte sauber erzählen, danach optional v2 und v3 gemeinsam darstellen.
5. **Entscheidung über Folgeexperimente.** Erst nach v3 festlegen, ob Tiefe-Profile, Formelvarianten oder realistischere Benchmarks als nächstes kommen.

## Bewertungsgrundlagen bleiben gültig

Die mathematischen und methodischen Grundlagen sind derzeit vollständig genug für belastbare 2D-Aussagen:

- **Aktivierungsformel:** eingefrorene PreCap-v9-Urformel mit mathematischer Spezifikation.
- **Primärmetrik:** \(d=\log_{10}(\mathrm{MSE}_{PreCap}/\mathrm{MSE}_{Ref})\), negativ begünstigt PreCap.
- **Primärentscheidung:** Auf den drei Kernaufgaben müssen mindestens 2 von 3 Aufgaben im Median gegen **beide** Referenzen gewinnen; zusätzlich keine höhere bad-rate.
- **Sekundärgrößen:** gepaarte Sign-Tests, Aktivierungspeaks, Gradienten, finaler Cap-Anteil.
- **Interpretationsregel:** Neue Aufgaben und größere Tiefen erweitern die Evidenz, überschreiben aber nicht rückwirkend die v2-Kernaussage.

## Visuelle Darstellung von v2

Für v2 ist eine starke, gut lesbare Darstellung sinnvoll und machbar. Empfohlene Ebenen:

1. **Entscheidungsebene:** Scorecard oder Matrix mit PASS/FAIL pro Aufgabe gegen SiLU und GELU.
2. **Strukturebene:** Heatmap Aufgabe × Tiefe mit den gepaarten Effektgrößen \(d\).
3. **Mechanikebene:** zusätzliche Panels für `hmax_peak`, `grad_final` und `cap_final`, damit sichtbar wird, dass PreCap nicht nur Fehler verschiebt, sondern auch Extreme reduziert.
4. **Narrative Ebene:** ein kompakter Zeitstrahl oder Ablaufblock, der von 1D-v9 über 2D-v1b zu 2D-v2 führt.

Das lässt sich sehr gut an euren vorhandenen Stil anpassen: klarer Forschungs-/Lab-Look, dunkler Hintergrund, leuchtende, aber wenige Akzentfarben, klare Trennung zwischen "gesichert", "offen" und "nächste Frage".

## Live-Modell / Neuronen-Rendering

Eine interaktive Darstellung einzelner Neuronen, Aktivierungen und Layerflüsse ist **grundsätzlich machbar**, aber man sollte zwei Varianten unterscheiden.

### Variante A — leichte, sinnvolle Lösung

- Ein gespeichertes Trainingsmodell oder mehrere Snapshots laden.
- Für ein 2D-Gitter Eingaben durch das Netz schicken.
- Pro Layer Aktivierungskarten, Residualbeiträge, PreCap-Interventionen und Outputflächen rendern.
- Slider für Tiefe, Seed, Aufgabe, Aktivierung und Trainingsschritt.

Diese Variante ist auf lokaler Hardware sehr gut machbar, weil sie nur Vorwärtsläufe und vorberechnete Snapshots visualisiert. Sie ist wissenschaftlich nützlich, weil man wirklich sieht, **wo** PreCap eingreift und wie Repräsentationen sich über Tiefe verändern.

### Variante B — volle Live-Simulation mit Einzelneuronen-Tracking

- Training live mitlaufen lassen.
- Währenddessen pro Schritt oder in hoher Frequenz viele interne Tensoren, Aktivierungen, Gradienten und Gewichte abgreifen.
- Daraus in Echtzeit mehrere Visualisierungsebenen rendern.

Das ist ebenfalls machbar, aber deutlich schwerer und potentiell teuer. Auf lokaler Hardware kann so etwas schnell durch Logging-Overhead, Speicherverbrauch und Renderkosten die eigentliche Trainingsgeschwindigkeit verfälschen. Für euren aktuellen Erkenntnisstand wäre diese Variante eher **zweite Priorität**.

## Sinnvoll auf deiner Hardware?

Ja, aber mit Maß. Da dein Rechner nicht beliebig als Dauerläufer gedacht ist und du klare, robuste Projekt-Logs bevorzugst, ist die richtige Reihenfolge:

1. Ergebnisse erst sauber erzeugen.
2. Danach **off-line visualisieren** aus gespeicherten CSVs und ausgewählten Modell-Snapshots.
3. Erst später eine Live-Ansicht bauen, wenn klar ist, welche Größen wirklich aufschlussreich sind.

Für den jetzigen Stand wäre also ein Visual-Tool mit vorberechneten Snapshots sehr sinnvoll; ein permanentes Live-Neuronen-Monitoring dagegen eher dann, wenn v3 bestätigt, dass sich die Tiefenfrage wirklich lohnt.

## Praktische Empfehlung nach v3

- **Kurzfristig:** V2 als hochwertige statische/halbinteraktive Visualisierung bauen.
- **Mittelfristig:** V3 nachziehen und die Darstellung auf neue Tiefen 32/64 erweitern.
- **Danach:** Ein eigenes Explorations-Tool bauen, das Modell-Snapshots lädt und Layer-, Neuronen- und Aktivierungsansichten interaktiv zeigt.

So bleibt die wissenschaftliche Evidenz sauber von der Präsentation getrennt, und die Visualisierung wird nicht zum Flaschenhals der Experimente.
