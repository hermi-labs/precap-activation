# CapExp — Fachliche Folgerungen nach v2 und v3

## Verdichtete Arbeitslage

Die bisherige Kernhypothese bleibt erhalten: **PreCap zeigt ein replizierbares 2D-Signal auf den vorab definierten Kernaufgaben**. Diese Aussage ist jetzt nicht mehr nur pilotartig, sondern durch den v2-Holdout und die große v3-Replikation gestützt.

Gleichzeitig muss die stärkere Tiefenhypothese enger formuliert werden. Die Daten sprechen **nicht** für einen unbegrenzt stabilen oder mit Tiefe automatisch wachsenden Vorteil. Stattdessen ergibt sich ein klareres Profil: robust in flachen bis mittleren Tiefen, tragfähig noch bis 32, aber deutlich abgeschwächt bei 64, besonders gegen GELU.

## Neue Arbeits-Hypothese

> PreCap verbessert unter dem derzeitigen Trainingsprotokoll die Test-MSE auf bestimmten strukturhaltigen 2D-Aufgaben replizierbar, wahrscheinlich teilweise über stabilere interne Dynamik. Der Vorteil ist jedoch aufgaben- und tiefenabhängig und verliert in sehr tiefen Netzen deutlich an Robustheit.

## Konsequenzen für die nächste Forschungsphase

1. **Nicht mehr fragen, ob überhaupt ein Signal existiert.** Das ist derzeit hinreichend belegt.
2. **Mechanismusfrage priorisieren.** Warum gehen Aktivierungspeaks und Gradienten oft zurück, und wann übersetzt sich das in bessere Generalisierung?
3. **Grenzfrage priorisieren.** Warum kippt der Vorteil bei Tiefe 64? Liegt es an Optimierung, Architektur, Aufgabenfrequenz oder an einer Sättigungsgrenze der Intervention?
4. **Aufgabenstruktur systematisieren.** Knickartige und gemischte Strukturen sehen günstiger aus als hochfrequente Gitter; das sollte künftig als Hypothese explizit getestet werden.
5. **Visualisierung und Snapshot-Analyse vorbereiten.** Für Mechanismusarbeit reichen rohe CSVs nicht mehr; benötigt werden gezielte Modell-Snapshots und Layer-/Aktivierungsansichten.

## Vorläufige Fachlesart

Die 2D-Linie hat jetzt einen belastbaren Befund und zugleich eine sichtbare Grenze. Genau diese Kombination ist wissenschaftlich wertvoll: Das Projekt hat die Schwelle von "interessanter Effekt" zu "replizierbarer, aber strukturabhängiger Intervention" überschritten.
