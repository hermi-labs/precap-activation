# CapExp — Gesamterkenntnisse aus der 1D-Studienreihe

Stand: 20.10.2026. Diese Notiz trennt robuste Befunde, eingeschränkte Befunde und offene Fragen. Sie ist keine Aussage über 2D-Netze; diese wurden bisher nicht getestet.

## Kurzfazit

PreCap ist in dieser 1D-Studienreihe keine universell bessere Aktivierung. In flachen bis mitteltiefen `plain`-Netzen ist sie meist schlechter als SiLU/GELU. In der residualen inversen Architektur (`res_inv`) ist PreCap dagegen über Aufgaben, Seeds und Optimierungsregime hinweg klar besser, wobei der Nutzen mit Tiefe und der Stabilitätsbedarf sichtbar wird. Zusätzliche gewichtete oder hybride Reparaturen verbessern PreCap nicht reproduzierbar.

## Eingefrorene 1D-Referenz

PreCap: c=1,0; a=1,25; gamma=0,01; soft_s=1,5; Knie=0,75. Diese Konfiguration bleibt die Referenz für die nächste Phase. Nicht als allgemeine Default-Aktivierung, sondern für residuale Modelle und die in der Studie geprüfte Aufgabenfamilie.

## Evidenzkette

| Phase | Umfang | Frage | Befund |
|---|---:|---|---|
| v10B Holdout | abgeschlossener frischer Holdout | Hält der zentrale PreCap-Befund auf unabhängigen Seeds? | Gate A bestanden; der Grundbefund repliziert. |
| v11 mini | 918 Runs | Können `precap_wg` oder `hybrid_gate` PreCap verbessern? | Kein Kandidat erfüllt Gate B. |
| v11 confirm | 2.400 Runs | Ist der knappe Hybrid-Kandidat besser als PreCap? | Nein: flach Median d=+0,021, Win-Rate 46,1 %; tiefe Toleranz verletzt; ein Hybrid-Ausfall. |
| v12 Mechanismus | 1.620 Runs | Ist die Plain-Schwäche nur Optimierung/Init/LR? | Nein: In `plain` erholt sich PreCap in keinem Regime; `res_inv` robust besser. |

## Leistung nach Architektur und Tiefe

### Plain

- Bei `plain` erreicht PreCap in v12 in keinem der drei Regime die vorab festgelegte Erholung (mindestens 2 von 3 Aufgaben besser als beide Referenzen).
- Der mittlere Nachteil wächst mit Tiefe: gegen SiLU liegt der Median d bei +0,13 (Tiefe 4), +0,29 (Tiefe 8), +0,42 (Tiefe 16). Positives d bedeutet höheren MSE für PreCap.
- Besonders klar ist die Schwäche auf `sin3_quad`; `sin_mix` liegt näher an Gleichstand und `abs_wave` gewinnt im Basisregime einmal.
- Fazit: Für einfache Feed-forward-Ketten ist PreCap im getesteten Bereich keine empfehlenswerte Standardwahl.

### Res_inv

- In `res_inv` ist PreCap in 8 von 9 Aufgabe-Regime-Zellen gleichzeitig besser als SiLU und GELU.
- Auf der Seed-Ebene gewinnt PreCap 81/90 Einheiten gegen SiLU und 83/90 gegen GELU (einseitige Vorzeichentests p < 1e-15).
- Der Effekt ist auf `abs_wave` am stärksten und bleibt auf `sin3_quad` und `sin_mix` überwiegend bestehen.
- Fazit: Der Cap-Mechanismus wirkt nicht isoliert von der Architektur. Der Residualpfad ist wahrscheinlich Teil der Bedingung, unter der PreCap seinen Nutzen ausspielt.

## Flache versus tiefe Netze

Die Aussage „PreCap ist flach auf Augenhöhe und gewinnt erst tief“ muss präzise eingeschränkt werden:

- Gegen **PreCap selbst** lieferte `hybrid_gate` flach nur einen kleinen, nicht stabilen Vorteil und tief keinen toleranzkonformen Nutzen. Daraus folgt kein Hybrid-Upgrade.
- Gegen **SiLU/GELU** ist PreCap in `res_inv` auch bei Tiefe 4 und 8 häufig besser; der Unterschied wird in tieferen bzw. schwieriger stabilisierten Konfigurationen besonders relevant.
- In `plain` entsteht gerade kein später Vorteil: Der Nachteil nimmt von Tiefe 4 zu 16 zu.
- Deshalb ist die beste zusammenfassende Formulierung: PreCap ist in residualen Architekturen schon flach konkurrenzfähig bis besser und kann seinen Nutzen bei größerer Tiefe/Stabilitätsdruck stärker ausspielen; ohne Residualpfad trägt sich der Vorteil nicht.

## Stabilität und Mechanismus-Indizien

- PreCap reduziert hmax_peak gegenüber SiLU/GELU in beiden Architekturen (Median-Verhältnis 0,36–0,59).
- Gleichzeitig sind die finalen Gradienten in `plain` 3–5-mal höher, in `res_inv` dagegen erheblich niedriger (Verhältnis 0,03–0,12).
- Der finale Cap-Anteil liegt in `res_inv` höher als in `plain` (0,150 versus 0,067), was zu stärkerer wirksamer Begrenzung im residualen Modell passt.
- Das sind konsistente Mechanismus-Indizien, aber kein kausaler Beweis: Die Studie hat den Residualpfad nicht komponentenweise abliert.

## Variantenentscheidungen

- `precap_wg`: verworfen; klar schlechter als PreCap im v11-mini-Screening.
- `hybrid_gate`: verworfen; v11-confirm zeigt keinen robusten Mehrwert gegen PreCap und einen Ausfall bei Tiefe 64.
- PreCap: als 1D-Referenz eingefroren; mit klarer Architekturgrenze.

## Gültigkeitsbereich und Grenzen
- Getestet wurden nur drei synthetische 1D-Aufgaben, feste Breite 64, maximal Tiefe 64 in v11 bzw. 16 im kontrollierten v12-Mechanismusteil und die geprüften Optimierungsregime.
- Die Resultate sind keine Behauptung über Bilddaten, Transformer, Faltungen oder reale 2D-Regression.
- Die stabilitätsbezogenen Beobachtungen sind beschreibend; für eine kausale Erklärung folgen gezielte 2D- und Architekturablationsstudien.

## Übergang zu 2D

Die 2D-Phase prüft keine neue Aktivierungsvariante. Sie repliziert die eingefrorene 1D-Referenz PreCap gegen SiLU und GELU unter gleicher Parameterbudgetierung, zuerst im residualen 2D-Pilot und danach — nur bei Replikation — in einem umfassenderen 2D-Holdout.

