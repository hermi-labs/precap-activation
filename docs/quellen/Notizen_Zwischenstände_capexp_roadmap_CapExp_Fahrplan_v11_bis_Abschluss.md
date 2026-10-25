# CapExp — Fahrplan von v11 bis zum Evidenzpaket

**Stand:** v10 ist vollständig ausgewertet; v10B-Holdout läuft. Dieser Fahrplan trennt Exploration, Bestätigung und finale Dokumentation strikt.

## Großes Ziel

Robust und reproduzierbar bestimmen, **wann** PreCap gegenüber SiLU und GELU Vorteile bietet, **unter welchen Architekturbedingungen** diese Vorteile auftreten und **welche Stabilitätsindikatoren** dazu passen. Die Aussage soll nicht lauten „PreCap gewinnt immer“, sondern präzise und fair abgegrenzt sein.

## Gesamt-Timeline

```text
v1–v7   Grundlagen, Aufgabenfamilie, frühe Architektur- und Aktivierungschecks
   │
v8–v9   Hypothesen schärfen; Stabilität, Residualdesign und Parameterraum eingrenzen
   │
v10     Exploration abgeschlossen: 6.360 Runs
   │      Tiefes res_inv-Setting: PreCap als starker Kandidat
   │
v10B    Bestätigungs-Holdout: eingefrorene Sieger, Seeds 10–19, 180 Runs
   │      └─ Gate A: hält der PreCap-Befund auf neuen Seeds?
   │
v11 mini  Mechanismen/Varianten gezielt testen: precap_wg und hybrid_gate
   │      └─ Gate B: verbessert eine Variante flache Netze ohne tiefen Vorteil zu verlieren?
   │
v11 full  Bestätigung der v11-mini-Gewinner über Aufgaben, Tiefen und neue Seeds
   │      └─ Gate C: reproduzierbarer, fairer Nettoeffekt?
   │
v12 optional  Robustheit außerhalb der bisherigen 1D-Suite
   │      └─ andere Zielfunktionen, Breiten, Trainingsbudgets und Störungen
   │
Finale Evidenzmappe  Daten, Tabellen, Grafiken, Tests, Reproduktion und Grenzen
```

## Gate A — v10B-Holdout auswerten

### Zweck

Die v10-Auswahl war explorativ. Deshalb werden die 18 jeweils besten, in v10 festgelegten Konfigurationen unverändert auf 10 neuen Seeds getestet. Es gibt keine neue Hyperparametersuche im Holdout.

### Design

| Dimension | Festlegung |
|---|---|
| Aufgaben | `sin3_quad`, `sin_mix`, `abs_wave` |
| Tiefen | 32 und 64 |
| Architektur | `res_inv` |
| Aktivierungen | PreCap, SiLU, GELU |
| Seeds | 10 bis 19 |
| Umfang | 3 × 2 × 3 × 10 = 180 Runs |
| Primärmetrik | Test-MSE pro Seed |
| Vergleich | Gepaarter Vergleich über identische Seeds |

### Bestehensregel

Weiter zu v11 mini, wenn PreCap bei Tiefe 32/64 auf mindestens zwei der drei Aufgaben gegen SiLU und GELU gewinnt oder praktisch gleichzieht, ohne neue numerische Ausfälle oder eine klar schlechtere Stabilitätsdiagnostik zu zeigen.

### Auswertung

1. Vollständigkeit: 180 eindeutige Zeilen, keine Lücken, keine Duplikate.
2. Qualität: Median, Quartile und Gewinnquote pro Aufgabe/Tiefe/Aktivierung.
3. Fairer Vergleich: PreCap gegen SiLU sowie PreCap gegen GELU, gepaart nach Seed.
4. Statistik: zweiseitiger Wilcoxon-Signed-Rank-Test sowie Effektgrößen/Verhältnis der Mediane.
5. Stabilität: Ausfälle, Aktivierungsmaxima, Gradientenspitzen, Kappungsanteil und Generalisierungslücke.
6. Entscheidung als kurze, vorab definierte Gate-A-Notiz dokumentieren.

## Gate B — v11 mini

### Zweck

Gezielt prüfen, ob Varianten die beobachtete Schwäche bei flachen Netzen adressieren können, ohne die Stärke im tiefen residualen Regime zu verlieren.

### Kandidaten

| Variante | Hypothese |
|---|---|
| `precap` | Referenz aus v10/v10B |
| `precap_wg` | Gewichtete bzw. gegatete Kappung kann bei flacheren Netzen weniger restriktiv sein |
| `hybrid_gate` | Misch-/Gate-Variante kann Anpassungsfähigkeit und Kontrolle kombinieren |
| SiLU, GELU | Unveränderte Referenzen |

### Kleines, faires Raster

| Bereich | Einstellung |
|---|---|
| Aufgaben | Drei bestehende Aufgaben; optional eine vorab definierte zusätzliche 1D-Aufgabe |
| Tiefen | 4, 8, 32, 64 |
| Architektur | `plain` und `res_inv` bei 4/8; `res_inv` bei 32/64 |
| Breite | 64 als Hauptvergleich |
| Seeds | 0–9 für Exploration, danach frische Holdout-Seeds für den Sieger |
| Budget | Gleiches Trainingsbudget und gleicher Datengenerator für alle Aktivierungen |
| Suche | Kleines, symmetrisches Parameterbudget pro neuer Variante |

### Entscheidungsregel

Eine neue Variante geht nur in v11 full, wenn sie mindestens eines erfüllt:

- flache Netze sichtbar verbessert und den tiefen PreCap-Befund nicht zerstört;
- tiefe Netze reproduzierbar verbessert;
- einen klareren Stabilitäts-/Mechanismusvorteil bei vergleichbarer Testqualität liefert.

## Gate C — v11 full

### Zweck

Die in v11 mini ausgewählten Varianten nicht erneut breit „optimieren“, sondern vorab eingefroren über neue Seeds und mehrere Bedingungen bestätigen.

### Bestätigungsdesign

- Gewinner aus v11 mini einfrieren; keine Anpassung nach ersten Holdout-Ergebnissen.
- Je Zelle mindestens 10 frische Seeds, identisch über alle verglichenen Aktivierungen.
- Zusätzlich eine Robustheitsachse variieren: Breite (z. B. 32/64/128) **oder** Trainingsbudget (z. B. 1.500/3.000/6.000 Schritte), nicht beides zugleich in derselben Hauptstudie.
- Je Bedingung gepaarte Seed-Vergleiche und klare Trennung zwischen primären und explorativen Metriken.

## V12 optional — externe Robustheit

Nur starten, wenn v10B und v11 full positive, konsistente Befunde zeigen.

- Zusätzliche Zielfunktionen mit anderer Glattheit/Frequenzstruktur.
- Breitenvariation oder andere Initialisierung.
- Leichte Label-/Eingabestörung oder abweichender Extrapolationsbereich.
- Optional kleines MLP- oder tabellarisches Real-World-Dataset, sofern Datenaufbereitung und Metrik vorab festgelegt sind.

## Finale Evidenzmappe

### Inhalt

1. **Protokoll:** Hypothesen, primäre Metriken, Seed-Schemata und Gates.
2. **Daten:** Roh-CSV, Manifeste, Konfigurationen und Prüfsummen.
3. **Auswertung:** Tabellen je Aufgabe/Tiefe, gepaarte Vergleiche, Fehlerquoten und Sensitivitäten.
4. **Grafiken:** Leistung vs. Tiefe, Seed-Streuung, Stabilitätsmetriken, sensible Parameterbereiche.
5. **Reproduktion:** ein Startbefehl, erwartete Ordnerstruktur und Validierungsskript.
6. **Interpretation:** bestätigte Aussage, Gegenbeispiele, Grenzen und offene Fragen.

## Ordnerstruktur

```text
PyCharmMiscProject/
├── v10/                 # abgeschlossene Exploration und v10-Auswertung
├── v10B_holdout/        # laufender/abgeschlossener, eingefrorener Holdout
│   ├── meta/            # Plan, Manifest, eingefrorene Konfigurationen, Validierung
│   ├── results/         # shardweise Rohresultate
│   └── logs/            # Worker-, Launcher- und Validierungslogs
├── capexp_roadmap/      # Projektfahrplan und Statusdokumente
│   ├── CapExp_Fahrplan_v11_bis_Abschluss.md
│   └── CapExp_Status_Timeline.md
└── v11_mini/            # erst nach Gate A anlegen
```

## Unmittelbar nach v10B

1. Ergebnisanzahl und eindeutige Manifest-Abdeckung validieren.
2. Holdout-Auswertung erzeugen und Gate A schriftlich entscheiden.
3. Bei bestandenem Gate A `v11_mini/` mit Manifest, Runner und Analysegerüst anlegen.
4. Bei nicht bestandenem Gate A die Aussage eng auf die v10-Exploration begrenzen und das v11-Design auf Diagnose statt Erweiterung ausrichten.

## Stand 20.10.2026

- v10B Holdout: abgeschlossen, Gate A PASS.
- v11 mini: abgeschlossen (918/918 Runs, keine Ausfälle). Gate B: kein neuer Kandidat besteht; `precap_wg` verworfen; `hybrid_gate` (Knie 0,75, beta 0,3) knapp verfehlt (Tiefe 64 `sin3_quad`: Faktor 1,66 > 1,5).
- v11 confirm: Manifest in `v11_confirm/V11_confirm_Manifest.md`; 2400 Runs, Tiefen 4/8/16/32/64, Seeds 100–109, nur `hybrid_gate` gegen PreCap, SiLU, GELU.

## v11 abgeschlossen, v12 gestartet (20.10.2026)

- **v11 confirm abgeschlossen:** 2400/2400 Runs; hybrid_gate verfehlt alle vorab fixierten Kriterien gegen PreCap. PreCap bleibt Referenz/Kandidat.
- **Evidenz:** Cap-Familie ist tief und mit `res_inv` gegenüber SiLU/GELU stark; die Verbesserung durch das hybride Gate gegenüber PreCap ist nicht reproduzierbar.
- **v12:** 1.620-Run-Mechanismusstudie zur Plain-Schwäche (PreCap/SiLU/GELU; Tiefen 4/8/16; `plain`/`res_inv`; drei Init/LR-Regime; Seeds 200–209). Manifest: `v12_plain_mechanism/V12_Manifest.md`.
- Nach v12: Ergebnisse schließen, Analyse/Abbildungen konsolidieren und Manuskript-/Abschlussfahrplan fortsetzen.

## v12 abgeschlossen (20.10.2026)

- 1620/1620 Runs ohne Ausfall. Plain-Schwäche nicht durch Init/LR erklärbar; strukturell, in `res_inv` dagegen sehr robuster PreCap-Vorteil. Details: `V12_Abschluss.md`, `v12_plain_mechanism/CapExp_v12_Auswertung.md`.
- Nächste Schritte: 1D-Referenz PreCap einfrieren (Parameter, Evidenzgrenzen: Vorteil an `res_inv`, kein Vorteil in `plain`); 2D-Protokoll präregistrieren; 2D-Pilot mit PreCap, SiLU, GELU.

## 1D-Synthese und nächste Phase (20.10.2026)

- Vollständige 1D-Synthese: `CapExp_1D_Gesamterkenntnisse.md`.
- Präregistrationsentwurf für die nächste Phase: `CapExp_2D_Pilot_Manifest_Entwurf.md`.
- Entscheidend: PreCap ist keine allgemeine Feed-forward-Aktivierung; der robuste 1D-Befund ist Architektur-gebunden (`res_inv`). 2D prüft jetzt nur die Übertragbarkeit dieses begrenzten Befunds gegen SiLU/GELU.

## 1D endgültig eingefroren — Übergang zu 2D

- **1D abgeschlossen:** keine weiteren Varianten- oder Optimierungssuchen in 1D. Der Abschlussordner im Projekt ist `v12_finale_1d/`.
- **Eingefrorene Referenz:** PreCap (c=1,0; a=1,25; gamma=0,01; soft_s=1,5; Knie=0,75), beschränkt auf `res_inv` und die getestete synthetische 1D-Aufgabenfamilie.
- **Robuster Befund:** Vorteil gegenüber SiLU/GELU in residualer Architektur; keine universelle Verbesserung und klarer Nachteil in `plain`.
- **Nächster Arbeitsblock:** 2D-Pilot präregistrieren und implementieren; erst nach bestandenem Pilot unabhängigen 2D-Holdout starten.

