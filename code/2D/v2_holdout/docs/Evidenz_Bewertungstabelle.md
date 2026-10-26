# CapExp v2 — lebende Evidenz- und Bewertungstabelle

> **Status:** Diese Tabelle wird aus dem eingefrorenen Ergebnisdatensatz `v2_holdout_results.csv` erzeugt. Sie trennt bestätigte Evidenz, Grenzen und die vorab festgelegten Kriterien für Folgestudien. Aktualisierung: `python update_evidence_scorecard.py`.

## Evidenzstatus

| Evidenzfrage | Ergebnis | Bewertung |
|---|---|---|
| Vollständigkeit | 270/270 Runs; 0 bad | Gültig |
| Replikation des v1b-Pilots | Neue Seeds 400–409, 2/3 Aufgaben gewinnen gegen SiLU und GELU | Bestätigt |
| Formelkonstanz | v9-Referenz, `formula_max_abs_err=0` vor Start | Bestätigt |
| Generalisierbarkeit | Nur synthetische 2D-Aufgaben / Res-Inv, Breite 64 | Offen |
| Tiefe über 16 hinaus | Nicht getestet | Offen |
| Reale Daten | Nicht getestet | Offen |

**Primärmetrik:** $d=\log_{10}(\mathrm{MSE}_{PreCap}/\mathrm{MSE}_{Ref})$; negativ ist besser für PreCap. Klammern: Siege / gepaarte Seeds.

## Leistungsprofil

| Aufgabe | Tiefe | vs. SiLU | vs. GELU | Interpretation |
|---|---:|---|---|---|
| radial_ripple | 4 | -0.805 (10/10) | -0.713 (10/10) | robust gegen beide |
| radial_ripple | 8 | -0.953 (10/10) | -0.994 (10/10) | robust gegen beide |
| radial_ripple | 16 | -0.025 (5/10) | -0.438 (9/10) | robust gegen beide |
| ridge_mix | 4 | -0.880 (10/10) | -0.481 (9/10) | robust gegen beide |
| ridge_mix | 8 | -0.402 (8/10) | -0.239 (7/10) | robust gegen beide |
| ridge_mix | 16 | -0.415 (10/10) | -0.123 (7/10) | robust gegen beide |
| corner_kink | 4 | -0.516 (10/10) | -0.116 (6/10) | robust gegen beide |
| corner_kink | 8 | -0.329 (10/10) | +0.031 (4/10) | nur gegen SiLU |
| corner_kink | 16 | -0.359 (10/10) | +0.054 (2/10) | nur gegen SiLU |

## Aggregiert nach Tiefe

| Tiefe | vs. SiLU | vs. GELU | Lesart |
|---:|---|---|---|
| 4 | -0.751 (30/30) | -0.481 (25/30) | klar |
| 8 | -0.581 (28/30) | -0.366 (21/30) | klar |
| 16 | -0.328 (25/30) | -0.072 (18/30) | schwach / aufgabenabhängig |

## Stabilitätsprofil

| Kennzahl | Gegen SiLU | Gegen GELU | Interpretation |
|---|---:|---:|---|
| Aktivierungspeak: Median PreCap/Ref | 0.665 | 0.780 | unter 1: niedrigere Extreme |
| Finaler Gradient: Median PreCap/Ref | 0.713 | 0.795 | unter 1: niedrigere Extreme |
| Finaler Cap-Anteil, Median | 0.030 | — | selektive Intervention |

## Aussagegrenzen

- **Gesichert:** Unter diesem Trainingsprotokoll verbessert PreCap die Test-MSE robust auf `radial_ripple` und `ridge_mix`; gegen SiLU auch auf `corner_kink`.
- **Nicht gesichert:** Ein allgemeiner Vorteil gegen GELU auf knickartigen Zielstrukturen, ein wachsender Vorteil mit Tiefe, oder Übertragbarkeit auf reale Daten.
- **Wichtig:** Abnehmende Effektgröße mit Tiefe ist hier ein Befund, keine Regel über PreCap im Allgemeinen.

## V3-Erweiterung (große 2D-Studie)

| Achse | v2-Status | v3-Ergebnis | Bewertung |
|---|---|---|---|
| Vollständigkeit | 270/270 Runs; 0 bad | 900/900 Runs; 300/300 vollständige Paare; 0 bad | Bestätigt |
| Primäres Gate | offen für große Replikation | 3/3 Core-PASS-Aufgaben | Bestätigt |
| Tiefe 32 | nicht getestet | aggregiert negativ gegen SiLU und GELU | Bestätigt |
| Tiefe 64 | nicht getestet | nahezu neutral aggregiert; gegen GELU leicht positiv | Grenze sichtbar |
| radial_ripple | robust bis 16 | robust bis 32, kippt bei 64 | teilweise bestätigt |
| ridge_mix | robust bis 16 | robust bis 32, kippt bei 64 | teilweise bestätigt |
| corner_kink | nur gegen SiLU robust, GELU schwach | gegen SiLU robust bis 64; gegen GELU insgesamt noch negativ, aber 64 kippt | differenziert bestätigt |
| Neue Aufgaben | nicht Teil von v2 | smooth_bowl/angled_kink überwiegend günstig; highfreq_grid nahezu neutral | explorativ positiv/neutral |

**Lesart v3:** Das vorregistrierte Gate ist bestanden. PreCap trägt die Kernaussage der 2D-Linie nun über die großen Replikationsachsen hinweg, zeigt aber zugleich eine klare Grenze: Bei Tiefe 64 schrumpft der aggregierte Vorteil stark und wird gegen GELU insgesamt leicht positiv.

**Konsequenz:** Die nächste wissenschaftliche Frage ist nicht mehr, ob es auf den Kernaufgaben überhaupt ein replizierbares Signal gibt, sondern warum die Vorteile in sehr tiefen Netzen abflachen und auf welchen Aufgabenstrukturen sie stabil bleiben.

## Nächster großer Lauf

| Achse | Eingefrorene Entscheidung vor Start | Erfolgskriterium |
|---|---|---|
| Formel | Unverändert: `precap-v9` | Keine Formeländerung |
| Architektur | Res-Inv, Breite 64; Tiefen 4, 8, 16, 32, 64 | Effektprofil über Tiefe messbar |
| Aufgaben | Bestehende 3 plus neue, vorab spezifizierte glatte/hochfrequente/knickartige 2D-Holdout-Familien | Keine Auswahl nach Ergebnis |
| Seeds | Neue, noch nicht verwendete Seed-Liste | Vollständig gepaart |
| Datenrealismus | Separater Benchmark mit festem Protokoll, nicht in die 2D-Hauptentscheidung mischen | Bericht getrennt ausweisen |
| Primärgate | Mindestens 2/3 Kernaufgaben: Median d < 0 gegen SiLU und GELU; keine höhere bad-rate | PASS/FAIL |

## Pflegeprotokoll

- Für einen **identischen, vorab spezifizierten** Folgelauf: Ergebnisdatei in einen neuen Studienordner schreiben und diese Tabelle mit einem neuen, datierten Abschnitt ergänzen; den v2-Abschnitt niemals überschreiben.
- Für neue Architekturen, Daten oder Tiefen: eigene Scorecard-Sektion und getrennte PASS/FAIL-Entscheidung erstellen.
- Jede Änderung an Formel oder Parametern definiert eine neue Aktivierungsvariante, nicht eine Aktualisierung von PreCap-v9.