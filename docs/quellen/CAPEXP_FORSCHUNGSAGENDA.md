# CapExp / PreCap — Forschungsagenda und Experiment-Logbuch

> **Arbeitsdokument.** Dieses Dokument ist der rote Faden des Projekts. Nach jedem Run werden nur beobachtete Befunde ergänzt; Hypothesen und Entscheidungen bleiben klar getrennt.
>
> Letzte Aktualisierung: 2026-10-19
> Aktueller Status: v10 läuft noch; danach vollständige v10-Auswertung, dann v11-Mini-Entscheidung.

## 1. Nordstern

Ziel ist nicht, schnell eine „universelle Aktivierungsfunktion“ zu behaupten. Ziel ist, schrittweise zu prüfen, ob eine gekappte exponentielle Aktivierungsfamilie in tiefen Netzen unter klaren Bedingungen zuverlässig nützlich ist, warum das geschieht und wann sie nicht die richtige Wahl ist.

Die Arbeit folgt vier Prinzipien:

- **Vergleichbarkeit vor Bestwerten:** gleiche Aufgaben, Budget, Seeds und sinnvolle Baselines.
- **Robustheit vor Einzelresultaten:** Mediane, Streuung, Ausfälle und Holdout-Seeds zählen gemeinsam.
- **Mechanismus vor Marketing:** Aktivierungen, Architektur, Initialisierung und Optimierung werden getrennt abliert.
- **Verallgemeinerung in Stufen:** 1D-Synthese → 2D-Synthese → breitere Standard-Benchmarks → Architektur-/Matrixideen.

## 2. Kernhypothese

### Arbeitsformeln

Die bisherige Familie wird in der vorhandenen Implementierung als CapExp und PreCap untersucht. Die genaue Implementierung bleibt die Referenz im jeweiligen Versionsskript; mathematische Notation und Code müssen bei Änderungen zusammen aktualisiert werden.

Arbeitsidee:

1. Eine exponentielle positive Seite kann lokale Anpassungsfähigkeit liefern.
2. Eine Kappung bzw. Vor-Kappung kann extreme Aktivierungen begrenzen.
3. In tiefen Netzen kann das zusammen mit residualer Skalierung den Gradienten- und Aktivierungsmaßstab besser kontrollieren.
4. Dieser Nutzen kann nur in bestimmten Tiefen, Architekturen und Aufgaben auftreten; flache Modelle können weiterhin SiLU/GELU bevorzugen.

### Falsifizierbare Aussagen

- **H1 — Tiefe:** PreCap ist bei Tiefe 32/64 mit `res_inv` auf mehreren Aufgaben mindestens so gut wie SiLU und GELU, gemessen auf frischen Holdout-Seeds.
- **H2 — Stabilität:** Der mögliche Qualitätsvorteil geht mit kontrollierteren Aktivierungs- und Gradientenskalen einher, nicht nur mit Zufall oder einem einzelnen Seed.
- **H3 — Breite:** Die Beobachtung überträgt sich von 1D auf mindestens eine klar definierte 2D-Funktionsfamilie.
- **H4 — Schutzlogik:** v11-Varianten können die flache Schwäche reduzieren, ohne die tiefe Robustheit sichtbar zu beschädigen.
- **H5 — Struktur:** Eine spätere Matrix-/Faktorisierungsstruktur liefert einen Mehrwert über Residualskalierung und Aktivierungswahl hinaus.

## 3. Evidenzstufen

| Stufe | Mindestanforderung | Aussage, die zulässig ist |
|---|---|---|
| E0 — Idee | Formel + Motivation + Testspezifikation | „Prüfbare Hypothese“ |
| E1 — Exploration | Bestehende Seeds, mehrere 1D-Aufgaben | „Vielversprechendes Explorationssignal“ |
| E2 — Replikation | Frische Holdout-Seeds, festes Protokoll | „Replizierbarer Befund in diesem Setup“ |
| E3 — Breite | 2D-Aufgaben, mehrere Funktionsfamilien | „Über 1D hinaus beobachtet“ |
| E4 — Baselines | Stärkere Baselines, Budget-/Tuning-Fairness | „Wettbewerbsfähig unter diesem Benchmark“ |
| E5 — Transfer | Weitere Modellfamilie oder reale Aufgabe | „Hinweis auf breitere Übertragbarkeit“ |

Aktuell: **zwischen E1 und E2.** v10 liefert Exploration; erst Holdout-Seeds und der vollständige Endlauf entscheiden über E2.

## 4. Standardprotokoll

Jeder Run muss vor Start in diesem Dokument bzw. in seinem README festlegen:

- Forschungsfrage und konkreter Hypothesentest
- Version, Git-/Dateistand, Python/PyTorch-Gerät und Startzeit
- Aufgaben und Datengenerierung
- Modellbreite, Tiefe, Residualregel und Initialisierung
- Trainingsbudget, Optimierer, Lernraten, Batchgröße und Abbruchregel
- Kandidaten und Baselines
- Hyperparameterbudget **pro Familie**
- Seeds: Explorationsseeds getrennt von Holdout-Seeds
- Primäre Metrik: Median-Testloss pro Konfiguration
- Sekundäre Metriken: IQR, Seed-Siege, NaN/Inf, Gradienten, Aktivierungsmaxima, Laufzeit
- Vorab festgelegte Entscheidungsregel

### Tuning-Fairness

Mehr Tuning für eine neue Idee ist als Exploration erlaubt, muss aber offengelegt werden. Für eine starke Vergleichsaussage gilt mindestens eines:

- gleiches Hyperparameterbudget je Aktivierungsfamilie;
- oder ein getrenntes Auswahl-/Testprotokoll: Auswahl auf Explorationsseeds, Bewertung auf frischen Holdout-Seeds;
- oder eine Budgetkurve: beste Leistung gegen Anzahl getesteter Konfigurationen.

## 5. Run-Register
| Run | Frage | Status | Entscheidungsgate | Artefakte |
|---|---|---|---|---|
| v9 / v9b | Erste CapExp-/PreCap-Erkundung | abgeschlossen | Grundlage für v10 | `v9/`, `v9b/` |
| v10 | Tiefe, `res_inv`, PreCap-Verfeinerung auf drei 1D-Aufgaben | läuft | Vollauswertung + Holdout | `v10/` |
| v11 mini | Flache Schutzlogik gegen tiefe Robustheit | pausiert | Nur nach v10-Endentscheidung starten | `v11/` |
| v11 full | Breiter Sweep nur für vielversprechende v11-Kandidaten | geplant | Nur nach bestandenem v11 mini | `v11/` |
| v12 2D | Übertragung auf 2D-Synthese | geplant | Erst nach E2 | noch anzulegen |
| Matrix v1 | Lineare/Faktorisierungsbeschränkung isoliert prüfen | Idee | Erst nach stabilem Aktivierungsbefund | noch anzulegen |

## 6. v10: Abschlussprotokoll

### Offene Schritte

1. Warten, bis alle 6.360 v10-Runs abgeschlossen sind und `v10_done.txt` existiert.
2. `python analyze_v10.py` im Ordner `v10/` ausführen; dadurch entsteht `analysis_full/`.
3. Integrität prüfen: erwartete Zeilenanzahl, keine Duplikate, keine fehlenden Konfigurationen, Shards lesbar.
4. Qualität zusammenfassen: Aufgabe × Tiefe × Architektur × Aktivierung, mit Konfigurationsmedianen.
5. Tiefe 32/64 separat bewerten: Loss, IQR, Seed-Streuung, Gradienten, Aktivierungsmaxima, Kappungsanteil.
6. Faire Vergleichsanalyse: Auswahlkonfigurationen nicht auf denselben Holdout-Seeds bewerten.
7. V10-Endnotiz schreiben: Ergebnisse, Einschränkungen, Entscheidung zu v11 mini.

### Holdout-Protokoll für v10

Vor der endgültigen Aussage werden die je Aufgabe/Tiefe/Architektur ausgewählten v10-Konfigurationen für PreCap, SiLU und GELU mit **10 frischen Seeds** evaluiert. Die Auswahl bleibt fix; es wird nicht nachgetunt.

**Mindestkriterium für „tiefer PreCap-Befund trägt“:**

- PreCap gewinnt oder bindet auf mindestens 2 von 3 Aufgaben bei Tiefe 32 und 64;
- keine erhöhte Ausfallrate;
- medianer Gewinn oder Gleichstand auf Holdout-Seeds, nicht nur auf Explorationsseeds;
- kein einzelner Ausreißer-Seed erklärt das Ergebnis.

## 7. v11: Entscheidungsprotokoll

### Zweck

v11 prüft, ob `precap_wg` und `hybrid_gate` eine bessere flache Leistung ermöglichen, ohne den vermuteten Tiefenvorteil zu verlieren.

### Mini-Run

Der Mini-Run ist ein Filter, kein Beweis.

- Aufgaben: dieselben drei 1D-Aufgaben
- Tiefen: 8, 16, 32, 64
- Kandidaten: PreCap-Referenz, `precap_wg`, `hybrid_gate`, SiLU, GELU
- Architekturen: mindestens die passende Referenzregel, insbesondere `res_inv` tief
- Seeds: fest dokumentiert, danach frische Holdout-Seeds für Gewinner

### Go / No-Go

| Ergebnis im Mini-Run | Entscheidung |
|---|---|
| Flach klar besser, tief innerhalb von 10 % der PreCap-Referenz und stabil | v11 full rechtfertigen |
| Tief deutlich schlechter oder neue Instabilität | Variante stoppen oder Formel ändern |
| Nur auf einer Aufgabe besser | Kein v11 full; gezielte Diagnose statt Sweep |
| Gewinn trägt auf Holdout-Seeds | Kandidat in 2D aufnehmen |

## 8. Nächste Verallgemeinerung: 2D

### Ziel

Nicht sofort Standarddatensätze jagen. Zuerst kontrollierte 2D-Aufgaben, bei denen lokale Krümmung, Frequenzmischung und Singularitäten sichtbar getrennt werden können.

### Kandidaten

- separable und nichtseparable Mischungen: \(\sin(kx)\sin(ly)\), \(\sin(xy)\)
- radiale Funktionen: \(\exp(-r^2)\), Ring-/Bump-Funktionen
- stückweise glatte Funktionen: \(|x|+|y|\), Kanten und Diagonalen
- anisotrope Frequenzen: hohe Frequenz in einer, niedrige in der anderen Achse
- Kombinationen aus glatter Grundfunktion und lokaler Struktur

### Messung

Neben Testloss werden Fehlerkarten über einem festen 2D-Gitter gespeichert. Für jede Methode: Median-Fehlerkarte, Worst-Seed-Karte, Lerndynamik und Stabilitätsmetriken.

## 9. Matrix-/Strukturprogramm

Die Matrixidee bleibt eine eigene Forschungsachse: keine Vermischung mit dem Aktivierungsbeweis.

### Reihenfolge

1. **Baseline fixieren:** PreCap/SiLU/GELU unter einer festen tiefen Architektur.
2. **Ein Faktor je Experiment:** nur eine strukturelle Änderung gleichzeitig.
3. **Ablation:** Aktivierung × Struktur × Residualskalierung.
4. **Holdout:** nur die beste Strukturvariante mit neuen Seeds bestätigen.

### Erste sichere Varianten

- Skalierte Matrix: \(W = \alpha \tilde W\), mit festem oder lernbarem \(\alpha\)
- Diagonale Vor-/Nachskalierung: \(W = D_1 \tilde W D_2\)
- Niedrigrang-Faktorisierung: \(W = AB\), mit klarer Parameterkontrolle
- Spektral eingeschränkte oder orthogonalisierte Initialisierung
- Tiefenabhängige lineare Residualfaktoren zusätzlich zu `res_inv`
Eine Variante wird nur weiterverfolgt, wenn sie gegenüber gleicher Parameterzahl und gleichem Trainingsbudget einen wiederholbaren Vorteil liefert.

## 10. Dokumentationsvorlage nach jedem Run

Kopiere diesen Block in eine neue Datei unter `Notizen Zwischenstände/`.

```markdown
# [RUN-ID] — Kurzbericht

Datum:
Status: geplant / läuft / abgeschlossen / verworfen

## Frage

## Vorab-Hypothese

## Fixiertes Protokoll
- Aufgaben:
- Architektur:
- Kandidaten:
- Seeds:
- Budget:
- Primäre Metrik:

## Integrität
- Erwartete / vorhandene Runs:
- Duplikate:
- Fehlende Konfigurationen:
- Ausfälle:

## Befunde
- Beobachtung 1:
- Beobachtung 2:
- Negativer Befund:

## Grenzen
- Tuning-Fairness:
- Gültigkeitsbereich:
- Offene technische Risiken:

## Entscheidung
- Weiter / Stop / Replizieren:
- Nächster kleinstmöglicher Test:

## Artefakte
- Skript:
- Rohdaten:
- Auswertung:
- Grafiken:
```

## 11. Entscheidungslog

| Datum | Entscheidung | Begründung | Nächster Prüfpunkt |
|---|---|---|---|
| 2026-10-19 | v11 mini pausiert | v10 erhält volle Rechenleistung; v10 zuerst vollständig auswerten | v10-Endanalyse + Holdout |
| 2026-10-19 | v10-Auswertung in Stufen | Integrität, Qualität, Stabilität, faire Paare trennen | Vollständigkeit 6.360/6.360 |

## 12. Regeln gegen Selbsttäuschung

- Keine Aussage „besser“, wenn sie nur auf einer Aufgabe, einem Seed oder einem nachträglich gewählten Ausschnitt beruht.
- Keine neue Formel während eines laufenden Vergleichs; Änderungen erzeugen eine neue Version.
- Negative Ergebnisse bleiben dokumentiert.
- Grafiken sind Diagnosewerkzeuge, keine Beweise ohne zugrunde liegende Tabelle.
- Jede starke Behauptung braucht eine reproduzierbare Konfiguration, Rohdaten und frische Holdout-Seeds.
- Ein guter Stopp ist Erfolg: Er verhindert einen großen Sweep für eine unklare Idee.

