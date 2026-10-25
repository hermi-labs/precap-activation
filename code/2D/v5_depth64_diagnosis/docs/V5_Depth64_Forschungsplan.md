# CapExp v5 — Depth-64-Diagnoseplan

## Ziel

Die nächste Forschungsphase fokussiert **nicht** auf einen weiteren breiten 2D-Sweep, sondern auf die präzise Diagnose der Tiefe-64-Grenze. Ausgangspunkt ist der v4-Befund: Bei `radial_ripple` kippt PreCap bei Tiefe 64 am Endpunkt gegen beide Referenzen, obwohl die Aktivierungspeaks kleiner bleiben. Damit ist die Arbeitsfrage nicht mehr "ob" es eine Grenze gibt, sondern **welcher Mechanismus** sie erzeugt.

## Warum kein großer neuer Sweep?

Ein großer Sweep würde vor allem mehr Endmetriken liefern. Die aktuellen Daten enthalten bereits fünf Checkpoints, layerweise Verteilungen und gepaarte Seeds. Damit lässt sich jetzt deutlich gezielter prüfen,

- wann die Dynamik kippt,
- in welchen Layern die Abweichung zuerst sichtbar wird,
- ob das Kippen primär an Gradienten, Residualakkumulation oder Zielstruktur hängt,
- und ob sich daraus eine kleine, saubere Interventionsstudie ableiten lässt.

## Phase 1 — Aus vorhandenen v4-Daten maximal lesen

Vor jedem neuen Run werden die v4-Daten tiefer ausgewertet:

1. **Zeitfensteranalyse 64:** Für `radial_ripple` und `ridge_mix` bei Tiefe 64 Schritt 100/500/1500/3000 gegen SiLU und GELU vergleichen.
2. **Layer-Frühwarnkarte:** Pro Layer prüfen, wo `std_ratio` und `max_abs_ratio` erstmals systematisch ansteigen oder kippen.
3. **Grenzkontrast 32 vs 64:** `radial_ripple` 16 als Gewinneranker und 64 als Grenzfall gegenüberstellen.
4. **Peak-Gradient-Entkopplung:** Fälle markieren, in denen kleinere Peaks trotzdem mit schlechterer Testleistung einhergehen.

## Phase 2 — Kleine gezielte Folge-Studie

Nur wenn Phase 1 ein klares Muster liefert, folgt ein minimaler neuer Lauf. Die Priorität ist:

- nur Tiefe 64,
- nur Grenzfälle plus ein stabiler Gewinneranker,
- wenige frische Seeds,
- mehrere Worker,
- explizit mit Schlafschutz.

### Geplante Zellen für v5-run (vorläufig)

| Klasse | Aufgabe | Tiefe |
|---|---|---:|
| Grenzfall | radial_ripple | 64 |
| Grenzfall | ridge_mix | 64 |
| Anker | radial_ripple | 16 |
| Gegenbeispiel | highfreq_grid | 32 |

### Was zusätzlich gespeichert würde

- dichtere Checkpoints im kritischen Fenster, z. B. 100, 300, 500, 800, 1200, 1500, 2200, 3000
- Gradient pro Layer statt nur global
- Residual-zu-Aktivierungs-Verhältnis pro Layer
- optional fester Input-Grid-Snapshot ausgewählter Neuronen

## Schlafschutz und Worker-Modell

Lange Läufe sollen künftig nicht mehr an Schlafmodus oder seriellem Ablauf hängen. Der Standard für neue Langläufe wird daher:

- `caffeinate` für Schlafschutz auf macOS,
- mehrere Worker mit disjunkten Shards,
- je Worker eigene CSV-Dateien,
- deterministischer Merge mit Integritätscheck,
- Wiederaufnahme über bereits vollständige Run-IDs.

## Entscheidungsregel

Wenn die Phase-1-Analyse bereits klar zeigt, dass das Tiefe-64-Problem mit steigenden Gradienten oder einer späten Layer-Instabilität zusammenfällt, ist ein weiterer breiter Sweep nicht nötig. Dann genügt eine kleine bestätigende Folge-Studie mit Tiefe 64 und Mehr-Worker-Ausführung.
