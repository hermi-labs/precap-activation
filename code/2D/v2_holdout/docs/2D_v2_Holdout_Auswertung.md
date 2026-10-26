# CapExp 2D v2 — unabhängiger Holdout: Endauswertung

**Präregistrierter Status:** Die Formel, Architektur, Aufgaben, Seeds, Trainingsprotokoll, Metriken und PASS-Regel waren vor Abschluss eingefroren (`meta/manifest.json`).

## Integrität

- Ergebniszeilen: **270/270**.
- Vollständige gepaarte Zellen (Aufgabe × Tiefe × Seed): **90/90**.
- `bad=True`: **0**.
- Jede Zelle enthält genau PreCap, SiLU und GELU: **ja**.

Primärmetrik: \$d=\log_{10}(\mathrm{MSE}_{\mathrm{PreCap}}/\mathrm{MSE}_{\mathrm{Ref}})\$. Negative Werte begünstigen PreCap. Klammern zeigen PreCap-Siege / gepaarte Seeds.

## Primäre Resultate nach Aufgabe

| Aufgabe | gegen SiLU | gegen GELU | Gate gegen beide |
|---|---|---|---|
| radial_ripple | -0.793 (25/30) | -0.781 (29/30) | PASS |
| ridge_mix | -0.579 (28/30) | -0.355 (23/30) | PASS |
| corner_kink | -0.389 (30/30) | +0.027 (12/30) | FAIL |

## Resultate nach Tiefe

| Tiefe | gegen SiLU | gegen GELU |
|---:|---|---|
| 4 | -0.751 (30/30) | -0.481 (25/30) |
| 8 | -0.581 (28/30) | -0.366 (21/30) |
| 16 | -0.328 (25/30) | -0.072 (18/30) |

## Aufgabe × Tiefe

| Aufgabe | Tiefe | gegen SiLU | gegen GELU |
|---|---:|---|---|
| radial_ripple | 4 | -0.805 (10/10) | -0.713 (10/10) |
| radial_ripple | 8 | -0.953 (10/10) | -0.994 (10/10) |
| radial_ripple | 16 | -0.025 (5/10) | -0.438 (9/10) |
| ridge_mix | 4 | -0.880 (10/10) | -0.481 (9/10) |
| ridge_mix | 8 | -0.402 (8/10) | -0.239 (7/10) |
| ridge_mix | 16 | -0.415 (10/10) | -0.123 (7/10) |
| corner_kink | 4 | -0.516 (10/10) | -0.116 (6/10) |
| corner_kink | 8 | -0.329 (10/10) | +0.031 (4/10) |
| corner_kink | 16 | -0.359 (10/10) | +0.054 (2/10) |

## Replikationsstärke

| Referenz | PreCap besser | einseitiger gepaarter Sign-Test |
|---|---:|---:|
| silu | 83/90 | 6.58e-18 |
| gelu | 64/90 | 3.83e-05 |

## Stabilität und Intervention

| Referenz | Median hmax_peak PreCap/Ref | Median grad_final PreCap/Ref |
|---|---:|---:|
| silu | 0.665 | 0.713 |
| gelu | 0.780 | 0.795 |

Median finaler Cap-Anteil: **0.030**.

## Präregistrierte Entscheidung

- Aufgaben mit PASS gegen beide Referenzen: **2/3**.
- Höhere PreCap-bad-rate: **nein**.
- **Entscheidung: PASS.**

Interpretation: Der Holdout bestätigt die Pilotentscheidung nur im Rahmen dieser festen 2D-Syntheseaufgaben und dieser eingefrorenen Trainingskonfiguration. Er ist kein allgemeiner Beweis für jede Architektur, Datenverteilung oder Tiefe.