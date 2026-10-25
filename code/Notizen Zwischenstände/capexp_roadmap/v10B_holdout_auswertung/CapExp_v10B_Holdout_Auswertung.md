# CapExp v10B — Holdout-Auswertung

## Ergebnis
Der eingefrorene Holdout umfasst **180 von 180** vollständig validierte Runs ohne Ausfälle, fehlende Einträge oder Duplikate. Die Gate-A-Bewertung lautet: **PASS**. Diese Entscheidung basiert auf seed-gepaarten Vergleichen von PreCap gegen SiLU und GELU bei drei Aufgaben und den Tiefen 32/64.

## Versuchsdesign

- Aufgaben: `sin3_quad`, `sin_mix`, `abs_wave`
- Architektur: `res_inv`; Tiefen: 32 und 64
- Aktivierungen: PreCap, SiLU, GELU
- Frische, gepaarte Seeds: 10–19; 10 Runs pro Zelle
- Primärmetrik: Test-MSE; kleiner ist besser

## Mathematische Auswertung

Für jeden Seed wird der gepaarte Fehlerunterschied berechnet:

\[ \Delta_s = \operatorname{MSE}_{\mathrm{PreCap},s} - \operatorname{MSE}_{\mathrm{Baseline},s}. \]

Ein negativer Wert bedeutet einen PreCap-Vorteil. Ergänzend verwenden wir das Seed-Verhältnis

\[ r_s = \frac{\operatorname{MSE}_{\mathrm{PreCap},s}}{\operatorname{MSE}_{\mathrm{Baseline},s}}. \]

Ein Medianverhältnis unter 1 spricht für PreCap. Der Wilcoxon-Test ist zweiseitig und wird hier als deskriptiver, gepaarter Evidenzindikator berichtet; die Gate-Entscheidung folgt vor allem Richtung, Größenordnung und Konsistenz über die vorab festgelegten Zellen.

## Zellübersicht

| Aufgabe | Tiefe | Aktivierung | Median Test-MSE | Q1–Q3 | Ausfälle |
|---|---:|---|---:|---:|---:|
| abs_wave | 32 | gelu | 0.000605575 | 0.000505007–0.00109782 | 0 |
| abs_wave | 32 | precap | 9.44558e-05 | 6.61644e-05–0.000134544 | 0 |
| abs_wave | 32 | silu | 0.000431788 | 0.000217644–0.000655634 | 0 |
| abs_wave | 64 | gelu | 0.000447813 | 0.000323805–0.00381936 | 0 |
| abs_wave | 64 | precap | 8.94533e-05 | 6.06246e-05–0.000268111 | 0 |
| abs_wave | 64 | silu | 0.000238696 | 0.000160473–0.000276548 | 0 |
| sin3_quad | 32 | gelu | 0.000255722 | 0.00021982–0.00070279 | 0 |
| sin3_quad | 32 | precap | 7.88954e-06 | 5.68023e-06–2.11006e-05 | 0 |
| sin3_quad | 32 | silu | 0.000254258 | 6.4982e-05–0.00215024 | 0 |
| sin3_quad | 64 | gelu | 0.000277684 | 0.000175023–0.000524577 | 0 |
| sin3_quad | 64 | precap | 5.25544e-05 | 3.79413e-05–8.48813e-05 | 0 |
| sin3_quad | 64 | silu | 8.38351e-05 | 6.11338e-05–0.000142677 | 0 |
| sin_mix | 32 | gelu | 0.00130954 | 0.000907993–0.00153795 | 0 |
| sin_mix | 32 | precap | 0.000625066 | 0.000514656–0.000639863 | 0 |
| sin_mix | 32 | silu | 0.00159147 | 0.0013368–0.00167211 | 0 |
| sin_mix | 64 | gelu | 0.00136111 | 0.000929925–0.00159962 | 0 |
| sin_mix | 64 | precap | 0.000440236 | 0.000366439–0.000554208 | 0 |
| sin_mix | 64 | silu | 0.00172201 | 0.0015067–0.00180834 | 0 |

## Gepaarte Vergleiche

| Aufgabe | Tiefe | Vergleich | Median-Verhältnis | PreCap-Siege | Baseline-Siege | Wilcoxon p |
|---|---:|---|---:|---:|---:|---:|
| abs_wave | 32 | PreCap vs. SILU | 0.3185 | 10/10 | 0/10 | 0.001953 |
| abs_wave | 32 | PreCap vs. GELU | 0.1624 | 10/10 | 0/10 | 0.001953 |
| abs_wave | 64 | PreCap vs. SILU | 0.4036 | 7/10 | 3/10 | 0.625 |
| abs_wave | 64 | PreCap vs. GELU | 0.215 | 8/10 | 2/10 | 0.08398 |
| sin3_quad | 32 | PreCap vs. SILU | 0.05207 | 10/10 | 0/10 | 0.001953 |
| sin3_quad | 32 | PreCap vs. GELU | 0.02171 | 10/10 | 0/10 | 0.001953 |
| sin3_quad | 64 | PreCap vs. SILU | 0.5456 | 7/10 | 3/10 | 0.2324 |
| sin3_quad | 64 | PreCap vs. GELU | 0.1949 | 10/10 | 0/10 | 0.001953 |
| sin_mix | 32 | PreCap vs. SILU | 0.4222 | 10/10 | 0/10 | 0.001953 |
| sin_mix | 32 | PreCap vs. GELU | 0.4323 | 10/10 | 0/10 | 0.001953 |
| sin_mix | 64 | PreCap vs. SILU | 0.2715 | 10/10 | 0/10 | 0.001953 |
| sin_mix | 64 | PreCap vs. GELU | 0.4578 | 10/10 | 0/10 | 0.001953 |

## Gate A je Aufgabe/Tiefe

| Aufgabe | Tiefe | PreCap vor beiden Baselines? | Median-Verhältnis vs. SiLU | Median-Verhältnis vs. GELU |
|---|---:|---|---:|---:|
| abs_wave | 32 | Ja | 0.3185 | 0.1624 |
| sin3_quad | 32 | Ja | 0.05207 | 0.02171 |
| sin_mix | 32 | Ja | 0.4222 | 0.4323 |
| abs_wave | 64 | Ja | 0.4036 | 0.215 |
| sin3_quad | 64 | Ja | 0.5456 | 0.1949 |
| sin_mix | 64 | Ja | 0.2715 | 0.4578 |

## Interpretation und nächster Schritt

Die vorab definierte Mindestbedingung ist erfüllt: In 6 von 6 Aufgabe-Tiefe-Zellen ist PreCap gegen beide Baselines nach Median und Seed-Siegzahl vorn; es traten keine numerischen Ausfälle auf. Damit ist der Übergang zu **v11 mini** gerechtfertigt. v11 mini soll nicht den v10B-Befund nachoptimieren, sondern die Varianten `precap_wg` und `hybrid_gate` mit kleinem, symmetrischem Budget testen — insbesondere für die flachen Regime.

## Reproduzierbarkeit

- Rohresultate: `v10B_holdout/results/holdout_results_shard*.csv`
- Eingefrorenes Manifest: `v10B_holdout/meta/manifest.csv`
- Ergebnisprüfung: `v10B_holdout/logs/validate.log`
- Maschinenlesbare Tabellen: `v10B_zelluebersicht.csv`, `v10B_gepaarte_vergleiche.csv`
