# V17 – Ausführliche Auswertung (Stand 25.10.2026)

Datenbasis: `workers/w*_checkpoints.csv` (11520 Zeilen, alle `run_status=ok`, 1440 Runs). Die Zahlen unten sind explorativ nachgerechnet; die konfirmatorischen Tests stehen unverändert in `v17_confirmatory.csv`.

## 1. Absolute Test-MSE (Median über 20 Seeds, Schritt 3000)

Tiefe 128, konstante Skala:

| Aufgabe | GELU | PreCap-base | SiLU |
|---|---:|---:|---:|
| corner_kink | 27.9 | 0.0033 | 7.27 |
| radial_ripple | 26.8 | 0.017 | 4.12 |
| ridge_mix | 18.0 | 0.0119 | 7.21 |
| smooth_bowl | 21.6 | 0.0049 | 10.0 |

Anteil der Seeds mit Test-MSE > 1 bei Tiefe 128, konstante Skala: GELU 100 %, SiLU 95 %, PreCap 0 %.
Der Start-MSE (Schritt 0) liegt dort bei ca. 1e18 (GELU/SiLU) gegenüber ca. 3.8e3 (PreCap). Das Signal explodiert im Vorwärtspfad bei Residual-Skala 1 und Tiefe 128; der Cap begrenzt das bei PreCap.

## 2. Was das für den Befund "PreCap besser bei Tiefe 128, constant" bedeutet

Die 8 Signifikanzen zugunsten von PreCap bei konstanter Skala (Tiefe 128) beruhen darauf, dass SiLU und GELU nicht konvergieren, nicht darauf, dass PreCap gut approximiert. Mit Test-MSE 0.003 bis 0.017 ist PreCap dort 6- bis 115-mal schlechter als PreCap selbst mit inv_sqrt (Median-Verhältnis je Aufgabe) und bis zu rund 570-mal schlechter als das beste gedämpfte Netz (smooth_bowl: 0.0049 gegen 8.5e-6 bei inv_quart). Die Präregistrierung schließt solche Fälle aus ("ohne dass die Verbesserung ausschließlich auf instabile/fehlende Runs beruht").

Bei Tiefe 64, konstante Skala sind alle Netze stabil, und PreCap verliert auf allen vier Aufgaben (MSE 0.003 bis 0.016 gegen 2e-5 bis 4.4e-4 bei SiLU/GELU). Der PreCap-Fehler steigt dort im Training sogar (Median über Aufgaben 0.00245 bei Schritt 400, 0.00608 bei Schritt 3000), und die Gradientennorm ist mit 39 etwa 70- bis 170-mal so groß wie bei SiLU/GELU (0.23 bis 0.57). Bei PreCap liegen 84 % (Tiefe 64) bzw. 92 % (Tiefe 128) der Voraktivierungen über 0.5, und 36 % bzw. 43 % liegen im Cap-Bereich (gedämpft: 0.2 bis 0.7 %). Der Cap wird hier also massiv genutzt, und das schadet.

## 3. Entscheidungsregel H2 (präregistriert)

Regel: mindestens 3 von 4 Aufgaben mit kleinerem Verhältnis unter inv_quart oder constant als unter inv_sqrt, ohne Abhängigkeit von instabilen Runs.

- inv_quart: höchstens 2 von 4 Aufgaben (Tiefe 128: corner_kink und smooth_bowl, bei SiLU wie bei GELU; Tiefe 64: 1 von 4). Regel verfehlt.
- constant: Tiefe 128 formal 4 von 4, aber nur wegen divergierender Referenzen (siehe 2). Zählt nach der Regel nicht.

**H2 ist nicht gestützt.** Die Zwischenaussage, das Kriterium sei bei Tiefe 128/constant erfüllt, ist damit zurückzunehmen.

## 4. Effekt weniger Dämpfung auf jede Aktivierung

Mittlere Änderung von log10(MSE), inv_quart minus inv_sqrt (negativ = weniger Dämpfung hilft), gepaart über Seeds:

| Tiefe | GELU | PreCap | SiLU |
|---|---:|---:|---:|
| 64 | -0.26 | -0.29 | -0.44 |
| 128 | -0.41 | -0.36 | -0.37 |

Weniger Dämpfung (D^-1/4) hilft allen drei Aktivierungen etwa gleich (rund Faktor 2 bis 3 weniger Fehler). Der Unterschied zwischen PreCap und SiLU bzw. GELU ist nicht signifikant (Wilcoxon über 80 gepaarte Werte pro Tiefe: p = 0.12 / 0.53 gegen SiLU, 0.65 / 0.9 gegen GELU). Es gibt also keine PreCap-spezifische Reaktion auf mehr Aktivierungsamplitude.

## 5. Bester Betrieb pro Aufgabe (Median-MSE)

| Aufgabe | Tiefe 64 | Tiefe 128 |
|---|---|---|
| corner_kink | PreCap, inv_quart (7.3e-5) | PreCap, inv_quart (9.8e-5) |
| smooth_bowl | PreCap, inv_quart (6.5e-6) | PreCap, inv_quart (8.5e-6) |
| radial_ripple | SiLU, constant (1.8e-4) | SiLU, inv_quart (2.9e-4) |
| ridge_mix | SiLU, constant (1.5e-4) | GELU, inv_quart (1.5e-4) |

Bei inv_quart (Mittel von log10 MSE über vier Aufgaben) liegt PreCap bei Tiefe 64 leicht vorn (-3.99 gegen -3.87 GELU / -3.88 SiLU) und bei Tiefe 128 leicht hinten (-3.69 gegen -3.93 GELU / -3.70 SiLU). Alle diese Unterschiede sind nach Holm nicht signifikant, die Konfidenzintervalle sind weit.

## 6. Amplitude gegen Fehler

Die Korrelation post_rms_scaled gegen Test-MSE (Spearman 0.71 auf 24 Zellen, aus verdict.json) wird von der konstanten Skala getrieben: höhere Amplitude geht dort mit schlechterem Fehler einher. Ohne constant: seed-weise rho = 0.26 (p = 2e-6, n = 320), auf Zellebene rho = 0.20 (p = 0.45, n = 16). Das ist nicht die von H2 erwartete Richtung (mehr Nichtlinearitätsnutzung = besser).

## 7. Fazit

1. Der Residual-Dämpfungs-Mechanismus (H2) erklärt nicht, warum PreCap in tiefen Netzen den Vorsprung verliert. Weniger Dämpfung hilft allen Aktivierungen gleich.
2. PreCap hat eine klar belegte Eigenschaft: Der Cap hält ein ungedämpftes 128-Schichten-Netz numerisch trainierbar, wo SiLU und GELU divergieren. Absolut ist das Ergebnis aber deutlich schlechter als jedes gedämpfte Netz.
3. Ohne Dämpfung bei Tiefe 64 verliert PreCap deutlich, weil es dauerhaft im Cap arbeitet (36 % der Werte).
4. Bei sinnvoller Skalierung (inv_sqrt, inv_quart) gibt es keinen belastbaren Unterschied zwischen PreCap und SiLU/GELU: 2 Vorteile bei inv_sqrt (corner_kink gegen SiLU, beide Tiefen), 1 bei inv_quart (corner_kink, Tiefe 128, gegen SiLU), keine Nachteile; alles Übrige unklar.
5. Aufgabenabhängig liegt PreCap bei corner_kink und smooth_bowl vorn, SiLU/GELU bei radial_ripple und ridge_mix (explorativ, mediane MSE).

Geltungsbereich: 2D-Synthetikaufgaben, res_inv-Architektur, Adam 0.001 konstant, 3000 Schritte, 20 Seeds, kein Gewichtsabfall und keine Lernraten-Anpassung.
