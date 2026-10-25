# v10-Auswertung – Fahrplan

## Aktueller Modus

- v10 hat Priorität und läuft bis zur Datei `v10_done.txt`.
- v11 mini ist bewusst pausiert, damit keine CPU-Leistung von v10 abgezogen wird.
- Ein Anti-Ruhezustand-Schutz hält den Mac wach, solange v10 noch nicht abgeschlossen ist.

## Reihenfolge nach v10

1. **Integrität prüfen:** `v10_done.txt`, alle sieben Ergebnis-Shards, erwartete Gesamtzahl von 6.360 Konfigurationen, doppelte oder unvollständige Zeilen, Kollaps-/Fehlerfälle.
2. **Hauptwertung:** Median, Mittelwert, beste Werte und Streuung des Test-Loss, getrennt nach Aufgabe, Tiefe, Architektur, Aktivierung, Initialisierung und Lernrate.
3. **Stabilität:** Anteil fehlgeschlagener Läufe, Gradienten- und Aktivierungsdiagnostik aus den Kurven; Tiefe 32/64 besonders gewichten.
4. **Fairer Vergleich:** SiLU, GELU, CapExp und PreCap unter gleicher Aufgabe, Tiefe, Architektur, `init_k`, Lernrate und Seed vergleichen.
5. **Gewinner bestimmen:** Nicht nur Best-Seed. Hauptkriterium ist robuste Leistung über Seeds plus Stabilität in Tiefe.
6. **Entscheidung v11:** v11 mini erst wieder starten, wenn die v10-Referenzwerte dokumentiert sind; v11 full nur bei klarer Verbesserung oder klarer zusätzlicher Stabilität.

## Bewertungsbogen

| Dimension | Leitfrage | Gewicht |
|---|---|---:|
| Test-Loss | Ist die Aktivierung im Median gut? | Hoch |
| Tiefe | Funktioniert sie bei 32 und 64 Layern? | Sehr hoch |
| Stabilität | Gibt es Kollaps, NaNs oder starke Ausreißer? | Sehr hoch |
| Seed-Robustheit | Bleibt der Effekt über Seeds bestehen? | Hoch |
| Architektur | Verbessert `res_inv` die Aussage konsistent? | Mittel |
| Einfachheit | Rechtfertigt ein neuer Mechanismus seine Komplexität? | Mittel |

## Arbeitsprodukt

Die Auswertung soll liefern:

- eine aggregierte Rangliste je Aufgabe und Tiefe,
- Tabellen für die fairen Paarvergleiche,
- eine Kollaps-/Stabilitätsübersicht,
- ein kurzes Forschungsfazit mit Entscheidung für PreCap, v11 oder weitere Matrix-Ablationen.
