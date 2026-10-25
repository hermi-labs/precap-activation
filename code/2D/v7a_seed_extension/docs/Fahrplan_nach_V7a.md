# Fahrplan nach v7a

## Entscheidung
Keine neue globale c/s-Kalibrierung. PreCap-base (c=1, s=1.5) bleibt Referenz.

## Nächste Studie: v7b — Tiefe als primäre Achse
Ziel: Verstehen, warum s225 bei radial d16 konsistent gewinnt, bei d64 aber nicht robust gewinnt.

### Vorschlag (vor Start finalisieren)
- Aufgaben: radial_ripple als primäre Aufgabe; ridge_mix als Replikation.
- Tiefen: 8, 16, 32, 64.
- Varianten: base, s225, SiLU, GELU. c125 nicht erneut aufnehmen, weil v7a klar negativ ist.
- Seeds: mindestens 20 neue Seeds pro Zelle, unabhängig von 900–1019.
- Identisches Netz, gleiche Trainingsdauer; optional längere Trainingsdauer erst als separat eingefrorene zweite Achse.
- Primär: s225 vs base über die Tiefe. Hierarchische Kriterien vorab: Vorteil in mindestens zwei benachbarten Tiefen oder klare Interaktion Tiefe×Variante; Anker und Referenzen berichten.

## Parallel: Mechanismus aus vorhandenen Daten
- Testen, ob der s225-Vorteil bei d16 mit späterem, stabilerem Fehlerabfall statt kleineren Peaks verbunden ist.
- Layer-/Snapshot-Wiederholung nur für d16 s225/base und d64 s225/base, mit Seeds aus beiden Richtungen.
- Nicht behaupten, dass Peak-Reduktion der Mechanismus ist: V6 widerlegt diese einfache Erklärung.

## Danach
Eine adaptive c/s-Regel erst dann erwägen, wenn v7b eine klare Tiefe×s-Interaktion repliziert. Eine mögliche Hypothese wäre nicht „dynamisch ist besser“, sondern „die optimale Sättigungsbreite hängt systematisch von der effektiven Tiefe ab“.
