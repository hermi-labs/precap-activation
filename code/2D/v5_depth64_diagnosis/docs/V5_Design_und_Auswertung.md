# v5 — Design und Auswertung

## Frage

Bei Tiefe 64 verlor PreCap in v4 für `radial_ripple` den anfänglichen Testvorteil, obwohl die Peak-Aktivierung klein blieb. v5 untersucht, ob eine späte Zunahme globaler oder layerweiser Gradienten beziehungsweise ein ungünstiges Residual-/Aktivierungsverhältnis diesem Umschwung vorausgeht.

## Zellen

- Grenzfälle: `radial_ripple` 64, `ridge_mix` 64
- Anker: `radial_ripple` 16
- Gegenbeispiel: `highfreq_grid` 32

## Primäre deskriptive Kriterien

1. Zeitliche Reihenfolge: Unterschied in Gradient/Residual vor dem Umschwung des Testfehler-Quotienten.
2. Layerlokalisierung: wiederkehrende Auffälligkeit in frühen, mittleren oder späten Layern.
3. Spezifität: Grenzfälle gegenüber Anker und Gegenbeispiel unterscheiden.

## Worker und Schlafschutz

Vier deterministische Shards, eigene Worker-Dateien, danach Merge mit Eindeutigkeits- und Zeilenzahlprüfung. Der Launcher läuft unter `caffeinate -dimsu`, damit macOS nicht in den Schlafmodus geht, solange der Lauf aktiv ist.
