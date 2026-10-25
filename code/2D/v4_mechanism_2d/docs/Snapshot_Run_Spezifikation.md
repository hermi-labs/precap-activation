# Snapshot-Run-Spezifikation — Entwurf

## Zweck

Gezielte Re-Runs erzeugen Modell-Snapshots und schichtweise Metriken für die Mechanismusanalyse. Diese Runs sind **diagnostisch**, nicht Teil einer neuen primären Wirksamkeitsentscheidung.

## Pro Run zu speichern

- Metadaten: Aufgabe, Tiefe, Aktivierung, Seed, Formelparameter, Git/Datei-Hash
- Checkpoints: Schritte 0, 100, 500, 1.500, 3.000
- Modellzustand: `state_dict`
- Lernkurve: Trainings- und Test-MSE pro Checkpoint
- Global: Gradienten-Norm, Aktivierungspeak, Cap-Anteil
- Pro Layer: Pre-Activation/Activation/Residual-Statistik (Mittelwert, Standardabweichung, p01, p50, p99, max abs)
- Optional: ausgewählte Neuronen-Aktivierung auf festem 2D-Gitter

## Reproduzierbarkeit

- Fester Trainings- und Testsatz pro Seed
- Feste Checkpoint-Schritte
- Einheitliches Serialisierungsformat
- Separate Indexdatei, die jedes Snapshot-Artefakt mit Konfiguration verbindet

## Ressourcenregel

Snapshots nur für die vordefinierten diagnostischen Zellen und wenige Seeds speichern. Kein Voll-Sweep mit vollständigen Tensor-Dumps.
