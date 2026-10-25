# CapExp 2D v2 — unabhängiger Holdout

Dieser Lauf bestätigt oder verwirft den gültigen 2D-v1b-Pilot unabhängig. Alle Designentscheidungen sind aus v1b eingefroren: echte 1D-v9-PreCap-Formel, residuale Architektur, drei Aufgaben, Breite 64, Tiefen 4/8/16 und 3.000 Schritte.

## Was ist unabhängig?

Nur die Seeds: v1b nutzte 300–304; v2 verwendet neue Seeds 400–409. Es gibt 270 Runs (3 Aufgaben × 3 Tiefen × 3 Aktivierungen × 10 Seeds). Alle Parameter, Auswertung und PASS-Regeln stehen vor Start im Manifest fest.

## Start

```zsh
cd /Users/aaronschmitt/PyCharmMiscProject/2D/v2_holdout
./launch.sh
```

## Status

```zsh
/Users/aaronschmitt/PyCharmMiscProject/.venv/bin/python status.py
tail -20 logs/progress.log
```
