#!/bin/bash
# V1b starten (mit Claude): Netzteil pruefen, einfrieren, Selbsttest, Mechanik- und Auswertungstest, Rauchtest, dann 2 Worker in screen (v1bw0, v1bw1).
cd "$(dirname "$0")"
PY=${PY:-$HOME/precap_mlx_venv/bin/python}; PYS=${PYS:-$HOME/PyCharmMiscProject/.venv/bin/python}
D=$HOME/PyCharmMiscProject/data/cifar10/cifar-10-batches-bin
mkdir -p meta logs
if [ -d workers ] || [ -d tune ]; then echo "Ergebnisordner existiert schon - Abbruch" > logs/launch.txt; exit 1; fi
if screen -ls | grep -q -E "lg[0-9]w|d2w|d2bw|b[0-9]+w|v1w|v1bw"; then echo "andere Worker laufen - Abbruch" > logs/launch.txt; exit 1; fi
if ! pmset -g batt | grep -q "AC Power"; then echo "nicht am Netzteil - Abbruch" > logs/launch.txt; exit 1; fi
shasum -a 256 v1b_run.py v1b_analyze.py test_v1b_mechanik.py test_v1b_analyse.py V1b_Praeregistrierung.md src/precap/*.py $D/*.bin > meta/frozen_sha256.txt
date '+%Y-%m-%dT%H:%M:%S%z' > meta/frozen_at.txt
$PY v1b_run.py --selfcheck > meta/selfcheck.json 2>/dev/null || { echo "Selbsttest fehlgeschlagen" > logs/launch.txt; exit 1; }
grep -q '"n_tune": 32' meta/selfcheck.json || { echo "Selbsttest-Werte nicht ok" > logs/launch.txt; exit 1; }
$PY test_v1b_mechanik.py > meta/mechanik.txt 2>&1 || { echo "Mechanik-Test fehlgeschlagen" > logs/launch.txt; exit 1; }
$PYS -I test_v1b_analyse.py > meta/auswertungstest.txt 2>&1 || { echo "Auswertungstest fehlgeschlagen" > logs/launch.txt; exit 1; }
[ -d smoke ] && mv smoke "smoke_alt_$(date +%H%M%S)"
$PY v1b_run.py --smoke --worker 0 --workers 1 > logs/smoke.log 2>&1
t=$(grep -c ',ok,' smoke/tune/workers/w0.csv 2>/dev/null)
if [ "$t" != "4" ]; then echo "Rauchtest nicht ok (tune $t) - kein Start" > logs/launch.txt; exit 1; fi
echo "Tests ok $(date '+%H:%M')" > logs/launch.txt
for w in 0 1; do screen -dmS v1bw$w caffeinate -i bash -c "{ $PY v1b_run.py --worker $w --workers 2; } >> logs/w$w.log 2>&1"; done
sleep 2; { echo "gestartet $(date)"; screen -ls; } >> logs/launch.txt
