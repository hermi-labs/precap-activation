#!/bin/bash
# V1d starten (mit Claude): Netzteil pruefen, V1c/B15 gegen ihren Freeze pruefen, einfrieren, Selbsttest, Mechanik- und Auswertungstest, Rauchtest, dann 2 Worker in screen (v1dw0, v1dw1).
cd "$(dirname "$0")"
PY=${PY:-$HOME/precap_mlx_venv/bin/python}; PYS=${PYS:-$HOME/PyCharmMiscProject/.venv/bin/python}
D=$HOME/PyCharmMiscProject/data/cifar10/cifar-10-batches-bin
mkdir -p meta logs
if [ -d workers ]; then echo "Ergebnisordner existiert schon - Abbruch" > logs/launch.txt; exit 1; fi
if screen -ls | grep -q -E "\.[a-z0-9]+w[0-9]+[[:space:]]"; then echo "andere Worker laufen - Abbruch" > logs/launch.txt; exit 1; fi
if ! pmset -g batt | grep -q "AC Power"; then echo "nicht am Netzteil - Abbruch" > logs/launch.txt; exit 1; fi
(cd ../v1c && shasum -a 256 -c meta/frozen_sha256.txt > /dev/null) && (cd ../b15 && shasum -a 256 -c meta/frozen_sha256.txt > /dev/null) || { echo "V1c/B15 nicht wie eingefroren" > logs/launch.txt; exit 1; }
shasum -a 256 v1d_run.py v1d_analyze.py test_v1d_mechanik.py test_v1d_analyse.py V1d_Praeregistrierung.md launch_v1d.sh src/precap/*.py $D/*.bin > meta/frozen_sha256.txt
date '+%Y-%m-%dT%H:%M:%S%z' > meta/frozen_at.txt
$PY v1d_run.py --selfcheck > meta/selfcheck.json 2>/dev/null || { echo "Selbsttest fehlgeschlagen" > logs/launch.txt; exit 1; }
grep -q '"n_confirm": 26' meta/selfcheck.json || { echo "Selbsttest-Werte nicht ok" > logs/launch.txt; exit 1; }
$PY test_v1d_mechanik.py > meta/mechanik.txt 2>&1 || { echo "Mechanik-Test fehlgeschlagen" > logs/launch.txt; exit 1; }
$PYS -I test_v1d_analyse.py > meta/auswertungstest.txt 2>&1 || { echo "Auswertungstest fehlgeschlagen" > logs/launch.txt; exit 1; }
[ -d smoke ] && mv smoke "smoke_alt_$(date +%H%M%S)"
$PY v1d_run.py --smoke --worker 0 --workers 1 > logs/smoke.log 2>&1
b=$(grep -c ',ok,' smoke/workers/w0.csv 2>/dev/null)
if [ "$b" != "3" ]; then echo "Rauchtest nicht ok (confirm $b) - kein Start" > logs/launch.txt; exit 1; fi
echo "Tests ok $(date '+%H:%M')" > logs/launch.txt
for w in 0 1; do screen -dmS v1dw$w caffeinate -i bash -c "{ $PY v1d_run.py --worker $w --workers 2; } >> logs/w$w.log 2>&1"; done
sleep 2; { echo "gestartet $(date)"; screen -ls; } >> logs/launch.txt
