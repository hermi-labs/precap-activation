#!/bin/bash
# V3 starten (mit Claude): Netzteil und andere Worker pruefen, v1_run.py gegen den V1-Freeze pruefen, einfrieren, Selbsttest, Mechanik- und
# Auswertungstest, Rauchtest (16 Zeilen ok), dann 2 Worker in screen (v3w0, v3w1): Tuning, Wahl von LR_Q, Bestaetigung.
cd "$(dirname "$0")"
PY=${PY:-$HOME/precap_mlx_venv/bin/python}; PYS=${PYS:-$HOME/PyCharmMiscProject/.venv/bin/python}
mkdir -p meta logs
if [ -d workers ] || [ -d tune ]; then echo "Ergebnisordner existiert schon - Abbruch" > logs/launch.txt; exit 1; fi
if screen -ls | grep -q -E "\.[a-z0-9]+w[0-9]+[[:space:]]"; then echo "andere Worker laufen - Abbruch" > logs/launch.txt; exit 1; fi
if ! pmset -g batt | grep -q "AC Power"; then echo "nicht am Netzteil - Abbruch" > logs/launch.txt; exit 1; fi
want=$(grep ' v1_run.py$' ../v1/meta/frozen_sha256.txt | cut -d' ' -f1); have=$(shasum -a 256 ../v1/v1_run.py | cut -d' ' -f1)
if [ -z "$want" ] || [ "$want" != "$have" ]; then echo "v1_run.py weicht vom Freeze ab - Abbruch" > logs/launch.txt; exit 1; fi
shasum -a 256 v3_qat.py v3_analyze.py test_v3_mechanik.py test_v3_analyse.py V3_Praeregistrierung.md ../v1/v1_run.py > meta/frozen_sha256.txt
date '+%Y-%m-%dT%H:%M:%S%z' > meta/frozen_at.txt
$PY v3_qat.py --selfcheck > meta/selfcheck.json 2>/dev/null || { echo "Selbsttest fehlgeschlagen" > logs/launch.txt; exit 1; }
grep -q '"ok": true' meta/selfcheck.json || { echo "Selbsttest-Werte nicht ok" > logs/launch.txt; exit 1; }
$PY test_v3_mechanik.py > meta/mechanik.txt 2>&1 || { echo "Mechanik-Test fehlgeschlagen" > logs/launch.txt; exit 1; }
$PYS -I test_v3_analyse.py > meta/auswertungstest.txt 2>&1 || { echo "Auswertungstest fehlgeschlagen" > logs/launch.txt; exit 1; }
[ -d smoke ] && mv smoke "smoke_alt_$(date +%H%M%S)"
$PY v3_qat.py --smoke > logs/smoke.log 2>&1
n=$(cat smoke/tune/workers/w0.csv smoke/workers/w0.csv 2>/dev/null | grep -c ',ok,')
if [ "$n" != "16" ]; then echo "Rauchtest nicht ok ($n Zeilen) - kein Start" > logs/launch.txt; exit 1; fi
echo "Tests ok $(date '+%H:%M')" > logs/launch.txt
for w in 0 1; do screen -dmS v3w$w caffeinate -i bash -c "{ $PY v3_qat.py --stage both --worker $w --workers 2; } >> logs/w$w.log 2>&1"; done
sleep 2; { echo "gestartet $(date)"; screen -ls; } >> logs/launch.txt
