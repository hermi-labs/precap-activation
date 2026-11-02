#!/bin/bash
# D1 starten (Diagnose, beschreibend): Netzteil pruefen, keine anderen Worker, einfrieren, b10_run.py gegen den B10-Freeze pruefen,
# Selbsttest und Rauchtest der Messung, dann 2 Worker in screen (d1w0, d1w1). 12 Laeufe, etwa 30-35 min. Start erst nach B14.
cd "$(dirname "$0")"
PY=${PY:-$HOME/precap_mlx_venv/bin/python}
mkdir -p meta logs
if [ -d results ]; then echo "Ergebnisordner existiert schon - Abbruch" > logs/launch.txt; exit 1; fi
if screen -ls | grep -q -E "gpt2w|gpt3w|b[0-9]+[a-z]?w|d1w"; then echo "andere Worker laufen - Abbruch" > logs/launch.txt; exit 1; fi
if ! pmset -g batt | grep -q "AC Power"; then echo "nicht am Netzteil - Abbruch" > logs/launch.txt; exit 1; fi
want=$(grep ' b10_run.py$' ../b10/meta/frozen_sha256.txt | cut -d' ' -f1); have=$(shasum -a 256 ../b10/b10_run.py | cut -d' ' -f1)
if [ -z "$want" ] || [ "$want" != "$have" ]; then echo "b10_run.py weicht vom B10-Freeze ab - Abbruch" > logs/launch.txt; exit 1; fi
want=$(grep ' b12_run.py$' ../b12/meta/frozen_sha256.txt | cut -d' ' -f1); have=$(shasum -a 256 ../b12/b12_run.py | cut -d' ' -f1)
if [ -z "$want" ] || [ "$want" != "$have" ]; then echo "b12_run.py weicht vom B12-Freeze ab - Abbruch" > logs/launch.txt; exit 1; fi
want=$(grep ' b13_run.py$' ../b13/meta/frozen_sha256.txt | cut -d' ' -f1); have=$(shasum -a 256 ../b13/b13_run.py | cut -d' ' -f1)
if [ -z "$want" ] || [ "$want" != "$have" ]; then echo "b13_run.py weicht vom B13-Freeze ab - Abbruch" > logs/launch.txt; exit 1; fi
shasum -a 256 d1_diag.py d1_auswerten.py D1_Praeregistrierung.md test/rauch_messung.py ../b10/b10_run.py ../b12/b12_run.py ../b13/b13_run.py > meta/frozen_sha256.txt
date '+%Y-%m-%dT%H:%M:%S%z' > meta/frozen_at.txt
$PY d1_diag.py --selfcheck > meta/selfcheck.json 2>/dev/null || { echo "Selbsttest fehlgeschlagen" > logs/launch.txt; exit 1; }
grep -q '"ok": true' meta/selfcheck.json || { echo "Selbsttest-Werte nicht ok" > logs/launch.txt; exit 1; }
$PY test/rauch_messung.py > meta/rauchtest.txt 2>&1 || { echo "Rauchtest der Messung fehlgeschlagen" > logs/launch.txt; exit 1; }
[ "$(grep -c 'noise_gain_median' meta/rauchtest.txt)" = "2" ] || { echo "Rauchtest unvollstaendig" > logs/launch.txt; exit 1; }
echo "Tests ok $(date '+%H:%M')" > logs/launch.txt
for w in 0 1; do screen -dmS d1w$w caffeinate -i bash -c "{ $PY d1_diag.py --worker $w --workers 2; } >> logs/w$w.log 2>&1"; done
sleep 2; { echo "gestartet $(date)"; screen -ls; } >> logs/launch.txt
