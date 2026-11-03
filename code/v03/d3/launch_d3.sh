#!/bin/bash
# D3 starten (Diagnose, beschreibend, mit Claude): Netzteil pruefen, keine anderen Worker, b10_run.py und b15_run.py gegen ihren Freeze pruefen,
# einfrieren, Selbsttest, Rauchtest der Messung und Auswertungstest, dann 2 Worker in screen (d3w0, d3w1). 6 Laeufe, etwa 25-35 min.
cd "$(dirname "$0")"
PY=${PY:-$HOME/precap_mlx_venv/bin/python}
mkdir -p meta logs
if [ -d results ]; then echo "Ergebnisordner existiert schon - Abbruch" > logs/launch.txt; exit 1; fi
if screen -ls | grep -q -E "\.[a-z0-9]+w[0-9]+[[:space:]]"; then echo "andere Worker laufen - Abbruch" > logs/launch.txt; exit 1; fi
if ! pmset -g batt | grep -q "AC Power"; then echo "nicht am Netzteil - Abbruch" > logs/launch.txt; exit 1; fi
for s in b10 b15; do
  want=$(grep " ${s}_run.py$" ../$s/meta/frozen_sha256.txt | cut -d' ' -f1); have=$(shasum -a 256 ../$s/${s}_run.py | cut -d' ' -f1)
  if [ -z "$want" ] || [ "$want" != "$have" ]; then echo "${s}_run.py weicht vom Freeze ab - Abbruch" > logs/launch.txt; exit 1; fi
done
shasum -a 256 d3_diag.py d3_auswerten.py D3_Praeregistrierung.md test/rauch_messung.py test/test_auswertung.py ../b10/b10_run.py ../b15/b15_run.py > meta/frozen_sha256.txt
date '+%Y-%m-%dT%H:%M:%S%z' > meta/frozen_at.txt
$PY d3_diag.py --selfcheck > meta/selfcheck.json 2>/dev/null || { echo "Selbsttest fehlgeschlagen" > logs/launch.txt; exit 1; }
grep -q '"ok": true' meta/selfcheck.json || { echo "Selbsttest-Werte nicht ok" > logs/launch.txt; exit 1; }
$PY test/rauch_messung.py > meta/rauchtest.txt 2>&1 || { echo "Rauchtest der Messung fehlgeschlagen" > logs/launch.txt; exit 1; }
python3 -I test/test_auswertung.py > meta/auswertungstest.txt 2>&1 || { echo "Auswertungstest fehlgeschlagen" > logs/launch.txt; exit 1; }
echo "Tests ok $(date '+%H:%M')" > logs/launch.txt
for w in 0 1; do screen -dmS d3w$w caffeinate -i bash -c "{ $PY d3_diag.py --worker $w --workers 2; } >> logs/w$w.log 2>&1"; done
sleep 2; { echo "gestartet $(date)"; screen -ls; } >> logs/launch.txt
