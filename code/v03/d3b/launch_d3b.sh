#!/bin/bash
# D3b starten (beschreibend, mit Claude): Netzteil, keine anderen Worker, d3_diag.py und v1c_run.py gegen ihren Freeze pruefen,
# einfrieren, Selbsttest, Rauchtest und Auswertungstest, dann 2 Worker in screen (d3bw0, d3bw1). 4 Laeufe, etwa 15-25 min.
cd "$(dirname "$0")"
PY=${PY:-$HOME/precap_mlx_venv/bin/python}
mkdir -p meta logs
if [ -d results ]; then echo "Ergebnisordner existiert schon - Abbruch" > logs/launch.txt; exit 1; fi
if screen -ls | grep -q -E "\.[a-z0-9]+w[0-9]+[[:space:]]"; then echo "andere Worker laufen - Abbruch" > logs/launch.txt; exit 1; fi
if ! pmset -g batt | grep -q "AC Power"; then echo "nicht am Netzteil - Abbruch" > logs/launch.txt; exit 1; fi
for s in d3/d3_diag.py v1c/v1c_run.py; do
  d=${s%%/*}; f=${s##*/}
  want=$(grep " ${f}$" ../$d/meta/frozen_sha256.txt | cut -d' ' -f1); have=$(shasum -a 256 ../$s | cut -d' ' -f1)
  if [ -z "$want" ] || [ "$want" != "$have" ]; then echo "$s weicht vom Freeze ab - Abbruch" > logs/launch.txt; exit 1; fi
done
shasum -a 256 d3b_diag.py d3b_auswerten.py D3b_Praeregistrierung.md test/rauch_messung.py test/test_auswertung.py ../d3/d3_diag.py ../v1c/v1c_run.py \
  ../d3/results/d3_bp200_s42192.json ../d3/results/d3_bp200_s42193.json ../d3/results/d3_se200_s42192.json ../d3/results/d3_se200_s42193.json > meta/frozen_sha256.txt
date '+%Y-%m-%dT%H:%M:%S%z' > meta/frozen_at.txt
$PY d3b_diag.py --selfcheck > meta/selfcheck.json 2>/dev/null || { echo "Selbsttest fehlgeschlagen" > logs/launch.txt; exit 1; }
grep -q '"ok": true' meta/selfcheck.json || { echo "Selbsttest-Werte nicht ok" > logs/launch.txt; exit 1; }
$PY test/rauch_messung.py > meta/rauchtest.txt 2>&1 || { echo "Rauchtest fehlgeschlagen" > logs/launch.txt; exit 1; }
python3 -I test/test_auswertung.py > meta/auswertungstest.txt 2>&1 || { echo "Auswertungstest fehlgeschlagen" > logs/launch.txt; exit 1; }
echo "Tests ok $(date '+%H:%M')" > logs/launch.txt
for w in 0 1; do screen -dmS d3bw$w caffeinate -i bash -c "{ $PY d3b_diag.py --worker $w --workers 2; } >> logs/w$w.log 2>&1"; done
sleep 2; { echo "gestartet $(date)"; screen -ls; } >> logs/launch.txt
