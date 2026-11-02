#!/bin/bash
# B9 starten: Netzteil pruefen, einfrieren, Selbsttest, Mechanik- und Auswertungstest, Rauchtest (5 Laeufe mit 40 Schritten, alle ok, Zwischenspalten da),
# dann 2 Worker in screen (b9w0, b9w1). GPU-Lauf etwa 1,5 h, Freigabe von Aaron (31.10., allein arbeiten).
cd "$(dirname "$0")"
PY=${PY:-$HOME/precap_mlx_venv/bin/python}
PYS=${PYS:-$HOME/PyCharmMiscProject/.venv/bin/python}   # mit scipy fuer den Auswertungstest
D=$HOME/PyCharmMiscProject/data/cifar10/cifar-10-batches-bin
mkdir -p meta logs
if [ -d workers ] || [ -d tune ]; then echo "Ergebnisordner existiert schon - Abbruch" > logs/launch.txt; exit 1; fi
if screen -ls | grep -q -E "gpt2w|gpt3w|b1w|b2w|b3w|b4w|b5w|b6w|b7w|b8w|b9w"; then echo "andere Worker laufen - Abbruch" > logs/launch.txt; exit 1; fi
if ! pmset -g batt | grep -q "AC Power"; then echo "nicht am Netzteil - Abbruch (Akku pausiert Laeufe bei Deckel zu)" > logs/launch.txt; exit 1; fi
shasum -a 256 b9_run.py b9_analyze.py test_b9_mechanik.py test_b9_analyse.py B9_Praeregistrierung.md lr_from_a.csv lr_from_p3a3.csv lr_from_b3.csv lr_from_b6.csv src/precap/*.py $D/*.bin > meta/frozen_sha256.txt
date '+%Y-%m-%dT%H:%M:%S%z' > meta/frozen_at.txt
$PY b9_run.py --selfcheck > meta/selfcheck.json 2>/dev/null || { echo "Selbsttest fehlgeschlagen" > logs/launch.txt; exit 1; }
grep -q '"n_confirm": 38' meta/selfcheck.json && grep -q '"n_tune": 0' meta/selfcheck.json || { echo "Selbsttest-Werte nicht ok" > logs/launch.txt; exit 1; }
$PY test_b9_mechanik.py > meta/mechanik.txt 2>&1 || { echo "Mechanik-Test fehlgeschlagen" > logs/launch.txt; exit 1; }
$PYS -I test_b9_analyse.py > meta/auswertungstest.txt 2>&1 || { echo "Auswertungstest fehlgeschlagen" > logs/launch.txt; exit 1; }
rm -rf smoke; $PY b9_run.py --smoke --worker 0 --workers 1 > logs/smoke.log 2>&1
b=$(grep -c ',ok,' smoke/workers/w0.csv 2>/dev/null); k=$($PY -c "import csv; r=list(csv.DictReader(open('smoke/workers/w0.csv'))); print(sum(x.get('c20_test_err_fp32', '') != '' for x in r))" 2>/dev/null)
if [ "$b" != "5" ] || [ "$k" != "5" ]; then echo "Rauchtest nicht ok (confirm $b, Zwischenwerte $k) - kein Start" > logs/launch.txt; exit 1; fi
echo "Tests ok $(date '+%H:%M')" > logs/launch.txt
for w in 0 1; do screen -dmS b9w$w caffeinate -i bash -c "{ $PY b9_run.py --worker $w --workers 2; } >> logs/w$w.log 2>&1"; done
sleep 2; { echo "gestartet $(date)"; screen -ls; } >> logs/launch.txt
