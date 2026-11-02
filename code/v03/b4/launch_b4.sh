#!/bin/bash
# B4 starten: Netzteil pruefen, einfrieren, Selbsttest, Mechanik- und Auswertungstest, Rauchtest MNIST d100 (5 Laeufe, alle ok),
# dann 2 Worker in screen (b4w0, b4w1). GPU-Lauf etwa 2,5 h, Freigabe von Aaron fuer die Nacht 30./31.10..
cd "$(dirname "$0")"
PY=${PY:-$HOME/precap_mlx_venv/bin/python}
PYS=${PYS:-$HOME/PyCharmMiscProject/.venv/bin/python}   # mit scipy fuer den Auswertungstest
mkdir -p meta logs
if [ -d workers ]; then echo "Ergebnisordner existiert schon - Abbruch" > logs/launch.txt; exit 1; fi
if screen -ls | grep -q -E "gpt2w|gpt3w|b1w|b2w|b3w|b4w"; then echo "andere Worker laufen - Abbruch" > logs/launch.txt; exit 1; fi
if ! pmset -g batt | grep -q "AC Power"; then echo "nicht am Netzteil - Abbruch (Akku pausiert Laeufe bei Deckel zu)" > logs/launch.txt; exit 1; fi
shasum -a 256 b4_run.py b4_analyze.py test_b4_mechanik.py test_b4_analyse.py B4_Praeregistrierung.md lr_from_a.csv lr_from_p3a3.csv p7_hard_bound.json p11_gleiches_M.json src/precap/*.py > meta/frozen_sha256.txt
date '+%Y-%m-%dT%H:%M:%S%z' > meta/frozen_at.txt
$PY b4_run.py --selfcheck > meta/selfcheck.json 2>/dev/null || { echo "Selbsttest fehlgeschlagen" > logs/launch.txt; exit 1; }
grep -q '"brake_monotone": true' meta/selfcheck.json && grep -q '"brake_core_identity": true' meta/selfcheck.json && grep -q '"n_confirm": 194' meta/selfcheck.json || { echo "Selbsttest-Werte nicht ok" > logs/launch.txt; exit 1; }
$PY test_b4_mechanik.py > meta/mechanik.txt 2>&1 || { echo "Mechanik-Test fehlgeschlagen" > logs/launch.txt; exit 1; }
$PYS -I test_b4_analyse.py > meta/auswertungstest.txt 2>&1 || { echo "Auswertungstest fehlgeschlagen" > logs/launch.txt; exit 1; }
rm -rf smoke; $PY b4_run.py --smoke --worker 0 --workers 1 > logs/smoke.log 2>&1
b=$(grep -c ',ok,' smoke/workers/w0.csv 2>/dev/null)
if [ "$b" != "5" ]; then echo "Rauchtest nicht ok ($b) - kein Start" > logs/launch.txt; exit 1; fi
echo "Tests ok $(date '+%H:%M')" > logs/launch.txt
for w in 0 1; do screen -dmS b4w$w caffeinate -i bash -c "{ $PY b4_run.py --worker $w --workers 2; } >> logs/w$w.log 2>&1"; done
sleep 2; { echo "gestartet $(date)"; screen -ls; } >> logs/launch.txt
