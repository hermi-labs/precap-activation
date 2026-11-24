#!/bin/bash
# V2a2 starten (mit Claude, aus launch_v2a1.sh und launch_ki1.sh): Netzteil und andere Worker pruefen, einfrieren, Selbsttest, Mechanik-Test,
# Auswertungstest, Rauchtest, dann 1 Worker in screen (v2a2w0) mit caffeinate und Netzteil-Waechter v2a2wache. Laeufe sind wiederaufnehmbar.
cd "$(dirname "$0")" || exit 1
PY=${PY:-$HOME/precap_mlx_venv/bin/python}; PYS=${PYS:-$HOME/PyCharmMiscProject/.venv/bin/python}
mkdir -p meta logs
if [ -d workers ] || [ -d tune ]; then echo "Ergebnisordner existiert schon - Abbruch" > logs/launch.txt; exit 1; fi
if screen -ls | grep -q -E "\.[A-Za-z0-9]+w[0-9]+[[:space:]]|\.[A-Za-z0-9]+ctl[[:space:]]" || pgrep -f "_run.py --worke[r]" > /dev/null; then echo "andere Worker laufen - Abbruch" > logs/launch.txt; exit 1; fi
if ! pmset -g batt | grep -q "AC Power"; then echo "nicht am Netzteil - Abbruch" > logs/launch.txt; exit 1; fi
D=$HOME/PyCharmMiscProject/data/cifar10/cifar-10-batches-bin; E=../v2a_entwurf
shasum -a 256 v2a2_run.py v2a2_analyze.py test_v2a2_mechanik.py test_v2a2_analyse.py kL2B_d50.json v2a_tat_params.json V2a2_Praeregistrierung.md launch_v2a2.sh meta_vorab/mechanik_vorab.txt $E/v2a_pilot4.py $E/v2a_pilot.py $E/v2a_probe.py $E/v2a_konstanten.json $E/v2a_tat_params.json $E/v2a_pilot4_D_M1Pro.json $D/data_batch_*.bin $D/test_batch.bin > meta/frozen_sha256.txt
date '+%Y-%m-%dT%H:%M:%S%z' > meta/frozen_at.txt
cmp -s v2a_tat_params.json $E/v2a_tat_params.json && $PYS -I -c "import json,sys; sys.exit(0 if json.load(open('kL2B_d50.json')) == json.load(open('$E/v2a_konstanten.json'))['kL2B_d50'] else 1)" || { echo "Konstanten weichen vom Entwurf ab" > logs/launch.txt; exit 1; }
$PY v2a2_run.py --selfcheck > meta/selfcheck.json 2>/dev/null || { echo "Selbsttest fehlgeschlagen" > logs/launch.txt; exit 1; }
for k in '"n_tune": 36' '"n_confirm": 36' '"steps": 3440' '"Ef2_ok": true' '"delta_orth_ok": true' '"seeds_new": true' '"order_rotates": true'; do grep -q "$k" meta/selfcheck.json || { echo "Selbsttest-Wert nicht ok: $k" > logs/launch.txt; exit 1; }; done
grep -q "MECHANIK OK" meta_vorab/mechanik_vorab.txt || { echo "Mechanik vorab (volle Kontrolle) nicht ok" > logs/launch.txt; exit 1; }
$PY test_v2a2_mechanik.py > meta/mechanik.txt 2>&1 || { echo "Mechanik-Test fehlgeschlagen" > logs/launch.txt; exit 1; }
$PYS -I test_v2a2_analyse.py > meta/auswertungstest.txt 2>&1 || { echo "Auswertungstest fehlgeschlagen" > logs/launch.txt; exit 1; }
[ -d smoke ] && mv smoke "smoke_alt_$(date +%H%M%S)"
$PY v2a2_run.py --smoke > logs/smoke.log 2>&1
a=$(grep -c ',ok,' smoke/tune/workers/w0.csv 2>/dev/null); b=$(grep -c ',ok,' smoke/workers/w0.csv 2>/dev/null)
if [ "$a" != "6" ] || [ "$b" != "3" ] || [ ! -f smoke/tune/hp_sel.json ]; then echo "Rauchtest nicht ok ($a/$b) - kein Start" > logs/launch.txt; exit 1; fi
echo "Tests ok $(date '+%H:%M')" > logs/launch.txt
screen -dmS v2a2w0 caffeinate -i bash -c "{ $PY v2a2_run.py --worker 0 --workers 1; } >> logs/w0.log 2>&1"
screen -dmS v2a2wache bash -c "while screen -ls | grep -qE '\.v2a2w0[[:space:]]'; do s=0; if ! pmset -g batt | grep -q 'AC Power'; then sleep 60; pmset -g batt | grep -q 'AC Power' || s=1; fi; if [ \$s = 1 ]; then pkill -f 'v2a2_run.py --worker'; sleep 1; echo \"Waechter: beendet um \$(date +%H:%M)\" >> logs/launch.txt; break; fi; sleep 30; done"
sleep 2; { echo "gestartet $(date)"; screen -ls; } >> logs/launch.txt; cat logs/launch.txt
