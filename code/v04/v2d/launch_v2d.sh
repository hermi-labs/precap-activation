#!/bin/bash
# V2d starten (mit Claude, aus launch_v2a2.sh): Netzteil und andere Worker pruefen, einfrieren, Selbsttest, volle Kontrolle gegen V2a2,
# Auswertungstest, Rauchtest, dann 1 Worker in screen (v2dw0) mit caffeinate und Netzteil-Waechter v2dwache. Laeufe sind wiederaufnehmbar.
cd "$(dirname "$0")" || exit 1
PY=${PY:-$HOME/precap_mlx_venv/bin/python}; PYS=${PYS:-$HOME/PyCharmMiscProject/.venv/bin/python}
mkdir -p meta logs
if [ -d workers ] || [ -d tune ]; then echo "Ergebnisordner existiert schon - Abbruch" > logs/launch.txt; exit 1; fi
if screen -ls | grep -q -E "\.[A-Za-z0-9]+w[0-9]+[[:space:]]|\.[A-Za-z0-9]+ctl[[:space:]]|\.v2dpilot[[:space:]]" || pgrep -f "_run.py --worke[r]" > /dev/null; then echo "andere Worker laufen - Abbruch" > logs/launch.txt; exit 1; fi
if ! pmset -g batt | grep -q "AC Power"; then echo "nicht am Netzteil - Abbruch" > logs/launch.txt; exit 1; fi
D=$HOME/PyCharmMiscProject/data/cifar10/cifar-10-batches-bin; A=../v2a2; E=../v2d_entwurf
shasum -a 256 v2d_run.py v2d_analyze.py test_v2d_analyse.py V2d_Praeregistrierung.md launch_v2d.sh $A/v2a2_run.py $A/kL2B_d50.json $A/v2a_tat_params.json $A/workers/w0.csv $A/tune/hp_sel.json $A/meta/frozen_sha256.txt ../v2c/workers/w0.csv ../v2c/meta/frozen_sha256.txt $E/v2d_pilot.py $E/v2d_pilot_ctanh_1720.json $D/data_batch_*.bin $D/test_batch.bin > meta/frozen_sha256.txt
date '+%Y-%m-%dT%H:%M:%S%z' > meta/frozen_at.txt
(cd $A && shasum -a 256 -c meta/frozen_sha256.txt > /dev/null) && (cd ../v2c && shasum -a 256 -c meta/frozen_sha256.txt > /dev/null) || { echo "V2a2- oder V2c-Hashes weichen ab" > logs/launch.txt; exit 1; }
$PY v2d_run.py --selfcheck > meta/selfcheck.json 2>/dev/null || { echo "Selbsttest fehlgeschlagen" > logs/launch.txt; exit 1; }
for k in '"n_tune": 12' '"n_confirm": 12' '"steps": 3440' '"ctanh_wie_numpy": true' '"Ef2_ctanh": 1.0' '"chi1_ctanh": 1.04' '"bprecap_unveraendert": true' '"seeds_wie_v2a2": true' '"Ef_ctanh": 0.0'; do grep -q "$k" meta/selfcheck.json || { echo "Selbsttest-Wert nicht ok: $k" > logs/launch.txt; exit 1; }; done
$PYS -I test_v2d_analyse.py > meta/auswertungstest.txt 2>&1 || { echo "Auswertungstest fehlgeschlagen" > logs/launch.txt; exit 1; }
echo "Kontrolle laeuft $(date '+%H:%M')" > logs/launch.txt
$PY v2d_run.py --control > meta/kontrolle.txt 2>&1 || { echo "Volle Kontrolle gegen V2a2 fehlgeschlagen" >> logs/launch.txt; exit 1; }
[ -d smoke ] && mv smoke "smoke_alt_$(date +%H%M%S)"
$PY v2d_run.py --smoke > logs/smoke.log 2>&1
a=$(grep -c ',ok,' smoke/tune/workers/w0.csv 2>/dev/null); b=$(grep -c ',ok,' smoke/workers/w0.csv 2>/dev/null)
if [ "$a" != "2" ] || [ "$b" != "1" ] || [ ! -f smoke/tune/hp_sel.json ]; then echo "Rauchtest nicht ok ($a/$b) - kein Start" >> logs/launch.txt; exit 1; fi
echo "Tests ok $(date '+%H:%M')" >> logs/launch.txt
screen -dmS v2dw0 caffeinate -i bash -c "{ $PY v2d_run.py --worker 0 --workers 1; } >> logs/w0.log 2>&1"
screen -dmS v2dwache bash -c "while screen -ls | grep -qE '\.v2dw0[[:space:]]'; do s=0; if ! pmset -g batt | grep -q 'AC Power'; then sleep 60; pmset -g batt | grep -q 'AC Power' || s=1; fi; if [ \$s = 1 ]; then pkill -f 'v2d_run.py --worker'; sleep 1; echo \"Waechter: beendet um \$(date +%H:%M)\" >> logs/launch.txt; break; fi; sleep 30; done"
sleep 2; { echo "gestartet $(date)"; screen -ls; } >> logs/launch.txt; cat logs/launch.txt
