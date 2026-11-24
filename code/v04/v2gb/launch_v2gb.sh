#!/bin/bash
# V2gb starten (Fassung b von V2g; Selbsttest-Fehler jetzt nach logs/selfcheck_err.txt) (mit Claude, aus launch_v2d.sh): Netzteil und andere Worker pruefen, einfrieren, Hash-Pruefung V2a2/V2c/V2d, Selbsttest (mit bitgleicher 20-Schritt-Probe),
# Auswertungstest, Rauchtest, dann 1 Worker in screen (v2gbw0) mit caffeinate und Netzteil-Waechter v2gbwache. Laeufe sind wiederaufnehmbar.
cd "$(dirname "$0")" || exit 1
PY=${PY:-$HOME/precap_mlx_venv/bin/python}; PYS=${PYS:-$HOME/PyCharmMiscProject/.venv/bin/python}
mkdir -p meta logs
if [ -d workers ] || [ -d tune ]; then echo "Ergebnisordner existiert schon - Abbruch" > logs/launch.txt; exit 1; fi
if screen -ls | grep -q -E "\.[A-Za-z0-9]+w[0-9]+[[:space:]]|\.[A-Za-z0-9]+ctl[[:space:]]" || pgrep -f "_run.py --worke[r]" > /dev/null; then echo "andere Worker laufen - Abbruch" > logs/launch.txt; exit 1; fi
if ! pmset -g batt | grep -q "AC Power"; then echo "nicht am Netzteil - Abbruch" > logs/launch.txt; exit 1; fi
D=$HOME/PyCharmMiscProject/data/cifar10/cifar-10-batches-bin; A=../v2a2
shasum -a 256 v2gb_run.py v2gb_analyze.py test_v2gb_analyse.py V2gb_Praeregistrierung.md launch_v2gb.sh $A/v2a2_run.py $A/kL2B_d50.json $A/v2a_tat_params.json $A/workers/w0.csv $A/tune/hp_sel.json $A/meta/frozen_sha256.txt ../v2c/workers/w0.csv ../v2c/meta/frozen_sha256.txt ../v2d/workers/w0.csv ../v2d/meta/frozen_sha256.txt $D/data_batch_*.bin $D/test_batch.bin > meta/frozen_sha256.txt
date '+%Y-%m-%dT%H:%M:%S%z' > meta/frozen_at.txt
for s in $A ../v2c ../v2d; do (cd $s && shasum -a 256 -c meta/frozen_sha256.txt > /dev/null) || { echo "Hashes von $s weichen ab" > logs/launch.txt; exit 1; }; done
$PY v2gb_run.py --selfcheck > meta/selfcheck.json 2> logs/selfcheck_err.txt || { echo "Selbsttest fehlgeschlagen" > logs/launch.txt; exit 1; }
for k in '"n_tune": 12' '"n_confirm": 12' '"steps": 3440' '"fixpunkt_ok": true' '"chi1_qstar": 1.0' '"xtanh_ist_tanh": true' '"bprecap_init_bitgleich": true' '"xtanh_gewichte_sigma_w": true' '"bias_ok": true' '"ausgabe_unveraendert": true' '"bprecap_20schritte_bitgleich": true' '"seeds_wie_v2a2": true'; do grep -q "$k" meta/selfcheck.json || { echo "Selbsttest-Wert nicht ok: $k" > logs/launch.txt; exit 1; }; done
$PYS -I test_v2gb_analyse.py > meta/auswertungstest.txt 2>&1 || { echo "Auswertungstest fehlgeschlagen" > logs/launch.txt; exit 1; }
echo "Rauchtest laeuft $(date '+%H:%M')" > logs/launch.txt
[ -d smoke ] && mv smoke "smoke_alt_$(date +%H%M%S)"
$PY v2gb_run.py --smoke > logs/smoke.log 2>&1
a=$(grep -c ',ok,' smoke/tune/workers/w0.csv 2>/dev/null); b=$(grep -c ',ok,' smoke/workers/w0.csv 2>/dev/null)
if [ "$a" != "2" ] || [ "$b" != "1" ] || [ ! -f smoke/tune/hp_sel.json ]; then echo "Rauchtest nicht ok ($a/$b) - kein Start" >> logs/launch.txt; exit 1; fi
echo "Tests ok $(date '+%H:%M')" >> logs/launch.txt
screen -dmS v2gbw0 caffeinate -i bash -c "{ $PY v2gb_run.py --worker 0 --workers 1; } >> logs/w0.log 2>&1"
screen -dmS v2gbwache bash -c "while screen -ls | grep -qE '\.v2gbw0[[:space:]]'; do s=0; if ! pmset -g batt | grep -q 'AC Power'; then sleep 60; pmset -g batt | grep -q 'AC Power' || s=1; fi; if [ \$s = 1 ]; then pkill -f 'v2gb_run.py --worker'; sleep 1; echo \"Waechter: beendet um \$(date +%H:%M)\" >> logs/launch.txt; break; fi; sleep 30; done"
sleep 2; { echo "gestartet $(date)"; screen -ls; } >> logs/launch.txt; cat logs/launch.txt
