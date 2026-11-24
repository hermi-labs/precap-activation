#!/bin/bash
# V2f starten (mit Claude, aus launch_v2e.sh): Netzteil und andere Worker pruefen, einfrieren, Hash-Pruefung V2e/V2a2/V2c/V2d, Selbsttest,
# Auswertungstest, Rauchtest (2 Laeufe a 20 Schritte, beide Kalibrierungen), dann 1 Worker in screen (v2fw0) mit caffeinate und Netzteil-Waechter v2fwache.
# Jeder Lauf ist zugleich die volle Kontrolle gegen seine historische Zeile (kontrolle_n_diff). Laeufe sind wiederaufnehmbar.
cd "$(dirname "$0")" || exit 1
PY=${PY:-$HOME/precap_mlx_venv/bin/python}; PYS=${PYS:-$HOME/PyCharmMiscProject/.venv/bin/python}
mkdir -p meta logs
if [ -d workers ] || [ -d mess ] || [ -d gewichte ]; then echo "Ergebnisordner existiert schon - Abbruch" > logs/launch.txt; exit 1; fi
if screen -ls | grep -q -E "\.[A-Za-z0-9]+w[0-9]+[[:space:]]|\.[A-Za-z0-9]+ctl[[:space:]]" || pgrep -f "_run.py --worke[r]" > /dev/null || pgrep -f "_run.py --smok[e]" > /dev/null; then echo "andere Worker laufen - Abbruch" > logs/launch.txt; exit 1; fi
if ! pmset -g batt | grep -q "AC Power"; then echo "nicht am Netzteil - Abbruch" > logs/launch.txt; exit 1; fi
D=$HOME/PyCharmMiscProject/data/cifar10/cifar-10-batches-bin; A=../v2a2
shasum -a 256 v2f_run.py v2f_analyze.py test_v2f_analyse.py V2f_Praeregistrierung.md launch_v2f.sh ../v2e/v2e_run.py ../v2e/meta/frozen_sha256.txt ../v2e_kal/mess/*.json $A/v2a2_run.py $A/kL2B_d50.json $A/v2a_tat_params.json $A/workers/w0.csv $A/meta/frozen_sha256.txt ../v2c/v2c_run.py ../v2c/meta/frozen_sha256.txt ../v2d/v2d_run.py ../v2d/workers/w0.csv ../v2d/meta/frozen_sha256.txt $D/data_batch_*.bin $D/test_batch.bin > meta/frozen_sha256.txt
date '+%Y-%m-%dT%H:%M:%S%z' > meta/frozen_at.txt
for s in ../v2e $A ../v2c ../v2d; do (cd $s && shasum -a 256 -c meta/frozen_sha256.txt > /dev/null) || { echo "Hashes von $s weichen ab" > logs/launch.txt; exit 1; }; done
$PY v2f_run.py --selfcheck > meta/selfcheck.json 2>/dev/null || { echo "Selbsttest fehlgeschlagen" > logs/launch.txt; exit 1; }
for k in '"n_runs": 18' '"soll_zeilen": 36' '"hist_vollstaendig": true' '"seeds_neu_gegen_v2e": true' '"run_ids_wie_hist": true' '"act_ctanh_wie_v2d": true' '"Ef2_bprecap": 1.0' '"Ef2_ctanh": 1.0' '"mess_mse4_endlich": true' '"mess_max_endlich": true' '"mess_mse4_r_positiv": true' '"mess_max_r_positiv": true' '"mess_reproduzierbar": true'; do grep -q "$k" meta/selfcheck.json || { echo "Selbsttest-Wert nicht ok: $k" > logs/launch.txt; exit 1; }; done
$PYS -I test_v2f_analyse.py > meta/auswertungstest.txt 2>&1 || { echo "Auswertungstest fehlgeschlagen" > logs/launch.txt; exit 1; }
[ -d smoke ] && mv smoke "smoke_alt_$(date +%H%M%S)"
echo "Rauchtest laeuft $(date '+%H:%M')" > logs/launch.txt
$PY v2f_run.py --smoke > logs/smoke.log 2>&1
a=$(grep -c ',ok,' smoke/workers/w0.csv 2>/dev/null); b=$(ls smoke/mess/*.json 2>/dev/null | wc -l | tr -d ' '); c=$(ls smoke/gewichte/*.npz 2>/dev/null | wc -l | tr -d ' ')
if [ "$a" != "4" ] || [ "$b" != "4" ] || [ "$c" != "2" ]; then echo "Rauchtest nicht ok ($a/$b/$c) - kein Start" >> logs/launch.txt; exit 1; fi
echo "Tests ok $(date '+%H:%M')" >> logs/launch.txt
screen -dmS v2fw0 caffeinate -i bash -c "{ $PY v2f_run.py --worker 0 --workers 1; } >> logs/w0.log 2>&1"
screen -dmS v2fwache bash -c "while screen -ls | grep -qE '\.v2fw0[[:space:]]'; do s=0; if ! pmset -g batt | grep -q 'AC Power'; then sleep 60; pmset -g batt | grep -q 'AC Power' || s=1; fi; if [ \$s = 1 ]; then pkill -f 'v2f_run.py --worker'; sleep 1; echo \"Waechter: beendet um \$(date +%H:%M)\" >> logs/launch.txt; break; fi; sleep 30; done"
sleep 2; { echo "gestartet $(date)"; screen -ls; } >> logs/launch.txt; cat logs/launch.txt
