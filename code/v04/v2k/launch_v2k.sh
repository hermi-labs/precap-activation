#!/bin/bash
# V2k starten (mit Claude, aus launch_v2h.sh): wartet auf das Ende von v2hw0, prueft Netzteil und andere Worker, Selbsttest KOMPLETT vor dem Freeze
# (Regel seit dem V2g-Abbruch), dann Freeze, Hash-Pruefung V2e/V2f, Auswertungstest, Rauchtest, 1 Worker in screen v2kw0 mit caffeinate und Waechter v2kwache.
# Messungen sind wiederaufnehmbar (je run_id und Kalibrierung).
cd "$(dirname "$0")" || exit 1
PY=${PY:-$HOME/precap_mlx_venv/bin/python}; PYS=${PYS:-$HOME/PyCharmMiscProject/.venv/bin/python}
mkdir -p meta logs
echo "wartet auf v2hw0 seit $(date '+%H:%M')" > logs/launch.txt
while screen -ls | grep -qE '\.v2hw0[[:space:]]'; do sleep 60; done
while pgrep -f "v2h_run.py --worke[r]" > /dev/null; do sleep 30; done
echo "v2hw0 beendet $(date '+%H:%M')" >> logs/launch.txt
if [ -d workers ]; then echo "Ergebnisordner existiert schon - Abbruch" >> logs/launch.txt; exit 1; fi
if screen -ls | grep -q -E "\.[A-Za-z0-9]+w[0-9]+[[:space:]]|\.[A-Za-z0-9]+ctl[[:space:]]" || pgrep -f "_run.py --worke[r]" > /dev/null; then echo "andere Worker laufen - Abbruch" >> logs/launch.txt; exit 1; fi
if ! pmset -g batt | grep -q "AC Power"; then echo "nicht am Netzteil - Abbruch" >> logs/launch.txt; exit 1; fi
$PY v2k_run.py --selfcheck > meta/selfcheck.json 2> logs/selfcheck_err.txt || { echo "Selbsttest fehlgeschlagen $(date '+%H:%M')" >> logs/launch.txt; exit 1; }
grep -q '"alle_ok": true' meta/selfcheck.json || { echo "Selbsttest-Wert nicht ok $(date '+%H:%M')" >> logs/launch.txt; exit 1; }
echo "Selbsttest ok $(date '+%H:%M')" >> logs/launch.txt
E=../v2e; F=../v2f; A=../v2a2; D=$HOME/PyCharmMiscProject/data/cifar10/cifar-10-batches-bin
shasum -a 256 v2k_run.py v2k_analyze.py test_v2k_analyse.py V2k_Praeregistrierung.md launch_v2k.sh $E/v2e_run.py $E/meta/frozen_sha256.txt $A/v2a2_run.py ../v2c/v2c_run.py ../v2d/v2d_run.py \
  $F/meta/frozen_sha256.txt $F/gewichte/*.npz $F/mess/*.json $D/data_batch_*.bin $D/test_batch.bin > meta/frozen_sha256.txt
date '+%Y-%m-%dT%H:%M:%S%z' > meta/frozen_at.txt
for s in $E $F $A ../v2c ../v2d; do (cd $s && shasum -a 256 -c meta/frozen_sha256.txt > /dev/null) || { echo "Hashes von $s weichen ab" >> logs/launch.txt; exit 1; }; done
$PYS -I test_v2k_analyse.py > meta/auswertungstest.txt 2>&1 || { echo "Auswertungstest fehlgeschlagen" >> logs/launch.txt; exit 1; }
echo "Rauchtest laeuft $(date '+%H:%M')" >> logs/launch.txt
[ -d smoke ] && mv smoke "smoke_alt_$(date +%H%M%S)"
$PY v2k_run.py --smoke > logs/smoke.log 2>&1
a=$(grep -c ',ok,' smoke/workers/w0.csv 2>/dev/null)
if [ "$a" != "4" ]; then echo "Rauchtest nicht ok ($a/4) - kein Start" >> logs/launch.txt; exit 1; fi
echo "Tests ok $(date '+%H:%M')" >> logs/launch.txt
screen -dmS v2kw0 caffeinate -i bash -c "{ $PY v2k_run.py --worker 0 --workers 1; } >> logs/w0.log 2>&1"
screen -dmS v2kwache bash -c "while screen -ls | grep -qE '\.v2kw0[[:space:]]'; do s=0; if ! pmset -g batt | grep -q 'AC Power'; then sleep 60; pmset -g batt | grep -q 'AC Power' || s=1; fi; if [ \$s = 1 ]; then pkill -f 'v2k_run.py --worker'; sleep 1; echo \"Waechter: beendet um \$(date +%H:%M)\" >> logs/launch.txt; break; fi; sleep 30; done"
sleep 2; { echo "gestartet $(date)"; screen -ls; } >> logs/launch.txt; cat logs/launch.txt
