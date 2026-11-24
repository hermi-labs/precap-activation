#!/bin/bash
# V2m starten (mit Claude, aus launch_v2k.sh): Netzteil und andere Worker pruefen, Selbsttest KOMPLETT vor dem Freeze, Freeze, Hash-Pruefung
# V2a2/V2e/V2f/V2k, Auswertungstest, Rauchtest (1 Lauf, 20 Schritte, 50 Bilder), dann 1 Worker in screen v2mw0 mit caffeinate und Waechter v2mwache.
cd "$(dirname "$0")" || exit 1
PY=${PY:-$HOME/precap_mlx_venv/bin/python}; PYS=${PYS:-$HOME/PyCharmMiscProject/.venv/bin/python}
mkdir -p meta logs
echo "Start launch $(date '+%H:%M')" > logs/launch.txt
if [ -d workers ] || [ -d gewichte ] || [ -d mess ]; then echo "Ergebnisordner existiert schon - Abbruch" >> logs/launch.txt; exit 1; fi
if screen -ls | grep -q -E "\.[A-Za-z0-9]+w[0-9]+[[:space:]]|\.[A-Za-z0-9]+ctl[[:space:]]" || pgrep -f "_run.py --worke[r]" > /dev/null || pgrep -f "_run.py --smok[e]" > /dev/null; then echo "andere Worker laufen - Abbruch" >> logs/launch.txt; exit 1; fi
if ! pmset -g batt | grep -q "AC Power"; then echo "nicht am Netzteil - Abbruch" >> logs/launch.txt; exit 1; fi
$PY v2m_run.py --selfcheck > meta/selfcheck.json 2> logs/selfcheck_err.txt || { echo "Selbsttest fehlgeschlagen $(date '+%H:%M')" >> logs/launch.txt; exit 1; }
grep -q '"alle_ok": true' meta/selfcheck.json || { echo "Selbsttest-Wert nicht ok $(date '+%H:%M')" >> logs/launch.txt; exit 1; }
echo "Selbsttest ok $(date '+%H:%M')" >> logs/launch.txt
E=../v2e; F=../v2f; K=../v2k; A=../v2a2; D=$HOME/PyCharmMiscProject/data/cifar10/cifar-10-batches-bin
shasum -a 256 v2m_run.py v2m_analyze.py test_v2m_analyse.py v2m_konstanten.py kL2B_d50_rho04.json V2m_Praeregistrierung.md launch_v2m.sh ../../math/p4_bounded_depth.py \
  $E/v2e_run.py $K/v2k_run.py $K/workers/w0.csv $K/meta/frozen_sha256.txt $A/v2a2_run.py $A/kL2B_d50.json $A/workers/w0.csv $A/meta/frozen_sha256.txt ../v2c/v2c_run.py ../v2d/v2d_run.py \
  $E/meta/frozen_sha256.txt $F/meta/frozen_sha256.txt $D/data_batch_*.bin $D/test_batch.bin > meta/frozen_sha256.txt
date '+%Y-%m-%dT%H:%M:%S%z' > meta/frozen_at.txt
for s in $E $F $K $A ../v2c ../v2d; do (cd $s && shasum -a 256 -c meta/frozen_sha256.txt > /dev/null) || { echo "Hashes von $s weichen ab" >> logs/launch.txt; exit 1; }; done
echo "eingefroren, Hashes der Vorgaenger ok $(date '+%H:%M')" >> logs/launch.txt
$PYS -I test_v2m_analyse.py > meta/auswertungstest.txt 2>&1 || { echo "Auswertungstest fehlgeschlagen" >> logs/launch.txt; exit 1; }
grep -q "AUSWERTUNG OK" meta/auswertungstest.txt || { echo "Auswertungstest nicht ok" >> logs/launch.txt; exit 1; }
echo "Rauchtest laeuft $(date '+%H:%M')" >> logs/launch.txt
[ -d smoke ] && mv smoke "smoke_alt_$(date +%H%M%S)"
$PY v2m_run.py --smoke > logs/smoke.log 2>&1
a=$(grep -c ',ok,' smoke/workers/w0.csv 2>/dev/null); b=$(ls smoke/gewichte/*.npz 2>/dev/null | wc -l | tr -d ' '); c=$(ls smoke/mess/*.json 2>/dev/null | wc -l | tr -d ' ')
if [ "$a" != "1" ] || [ "$b" != "1" ] || [ "$c" != "1" ]; then echo "Rauchtest nicht ok ($a/$b/$c) - kein Start" >> logs/launch.txt; exit 1; fi
echo "Tests ok $(date '+%H:%M')" >> logs/launch.txt
screen -dmS v2mw0 caffeinate -i bash -c "{ $PY v2m_run.py --worker 0 --workers 1; } >> logs/w0.log 2>&1"
screen -dmS v2mwache bash -c "while screen -ls | grep -qE '\.v2mw0[[:space:]]'; do s=0; if ! pmset -g batt | grep -q 'AC Power'; then sleep 60; pmset -g batt | grep -q 'AC Power' || s=1; fi; if [ \$s = 1 ]; then pkill -f 'v2m_run.py --worker'; sleep 1; echo \"Waechter: beendet um \$(date +%H:%M)\" >> logs/launch.txt; break; fi; sleep 30; done"
sleep 2; { echo "gestartet $(date)"; screen -ls; } >> logs/launch.txt; cat logs/launch.txt
