#!/bin/bash
# V2a-1 starten (mit Claude): Netzteil und andere Worker pruefen, einfrieren, Selbsttest, Mechanik- und Auswertungstest, Rauchtest,
# dann 2 Worker in screen (v2a1w0, v2a1w1) mit caffeinate. Logs in logs/.
cd "$(dirname "$0")"
PY=${PY:-$HOME/precap_mlx_venv/bin/python}; PYS=${PYS:-$HOME/PyCharmMiscProject/.venv/bin/python}
mkdir -p meta logs
if [ -d workers ]; then echo "Ergebnisordner existiert schon - Abbruch" > logs/launch.txt; exit 1; fi
if screen -ls | grep -q -E "\.[a-z0-9]+w[0-9]+[[:space:]]|ctl[[:space:]]"; then echo "andere Worker laufen - Abbruch" > logs/launch.txt; exit 1; fi
if ! pmset -g batt | grep -q "AC Power"; then echo "nicht am Netzteil - Abbruch" > logs/launch.txt; exit 1; fi
D=$HOME/PyCharmMiscProject/data/cifar10/cifar-10-batches-bin; E=../v2a_entwurf
shasum -a 256 v2a1_run.py v2a1_analyze.py test_v2a1_mechanik.py test_v2a1_analyse.py kL2B_d32.json V2a1_Praeregistrierung.md launch_v2a1.sh $E/v2a_pilot4.py $E/v2a_pilot.py $E/v2a_probe.py $E/v2a_konstanten.json $E/v2a_tat_params.json $E/v2a_pilot4_A_M1Pro.json $D/data_batch_*.bin $D/test_batch.bin > meta/frozen_sha256.txt
date '+%Y-%m-%dT%H:%M:%S%z' > meta/frozen_at.txt
$PY v2a1_run.py --selfcheck > meta/selfcheck.json 2>/dev/null || { echo "Selbsttest fehlgeschlagen" > logs/launch.txt; exit 1; }
for k in '"n_confirm": 24' '"steps": 1720' '"bprecap_Ef2_ok": true' '"delta_orth_ok": true' '"seeds_new": true' '"workers_mixed": true'; do grep -q "$k" meta/selfcheck.json || { echo "Selbsttest-Wert nicht ok: $k" > logs/launch.txt; exit 1; }; done
$PY test_v2a1_mechanik.py > meta/mechanik.txt 2>&1 || { echo "Mechanik-Test fehlgeschlagen" > logs/launch.txt; exit 1; }
$PYS -I test_v2a1_analyse.py > meta/auswertungstest.txt 2>&1 || { echo "Auswertungstest fehlgeschlagen" > logs/launch.txt; exit 1; }
[ -d smoke ] && mv smoke "smoke_alt_$(date +%H%M%S)"
$PY v2a1_run.py --smoke > logs/smoke.log 2>&1
b=$(grep -c ',ok,' smoke/workers/w0.csv 2>/dev/null)
if [ "$b" != "4" ]; then echo "Rauchtest nicht ok ($b) - kein Start" > logs/launch.txt; exit 1; fi
echo "Tests ok $(date '+%H:%M')" > logs/launch.txt
for w in 0 1; do screen -dmS v2a1w$w caffeinate -i bash -c "{ $PY v2a1_run.py --worker $w --workers 2; } >> logs/w$w.log 2>&1"; done
sleep 2; { echo "gestartet $(date)"; screen -ls; } >> logs/launch.txt
