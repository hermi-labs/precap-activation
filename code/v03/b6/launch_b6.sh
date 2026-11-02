#!/bin/bash
# B6 starten: Netzteil pruefen, einfrieren, Selbsttest, Mechanik- und Auswertungstest, Rauchtest CIFAR-10 d100 (4 Tuning + 6 Bestaetigung, alle ok),
# dann 2 Worker in screen (b6w0, b6w1): erst Tuning, dann LR-Wahl, dann Bestaetigung. GPU-Lauf etwa 1 h, Freigabe von Aaron fuer die Nacht (30./31.10.).
cd "$(dirname "$0")"
PY=${PY:-$HOME/precap_mlx_venv/bin/python}
PYS=${PYS:-$HOME/PyCharmMiscProject/.venv/bin/python}   # mit scipy fuer den Auswertungstest
D=$HOME/PyCharmMiscProject/data/cifar10/cifar-10-batches-bin
mkdir -p meta logs
if [ -d workers ] || [ -d tune ]; then echo "Ergebnisordner existiert schon - Abbruch" > logs/launch.txt; exit 1; fi
if screen -ls | grep -q -E "gpt2w|gpt3w|b1w|b2w|b3w|b4w|b5w|b6w"; then echo "andere Worker laufen - Abbruch" > logs/launch.txt; exit 1; fi
if ! pmset -g batt | grep -q "AC Power"; then echo "nicht am Netzteil - Abbruch (Akku pausiert Laeufe bei Deckel zu)" > logs/launch.txt; exit 1; fi
shasum -a 256 b6_run.py b6_analyze.py test_b6_mechanik.py test_b6_analyse.py B6_Praeregistrierung.md lr_from_a.csv lr_from_p3a3.csv lr_from_b3.csv src/precap/*.py $D/*.bin > meta/frozen_sha256.txt
date '+%Y-%m-%dT%H:%M:%S%z' > meta/frozen_at.txt
$PY b6_run.py --selfcheck > meta/selfcheck.json 2>/dev/null || { echo "Selbsttest fehlgeschlagen" > logs/launch.txt; exit 1; }
grep -q '"n_confirm": 61' meta/selfcheck.json && grep -q '"n_tune": 48' meta/selfcheck.json || { echo "Selbsttest-Werte nicht ok" > logs/launch.txt; exit 1; }
$PY test_b6_mechanik.py > meta/mechanik.txt 2>&1 || { echo "Mechanik-Test fehlgeschlagen" > logs/launch.txt; exit 1; }
$PYS -I test_b6_analyse.py > meta/auswertungstest.txt 2>&1 || { echo "Auswertungstest fehlgeschlagen" > logs/launch.txt; exit 1; }
rm -rf smoke; $PY b6_run.py --smoke --worker 0 --workers 1 > logs/smoke.log 2>&1
a=$(grep -c ',ok,' smoke/tune/workers/w0.csv 2>/dev/null); b=$(grep -c ',ok,' smoke/workers/w0.csv 2>/dev/null)
if [ "$a" != "4" ] || [ "$b" != "6" ]; then echo "Rauchtest nicht ok (tune $a, confirm $b) - kein Start" > logs/launch.txt; exit 1; fi
echo "Tests ok $(date '+%H:%M')" > logs/launch.txt
for w in 0 1; do screen -dmS b6w$w caffeinate -i bash -c "{ $PY b6_run.py --worker $w --workers 2; } >> logs/w$w.log 2>&1"; done
sleep 2; { echo "gestartet $(date)"; screen -ls; } >> logs/launch.txt
