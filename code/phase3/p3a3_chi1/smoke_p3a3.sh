#!/bin/bash
# Probelauf der gesamten Pipeline (20 Schritte, reduziertes Raster)
cd "$(dirname "$0")"; PY=/Users/aaronschmitt/PyCharmMiscProject/.venv/bin/python; W=7
[ -e smoke/smoke_done.txt ] && { echo "Probelauf schon erledigt"; exit 0; }
[ -e meta/run.lock ] && { echo "Hauptlauf aktiv"; exit 0; }
rm -rf smoke; mkdir -p smoke/logs
for i in $(seq 0 $((W-1))); do $PY run_p3a3.py --smoke --phase tune --worker $i --workers $W > smoke/logs/t$i.log 2>&1 & done; wait
$PY run_p3a3.py --smoke --integrity --phase tune > smoke/integrity_tune.json 2>&1
$PY run_p3a3.py --smoke --select > smoke/select.out 2>&1
for i in $(seq 0 $((W-1))); do $PY run_p3a3.py --smoke --phase confirm --worker $i --workers $W > smoke/logs/c$i.log 2>&1 & done; wait
$PY run_p3a3.py --smoke --integrity --phase confirm > smoke/integrity_all.json 2>&1
$PY analyze_p3a3.py --dir smoke --final 20 --tune 42000-42001 --conf 42010-42010 > smoke/analyze.out 2>&1
echo fertig > smoke/smoke_done.txt
