#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"; PY=/Users/aaronschmitt/PyCharmMiscProject/.venv/bin/python
if [[ -z "${CAFFEINATED:-}" ]]; then export CAFFEINATED=1; exec caffeinate -dimsu "$0" "$@"; fi
W=${1:-7}; mkdir -p logs workers meta analysis
echo "[$(date +%H:%M:%S)] V18 Start mit $W Workern (Schlafschutz: caffeinate -dimsu aktiv)" | tee -a logs/launcher.log
$PY run_v18_lr.py --selfcheck > logs/prelaunch_selfcheck.log
pids=(); for ((i=0;i<W;i++)); do nohup "$PY" run_v18_lr.py --worker "$i" --workers "$W" > "logs/worker_$i.stdout.log" 2>&1 & pids+=($!); echo "[$(date +%H:%M:%S)] Worker $i gestartet (PID $!)" | tee -a logs/launcher.log; done
echo "[$(date +%H:%M:%S)] Warte auf Worker. Fortschritt: python status.py" | tee -a logs/launcher.log
for p in "${pids[@]}"; do wait "$p" || true; done
echo "[$(date +%H:%M:%S)] Worker fertig, Integritaet und Analyse" | tee -a logs/launcher.log
$PY run_v18_lr.py --integrity > logs/integrity.json
$PY analyze_v18.py > logs/analysis.log 2>&1 || echo "Analyse fehlgeschlagen, siehe logs/analysis.log" | tee -a logs/launcher.log
echo 'V18 complete' > done.txt; echo "[$(date +%H:%M:%S)] V18 fertig" | tee -a logs/launcher.log
