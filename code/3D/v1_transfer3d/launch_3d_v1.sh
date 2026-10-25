#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"; PY=/Users/aaronschmitt/PyCharmMiscProject/.venv/bin/python
if [[ -z "${CAFFEINATED:-}" ]]; then export CAFFEINATED=1; exec caffeinate -dimsu "$0" "$@"; fi
W=${1:-7}; mkdir -p logs workers meta analysis
if ! mkdir meta/run.lock 2>/dev/null; then echo "ABBRUCH: 3D-V1 laeuft bereits (meta/run.lock existiert)" | tee -a logs/launcher.log; exit 1; fi
trap 'rmdir meta/run.lock 2>/dev/null' EXIT
echo "[$(date +%H:%M:%S)] 3D-V1 Start mit $W Workern" | tee -a logs/launcher.log
$PY run_3d_v1.py --selfcheck > logs/prelaunch_selfcheck.log
pids=(); for ((i=0;i<W;i++)); do nohup "$PY" run_3d_v1.py --worker "$i" --workers "$W" > "logs/worker_$i.stdout.log" 2>&1 & pids+=($!); echo "[$(date +%H:%M:%S)] Worker $i gestartet (PID $!)" | tee -a logs/launcher.log; done
for p in "${pids[@]}"; do wait "$p" || true; done
echo "[$(date +%H:%M:%S)] Worker fertig, Integritaet und Analyse" | tee -a logs/launcher.log
$PY run_3d_v1.py --integrity > logs/integrity.json
$PY analyze_3d_v1.py > logs/analysis.log 2>&1 || echo "Analyse fehlgeschlagen" | tee -a logs/launcher.log
echo '3D-V1 complete' > done.txt; echo "[$(date +%H:%M:%S)] 3D-V1 fertig" | tee -a logs/launcher.log
