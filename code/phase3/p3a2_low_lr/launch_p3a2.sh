#!/bin/bash
# Phase 3a-2 Hauptlauf: Tuning -> Integritaet -> LR-Wahl -> Bestaetigung -> Integritaet -> Analyse
set -uo pipefail
cd "$(dirname "$0")"; PY=/Users/aaronschmitt/PyCharmMiscProject/.venv/bin/python
if [[ -z "${CAFFEINATED:-}" ]]; then export CAFFEINATED=1; exec caffeinate -dimsu /bin/bash "$0" "$@"; fi
W=${1:-7}; mkdir -p logs workers meta analysis
if ! mkdir meta/run.lock 2>/dev/null; then echo "ABBRUCH: P3a-2 laeuft bereits (meta/run.lock existiert)" | tee -a logs/launcher.log; exit 1; fi
trap 'rmdir meta/run.lock 2>/dev/null' EXIT
log(){ echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a logs/launcher.log; }
log "P3a-2 Start mit $W Workern"
shasum -a 256 run_p3a2.py analyze_p3a2.py launch_p3a2.sh > meta/launch_sha256.txt
if ! diff -q <(awk '{print $1}' meta/frozen_sha256.txt) <(awk '{print $1}' meta/launch_sha256.txt) >/dev/null; then log "ABBRUCH: Code weicht vom eingefrorenen Stand ab"; exit 1; fi
$PY run_p3a2.py --selfcheck > logs/prelaunch_selfcheck.json 2>&1 || { log "ABBRUCH: Selbsttest"; exit 1; }
phase(){ local ph=$1; pids=()
  for ((i=0;i<W;i++)); do nohup "$PY" run_p3a2.py --phase "$ph" --worker "$i" --workers "$W" >> "logs/${ph}_worker_$i.log" 2>&1 & pids+=($!); log "$ph Worker $i (PID $!)"; done
  for p in "${pids[@]}"; do wait "$p" || log "Worker PID $p mit Fehler beendet"; done; }
phase tune
$PY run_p3a2.py --integrity --phase tune > logs/integrity_tune.json 2>&1; log "Integritaet Tuning: $(tr -d '\n ' < logs/integrity_tune.json)"
if [[ ! -e meta/lr_selection.json ]]; then
  $PY run_p3a2.py --select > logs/select.log 2>&1 || { log "ABBRUCH: LR-Wahl (Tuning unvollstaendig?)"; exit 1; }
  shasum -a 256 meta/lr_selection.json > meta/lr_selection_sha256.txt; log "LR-Wahl gespeichert"
else log "LR-Wahl existiert bereits, wird beibehalten"; fi
phase confirm
$PY run_p3a2.py --integrity --phase confirm > logs/integrity_all.json 2>&1; log "Integritaet gesamt: $(tr -d '\n ' < logs/integrity_all.json)"
$PY analyze_p3a2.py > logs/analysis.log 2>&1 || log "Analyse fehlgeschlagen"
echo 'P3a-2 complete' > done.txt; log "P3a-2 fertig"
