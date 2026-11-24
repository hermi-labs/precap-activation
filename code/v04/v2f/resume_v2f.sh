#!/bin/bash
# V2f nach Netzteil-Ausfall wieder aufnehmen (mit Claude, 20.11.): wartet auf das Netzteil (5 min stabil) und startet dann denselben Worker
# und Waechter wie launch_v2f.sh, ohne neuen Freeze (eingefrorene Dateien unveraendert, Pruefung per shasum -c). Der Worker ist wiederaufnehmbar.
# Startet nur bis zur Frist 21.11.2026 06:00, danach Abbruch ohne Start.
cd "$(dirname "$0")" || exit 1
PY=${PY:-$HOME/precap_mlx_venv/bin/python}; FRIST=$(date -j -f '%Y-%m-%d %H:%M' '2026-11-21 06:00' +%s)
while :; do
  if [ "$(date +%s)" -gt "$FRIST" ]; then echo "resume: Frist vorbei, kein Start $(date '+%d.%m. %H:%M')" >> logs/launch.txt; exit 0; fi
  if pmset -g batt | grep -q "AC Power"; then sleep 300; pmset -g batt | grep -q "AC Power" && break; fi
  sleep 60
done
if screen -ls | grep -qE "\.v2fw0[[:space:]]" || pgrep -f "_run.py --worke[r]" > /dev/null; then echo "resume: Worker laeuft schon, kein Start $(date '+%H:%M')" >> logs/launch.txt; exit 0; fi
shasum -a 256 -c meta/frozen_sha256.txt > /dev/null || { echo "resume: Hashes weichen ab, kein Start $(date '+%H:%M')" >> logs/launch.txt; exit 1; }
screen -dmS v2fw0 caffeinate -i bash -c "{ $PY v2f_run.py --worker 0 --workers 1; } >> logs/w0.log 2>&1"
screen -dmS v2fwache bash -c "while screen -ls | grep -qE '\.v2fw0[[:space:]]'; do s=0; if ! pmset -g batt | grep -q 'AC Power'; then sleep 60; pmset -g batt | grep -q 'AC Power' || s=1; fi; if [ \$s = 1 ]; then pkill -f 'v2f_run.py --worker'; sleep 1; echo \"Waechter: beendet um \$(date +%H:%M)\" >> logs/launch.txt; break; fi; sleep 30; done"
sleep 2; { echo "resume: wieder gestartet $(date)"; screen -ls; } >> logs/launch.txt
