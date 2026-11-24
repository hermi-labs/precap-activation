#!/bin/bash
# V2g nach V2f starten (mit Claude, 20.11., Freigabe Aaron 20.11. abends: "Du kannst es gerne starten"): wartet, bis screen v2fw0 beendet ist
# und kein Worker mehr laeuft, prueft V2f auf 36 ok-Zeilen und das Netzteil, dann launch_v2g.sh. Frist: Start nur bis 21.11.2026 16:00.
cd "$(dirname "$0")" || exit 1
mkdir -p logs; L=logs/start_nach_v2f.txt; F=../v2f/workers/w0.csv; FRIST=$(date -j -f '%Y-%m-%d %H:%M' '2026-11-21 16:00' +%s)
echo "wartet seit $(date '+%d.%m. %H:%M')" > $L
while screen -ls | grep -qE "\.v2fw0[[:space:]]" || pgrep -f "_run.py --worke[r]" > /dev/null; do
  [ "$(date +%s)" -gt "$FRIST" ] && { echo "Frist vorbei, V2f noch nicht fertig - kein Start $(date '+%d.%m. %H:%M')" >> $L; exit 0; }
  sleep 60
done
n=$(grep -c ',ok,' $F 2>/dev/null)
if [ "$n" != "36" ]; then echo "V2f nicht vollstaendig ($n/36) - kein Start $(date '+%d.%m. %H:%M')" >> $L; exit 1; fi
until pmset -g batt | grep -q "AC Power"; do
  [ "$(date +%s)" -gt "$FRIST" ] && { echo "Frist vorbei, kein Netzteil - kein Start $(date '+%d.%m. %H:%M')" >> $L; exit 0; }
  sleep 60
done
echo "V2f fertig (36/36), starte V2g $(date '+%d.%m. %H:%M')" >> $L
bash launch_v2g.sh >> $L 2>&1
