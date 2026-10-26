#!/bin/bash
# Status: bash status_p3a.sh
cd "$(dirname "$0")"
echo "Tuning-Runs fertig:   $(cat logs/tune_worker_*.log 2>/dev/null | grep -c '^done') / 1620"
echo "Bestaetigung fertig:  $(cat logs/confirm_worker_*.log 2>/dev/null | grep -c '^done') / 810"
tail -3 logs/launcher.log 2>/dev/null; [ -e done.txt ] && echo "FERTIG"
