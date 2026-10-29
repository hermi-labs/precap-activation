#!/bin/bash
cd "$(dirname "$0")"; n=$(cat workers/w*.csv 2>/dev/null | grep -vc '^run_id'); echo "fertig: $n / 240"; tail -n 2 logs/w0.log logs/w1.log; screen -ls | grep q1w || echo "keine Worker aktiv"
