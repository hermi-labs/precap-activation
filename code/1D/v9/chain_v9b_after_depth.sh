#!/bin/bash
cd /Users/aaronschmitt/PyCharmMiscProject
LOG=chain.log
echo "$(date '+%F %T') chain armed: waiting for v9_depth" >> $LOG
while pgrep -f 'capexp_v9_depth.py --shard' > /dev/null; do sleep 30; done
if [ -f v9_depth/v9_depth_done.txt ]; then echo "$(date '+%F %T') v9_depth finished OK" >> $LOG; else echo "$(date '+%F %T') v9_depth ended WITHOUT done marker (check logs); starting v9b anyway" >> $LOG; fi
echo "$(date '+%F %T') starting v9b" >> $LOG
bash v9b/run_v9b.sh >> $LOG 2>&1 && echo "$(date '+%F %T') v9b finished OK" >> $LOG || echo "$(date '+%F %T') v9b FAILED" >> $LOG
