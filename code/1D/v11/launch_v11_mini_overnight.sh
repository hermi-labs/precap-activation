#!/bin/zsh
set -euo pipefail
cd /Users/aaronschmitt/PyCharmMiscProject/v11
mkdir -p logs meta results
(
  caffeinate -dimsu bash -lc '
    cd /Users/aaronschmitt/PyCharmMiscProject/v11 || exit 1
    bash run_v11_mini.sh > logs/mini_overnight.log 2>&1
  '
) > logs/mini_launcher.log 2>&1 &
echo $! | tee meta/mini_launcher.pid
