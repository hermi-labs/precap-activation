#!/bin/bash
# Q1 starten: einfrieren, Selbsttest, Tempo-Probe, Rauchtest auf echten Daten, dann 2 Worker in screen
set -e
cd "$(dirname "$0")"
PY=${PY:-$HOME/precap_mlx_venv/bin/python}
mkdir -p meta logs
if [ -d workers ]; then echo "workers/ existiert schon - Abbruch, nichts ueberschrieben"; exit 1; fi
shasum -a 256 q1_run.py q1_analyze.py Q1_Praeregistrierung.md lr_from_p3a3.csv src/precap/*.py > meta/frozen_sha256.txt
date '+%Y-%m-%dT%H:%M:%S%z' > meta/frozen_at.txt
$PY -c "import mlx.core as mx, scipy; print(mx.__version__, mx.default_device(), scipy.__version__)" > meta/env.txt 2>&1 || $PY -c "import mlx.core as mx; print(mx.__version__, mx.default_device(), 'scipy fehlt')" > meta/env.txt
$PY q1_run.py --selfcheck > meta/selfcheck.json
$PY q1_run.py --smoke --limit 4 --worker 0 > logs/smoke.log 2>&1
$PY q1_run.py --probe > meta/probe.json 2> logs/probe.err
for w in 0 1; do screen -dmS q1w$w bash -c "$PY q1_run.py --worker $w --workers 2 >> logs/w$w.log 2>&1"; done
echo "gestartet $(date)"; cat meta/probe.json
