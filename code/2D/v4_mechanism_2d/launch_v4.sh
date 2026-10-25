#!/bin/zsh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
PY="/Users/aaronschmitt/PyCharmMiscProject/.venv/bin/python"
cd "$ROOT"
"$PY" run_v4_snapshots.py selfcheck > logs/prelaunch_selfcheck.log
"$PY" - <<'PY'
from pathlib import Path
import csv, importlib.util
root=Path('.').resolve()
runner_path=root/'run_v4_snapshots.py'
spec=importlib.util.spec_from_file_location('run_v4_snapshots_mod', runner_path)
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
metrics=root/'results'/'checkpoint_metrics.csv'
done=set()
if metrics.exists():
    for r in csv.DictReader(metrics.open()):
        if r['step']=='3000':
            done.add(r['run_id'])
cs=list(mod.configs())
remaining=[c for c in cs if mod.run_id(c) not in done]
with (root/'logs'/'progress.log').open('a') as log:
    log.write(f"launch total={len(cs)} already_complete={len(done)} remaining={len(remaining)}\n")
for i,c in enumerate(remaining,1):
    rid=mod.run_id(c)
    with (root/'logs'/'progress.log').open('a') as log:
        log.write(f"start {i}/{len(remaining)} {rid}\n")
    mod.execute(c)
    with (root/'logs'/'progress.log').open('a') as log:
        log.write(f"done {i}/{len(remaining)} {rid}\n")
print(f"completed_remaining={len(remaining)}")
PY
"$PY" analyze_v4_snapshots.py > logs/analyze_final.log
"$PY" - <<'PY'
import csv
from pathlib import Path
root=Path('.').resolve()
rows=list(csv.DictReader((root/'results'/'checkpoint_metrics.csv').open()))
full={r['run_id'] for r in rows if r['step']=='3000'}
expected=75
assert len(full)==expected, (len(full),expected)
assert len(rows)==expected*5, (len(rows),expected*5)
(root/'done.txt').write_text(f'DONE full_runs={len(full)} checkpoint_rows={len(rows)}\n')
(root/'logs'/'integrity.txt').write_text(f'full_runs={len(full)} expected={expected} checkpoint_rows={len(rows)} expected_checkpoint_rows={expected*5}\n')
print((root/'logs'/'integrity.txt').read_text().strip())
PY
