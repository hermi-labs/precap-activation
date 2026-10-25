#!/bin/zsh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
cat <<'TXT'
For future long studies on this Mac, use this pattern:

caffeinate -dimsu /Users/aaronschmitt/PyCharmMiscProject/.venv/bin/python run_*.py ...

Meaning:
- d: prevent display sleep
- i: prevent idle sleep
- m: prevent disk sleep
- s: keep system awake while on AC power
- u: declare user activity

For multi-worker runs:
1. Start one caffeinate-wrapped worker per shard.
2. Give each worker its own output CSV.
3. Merge only after all shards complete.
TXT
