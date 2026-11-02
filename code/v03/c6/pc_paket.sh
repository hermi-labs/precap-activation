#!/bin/bash
# Baut studien/c6_paket.tar fuer Aarons PC: Code von C6 (ohne smoke/meta/logs/workers) und CIFAR-10 (Binaerdateien). Nichts wird geloescht.
set -e
cd "$(dirname "$0")/.."
OUT=c6_paket.tar
if [ -e "$OUT" ]; then echo "$OUT existiert schon - bitte umbenennen" ; exit 1; fi
STAGE=$(mktemp -d)
mkdir -p "$STAGE/c6" "$STAGE/data/cifar10"
rsync -a --exclude /smoke --exclude /meta --exclude /logs --exclude /workers --exclude /test/alt --exclude '__pycache__' c6/ "$STAGE/c6/"
rsync -a ../data/cifar10/cifar-10-batches-bin "$STAGE/data/cifar10/"
tar -cf "$OUT" -C "$STAGE" c6 data
echo "gebaut: $(pwd)/$OUT ($(du -h "$OUT" | cut -f1))"; tar -tf "$OUT" | grep -c ""
