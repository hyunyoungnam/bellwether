#!/usr/bin/env bash
# Both extraction passes, census first (sparsity depends on it). Resumable:
# re-running skips papers already present in each output file.
set -u
set -o pipefail   # else $? reports grep, and a crashed run looks like success
cd /home/hyunyoungnam/ICML2026
export HF_HOME=/home/hyunyoungnam/ICML2026/.cache/hf
export PATH="/home/hyunyoungnam/ICML2026/.venv/bin:$PATH"
export PYTHONPATH=src
for SRC in abstract fulltext; do
  echo "=== $SRC pass starting $(date -Is) ==="
  .venv/bin/python -m icml.extract_facts --source "$SRC" 2>&1 \
    | grep -vE "it/s\]|Capturing|^$"
  echo "=== $SRC pass exit=${PIPESTATUS[0]} $(date -Is) ==="
done
