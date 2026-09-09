#!/usr/bin/env bash
# Every audit in one go. The three HTML harnesses drive the real pages in an
# iframe; firefox --screenshot fires on load, and each page holds the load
# event open with /slow?s=N until its run is done, so the shot IS the report.
set -u
cd "$(dirname "$0")/../.."
BASE=${1:-http://127.0.0.1:8001}
PROF="$PWD/reports/.preview/prof"; mkdir -p "$PROF" reports/.preview
cp scripts/audit/*.html reports/.preview/

echo "=== api ==="
PYTHONPATH=src python3 scripts/audit/api_audit.py "$BASE"

echo "=== verifier (accept the paper's words, reject edited ones) ==="
PYTHONPATH=src python3 scripts/audit/verify_bench.py 200

LOG=${WNAI_SERVE_LOG:-run/serve.log}
shot(){ local mark; mark=$(wc -l < "$LOG" 2>/dev/null || echo 0)
  timeout 420 firefox --headless --profile "$PROF" --window-size="$2" \
    --screenshot "$PWD/reports/.preview/audit-$1.png" \
    "$BASE/.preview/$1.html" >/dev/null 2>&1
  # each harness posts its verdict to the server log; the shot is the detail
  tail -n +$((mark+1)) "$LOG" 2>/dev/null | grep '^\[audit\]' || echo "  (no verdict — see the shot)"
  echo "  -> reports/.preview/audit-$1.png"; }
echo "=== ui (chat shell) ===";    shot ui_audit 1000,1500
echo "=== browse ===";             shot browse_audit 1000,1000
echo "=== live (two real runs) ==="; shot live_audit 1000,900
