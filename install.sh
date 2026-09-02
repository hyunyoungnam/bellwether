#!/usr/bin/env bash
# One-line install for Linux / macOS / Windows-via-WSL:
#
#   curl -fsSL https://raw.githubusercontent.com/hyunyoungnam/whatsnewai/main/install.sh | bash
#
# What it does: clone the repo to ~/whatsnewai, create a private venv (PEP 668
# machines refuse bare pip), install the `wnai` command onto PATH, fetch the
# search-engine binary and keys. If WNAI_BUNDLE (a file path or URL) is set,
# the data bundle is fetched too — otherwise `wnai fetch-data` is the one step
# left before `wnai serve`.
#
# Overrides, mainly for testing: WNAI_HOME (install dir), WNAI_REPO (clone
# source), WNAI_BIN (where the wnai symlink goes).
set -euo pipefail

REPO="${WNAI_REPO:-https://github.com/hyunyoungnam/whatsnewai}"
DIR="${WNAI_HOME:-$HOME/whatsnewai}"
BIN="${WNAI_BIN:-$HOME/.local/bin}"

say() { printf '\033[1m%s\033[0m\n' "$*"; }

command -v git >/dev/null || { echo "git is required (apt/brew install git)"; exit 1; }
command -v python3 >/dev/null || { echo "python3 is required (3.10+)"; exit 1; }
python3 - <<'EOF' || { echo "python 3.10+ is required"; exit 1; }
import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)
EOF

if [ -d "$DIR/.git" ]; then
    say "updating existing install in $DIR"
    git -C "$DIR" pull --ff-only
else
    say "cloning into $DIR"
    git clone --depth 1 "$REPO" "$DIR"
fi

say "creating the serving venv"
python3 -m venv "$DIR/.venv-serve" 2>/dev/null || {
    echo "python3 -m venv failed — on Debian/Ubuntu: sudo apt install python3-venv"
    exit 1
}
"$DIR/.venv-serve/bin/pip" install -q -e "$DIR"

mkdir -p "$BIN"
ln -sf "$DIR/.venv-serve/bin/wnai" "$BIN/wnai"
case ":$PATH:" in
    *":$BIN:"*) ;;
    *) echo "note: add $BIN to PATH (usually: restart the shell)";;
esac

say "fetching the search engine + keys"
"$BIN/wnai" setup

if [ -n "${WNAI_BUNDLE:-}" ]; then
    say "fetching the data bundle"
    case "$WNAI_BUNDLE" in
        http*) "$BIN/wnai" fetch-data --url "$WNAI_BUNDLE";;
        *)     "$BIN/wnai" fetch-data --file "$WNAI_BUNDLE";;
    esac
    say "done — run: wnai serve"
else
    say "done — next: wnai fetch-data --file <bundle.tar.gz>   then: wnai serve"
fi
