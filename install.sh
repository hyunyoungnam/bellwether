#!/usr/bin/env bash
# One-line install for Linux / macOS / Windows-via-WSL:
#
#   curl -fsSL https://raw.githubusercontent.com/hyunyoungnam/bellwether/main/install.sh | bash
#
# What it does: clone the repo to ~/.bellwether, create a private venv (PEP 668
# machines refuse bare pip), install the `bellwether` command onto PATH, fetch the
# search-engine binary and keys. If WNAI_BUNDLE (a file path or URL) is set,
# the data bundle is fetched too — otherwise `bellwether fetch-data` is the one step
# left before `bellwether serve`.
#
# Overrides, mainly for testing: WNAI_HOME (install dir), WNAI_REPO (clone
# source), WNAI_BIN (where the wnai symlink goes).
set -euo pipefail

REPO="${WNAI_REPO:-https://github.com/hyunyoungnam/bellwether}"
# an app dir, not a workspace: hidden by default, like other installed tools.
# Everything inside stays inspectable — the data being auditable is a feature.
DIR="${WNAI_HOME:-$HOME/.bellwether}"
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
if ! python3 -m venv "$DIR/.venv-serve" 2>/dev/null; then
    rm -rf "$DIR/.venv-serve"    # a half-made venv breaks the retry
    PYV=$(python3 -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
    echo "python3 -m venv failed — on Debian/Ubuntu/WSL run:"
    echo "    sudo apt install python${PYV}-venv"
    echo "then re-run this installer."
    exit 1
fi
"$DIR/.venv-serve/bin/pip" install -q -e "$DIR"

mkdir -p "$BIN"
ln -sf "$DIR/.venv-serve/bin/bellwether" "$BIN/bellwether"
case ":$PATH:" in
    *":$BIN:"*) ;;
    *) echo "note: $BIN is not on PATH yet — open a new terminal (or:"
       echo "      source ~/.profile) — until then, use $BIN/bellwether";;
esac

say "fetching the search engine + keys"
"$BIN/bellwether" setup

if [ -n "${WNAI_BUNDLE:-}" ]; then
    say "fetching the data bundle"
    case "$WNAI_BUNDLE" in
        http*) "$BIN/bellwether" fetch-data --url "$WNAI_BUNDLE";;
        *)     "$BIN/bellwether" fetch-data --file "$WNAI_BUNDLE";;
    esac
    say "done — run: bellwether serve"
else
    say "done — next: bellwether fetch-data --file <bundle.tar.gz>   then: bellwether serve"
fi
