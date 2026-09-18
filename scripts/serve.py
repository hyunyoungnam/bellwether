"""Back-compat wrapper — the server moved to src/wnai/server.py.

Existing invocations (`python3 scripts/serve.py`, the tunnel restart line in
CLAUDE.md) keep working; new installs use `wnai serve`, which also starts
Meilisearch and prints the local address.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bellwether.server import make_server  # noqa: E402

if __name__ == "__main__":
    make_server().serve_forever()
