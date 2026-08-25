"""One port for the whole product: static files + the search endpoint.

The cloudflared quick tunnel forwards exactly one local port, so the static
site (reports/) and Meilisearch must share it. This proxy serves the files and
forwards TWO paths — POST /meili/search and POST /meili/similar — to the local
Meilisearch with a search-only key injected server-side. Nothing else of
Meilisearch is exposed: no write endpoints, no keys in the client, no other
indexes.

    python3 scripts/serve.py          # port 8001, same as the old http.server
"""
import json
import urllib.request
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KEY = (ROOT / "data/meili/search_key").read_text().strip()
DOCS = str(ROOT / "reports")
PORT = 8001
MEILI_BASE = "http://127.0.0.1:7700/indexes/papers"
ROUTES = {"/meili/search": MEILI_BASE + "/search",
          "/meili/similar": MEILI_BASE + "/similar"}


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=DOCS, **kw)

    def log_message(self, *a):  # quiet
        pass

    def end_headers(self):
        # Without this the browser caches index.html heuristically and users
        # keep running week-old JS. no-cache means revalidate, not re-download:
        # an unchanged file answers 304, so the cost is one round-trip.
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def do_POST(self):
        MEILI = ROUTES.get(self.path)
        if MEILI is None:
            self.send_error(404)
            return
        try:
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            req = urllib.request.Request(
                MEILI, data=body, method="POST",
                headers={"Authorization": f"Bearer {KEY}",
                         "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as fh:
                out = fh.read()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)
        except Exception as exc:  # noqa: BLE001
            msg = json.dumps({"error": type(exc).__name__}).encode()
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(msg)))
            self.end_headers()
            self.wfile.write(msg)


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
