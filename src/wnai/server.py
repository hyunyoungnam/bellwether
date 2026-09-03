"""One port for the whole product: static files + the search endpoints.

The static site (reports/) and Meilisearch share one port so that a LAN URL —
or the cloudflared quick tunnel, which forwards exactly one port — is the whole
address. This proxy serves the files and forwards TWO paths, POST /meili/search
and POST /meili/similar, to the local Meilisearch with a search-only key
injected server-side. Nothing else of Meilisearch is exposed: no write
endpoints, no keys in the client, no other indexes.

Degrades without Meilisearch: if the key file is missing or the engine is down,
the two POST paths answer 502 and the page falls back to its shipped indexes
(word-AND search, 20-neighbour similar) — the same failure mode the client
already handles.
"""
from __future__ import annotations

import json
import os
import urllib.request
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = str(ROOT / "reports")
KEY_FILE = ROOT / "data/meili/search_key"
MEILI_ADDR = os.environ.get("WNAI_MEILI_ADDR", "127.0.0.1:7700")
MEILI_BASE = f"http://{MEILI_ADDR}/indexes/papers"
ROUTES = {"/meili/search": MEILI_BASE + "/search",
          "/meili/similar": MEILI_BASE + "/similar"}


def search_key() -> str | None:
    try:
        return KEY_FILE.read_text().strip()
    except OSError:
        return None


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, directory: str = DOCS, **kw):
        super().__init__(*a, directory=directory, **kw)

    def log_message(self, *a):  # quiet
        pass

    def end_headers(self):
        # Without this the browser caches index.html heuristically and users
        # keep running week-old JS. no-cache means revalidate, not re-download:
        # an unchanged file answers 304, so the cost is one round-trip.
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def do_GET(self):
        # the conversation is the front door; the built corpus view lives on
        # at /browse as the evidence surface its cite chips open
        if self.path in ("/", "/index.htm"):
            self.path = "/chat.html"
        elif self.path.split("#")[0].split("?")[0] == "/browse":
            self.path = "/index.html"
        elif self.path.startswith("/paper/"):
            from . import chat
            try:
                self._json(200, chat.card(int(self.path[7:])))
            except Exception as exc:  # noqa: BLE001
                self._json(400, {"error": type(exc).__name__})
            return
        elif self.path == "/chats" or self.path.startswith("/chats/"):
            from . import chat
            try:
                cid = self.path[7:]
                out = chat.get_chat(cid) if cid else chat.list_chats()
                self._json(200, out)
            except FileNotFoundError:
                self._json(404, {"error": "no such chat"})
            except Exception as exc:  # noqa: BLE001
                self._json(400, {"error": type(exc).__name__})
            return
        # test hook: a resource that finishes late, so headless screenshots
        # taken "after load" happen after the page's async work too
        if self.path.startswith("/slow"):
            import time as _t
            _t.sleep(4)
            body = b"ok"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()

    def do_POST(self):
        if self.path == "/chat/stream":
            try:
                body = json.loads(
                    self.rfile.read(int(self.headers.get("Content-Length", 0)))
                    or b"{}")
                from . import chat
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("X-Accel-Buffering", "no")
                self.end_headers()

                def emit(ev: dict) -> None:
                    self.wfile.write(
                        b"data: " + json.dumps(ev, ensure_ascii=False).encode()
                        + b"\n\n")
                    self.wfile.flush()

                chat.stream(body, emit)
            except (BrokenPipeError, ConnectionResetError):
                pass                        # reader left mid-generation
            except Exception as exc:  # noqa: BLE001
                try:
                    self._json(502, {"error": f"{type(exc).__name__}: {exc}"[:300]})
                except OSError:
                    pass
            return
        if self.path == "/chat":
            # the conversational loop: spawns the user's own logged-in coding
            # agent (no API key), verifies every quoted anchor before replying
            try:
                body = json.loads(
                    self.rfile.read(int(self.headers.get("Content-Length", 0)))
                    or b"{}")
                from . import chat
                self._json(200, chat.handle(body))
            except Exception as exc:  # noqa: BLE001
                self._json(502, {"error": f"{type(exc).__name__}: {exc}"[:300]})
            return
        meili = ROUTES.get(self.path)
        key = search_key()
        if meili is None or key is None:
            self._json(404 if meili is None else 502,
                       {"error": "NoSearchKey" if meili else "NotFound"})
            return
        try:
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            req = urllib.request.Request(
                meili, data=body, method="POST",
                headers={"Authorization": f"Bearer {key}",
                         "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as fh:
                out = fh.read()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)
        except Exception as exc:  # noqa: BLE001
            self._json(502, {"error": type(exc).__name__})

    def _json(self, code: int, obj: dict) -> None:
        msg = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(msg)))
        self.end_headers()
        self.wfile.write(msg)


def make_server(host: str = "0.0.0.0", port: int = 8001,
                docs: str = DOCS) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), partial(Handler, directory=docs))


if __name__ == "__main__":
    make_server().serve_forever()
