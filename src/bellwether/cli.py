"""The launcher: install on Linux/Windows, serve on the local network.

    bellwether serve            # search engine + site on one port; prints the LAN URL
    bellwether setup            # fetch the Meilisearch binary, generate keys
    bellwether status           # what is running, what data exists

`serve` is the whole runtime — everything else in this repo (collection,
extraction, embedding) BUILDS the data this serves and never has to run on the
user's machine. Stdlib-only on purpose: the serving layer must work on a
machine that can't run the pipeline (Windows, no GPU, no .venv).
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import secrets
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BIN_DIR = ROOT / "bin"
MEILI_DIR = ROOT / "data" / "meili"
MEILI_VERSION = "v1.53.1"          # the version the index was built with
MEILI_ADDR = os.environ.get("WNAI_MEILI_ADDR", "127.0.0.1:7700")

# what a fresh install needs to SERVE (the pipeline that builds these never
# runs on the user's machine): the site, the search index, and the processed
# files the MCP tools read. Embedding matrices and interim files stay out.
BUNDLE_GLOBS = [
    "reports/index.html", "reports/chat.html", "reports/data",
    "data/meili/db",
    "data/processed/union.json", "data/processed/topics*.json",
    "data/processed/card_terms.json",
    "data/processed/citations.json", "data/processed/neighbors_union.json",
    "data/processed/contacts.json", "data/processed/papers*.jsonl",
]

_IS_WIN = platform.system() == "Windows"
_MEILI_BIN = BIN_DIR / ("meilisearch.exe" if _IS_WIN else "meilisearch")
_MEILI_PORT = int(MEILI_ADDR.rsplit(":", 1)[1])


# ---------------------------------------------------------------- helpers

def _port_open(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket() as s:
        s.settimeout(0.3)
        return s.connect_ex((host, port)) == 0


def _lan_ip() -> str:
    """The address other devices on the network reach this machine at."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))   # no packet is sent; picks the route
            return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"


def _meili_asset() -> str:
    sysname = platform.system()
    arch = platform.machine().lower()
    if sysname == "Linux":
        return "meilisearch-linux-" + ("aarch64" if arch in ("arm64", "aarch64") else "amd64")
    if sysname == "Windows":
        return "meilisearch-windows-amd64.exe"
    if sysname == "Darwin":
        return "meilisearch-macos-" + ("apple-silicon" if arch == "arm64" else "amd64")
    raise SystemExit(f"unsupported platform: {sysname}/{arch}")


def _download(url: str, dest: Path) -> None:
    """urlretrieve, minus the Python 3.13+ trap: default contexts now set
    VERIFY_X509_STRICT, which rejects certificates lacking an Authority Key
    Identifier — common behind AV/proxy TLS re-signing (seen on real WSL).
    curl and git accept those chains; so do we. Chain and hostname are still
    verified — only the strict extension checks are relaxed."""
    import ssl
    ctx = ssl.create_default_context()
    if hasattr(ssl, "VERIFY_X509_STRICT"):
        ctx.verify_flags &= ~ssl.VERIFY_X509_STRICT
    req = urllib.request.Request(url, headers={"User-Agent": "bellwether"})
    with urllib.request.urlopen(req, context=ctx, timeout=60) as r, \
            open(dest, "wb") as fh:
        total = int(r.headers.get("Content-Length") or 0)
        got = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
            got += len(chunk)
            if total > 100 << 20 and got % (50 << 20) < (1 << 20):
                print(f"  {got / 1e6:,.0f} / {total / 1e6:,.0f} MB", flush=True)


def _http(url: str, data: dict | None = None, key: str | None = None,
          timeout: int = 10) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(data).encode() if data is not None else None,
        method="POST" if data is not None else "GET",
        headers={"Content-Type": "application/json",
                 **({"Authorization": f"Bearer {key}"} if key else {})})
    with urllib.request.urlopen(req, timeout=timeout) as fh:
        return json.loads(fh.read() or b"{}")


def _spawn_meili(master_key: str) -> subprocess.Popen:
    cmd = [str(_MEILI_BIN), "--db-path", str(MEILI_DIR / "db"),
           "--http-addr", MEILI_ADDR, "--master-key", master_key,
           "--no-analytics"]
    kw: dict = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
    if _IS_WIN:
        kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
    return subprocess.Popen(cmd, **kw)


def _wait_health(seconds: float = 20.0) -> bool:
    end = time.time() + seconds
    while time.time() < end:
        try:
            _http(f"http://{MEILI_ADDR}/health", timeout=2)
            return True
        except (urllib.error.URLError, OSError):
            time.sleep(0.4)
    return False


def _refresh_search_key(master: str, force: bool = False) -> bool:
    """Write data/meili/search_key from the running/spawnable engine.

    Key VALUES derive from the master key + the key's uid stored in the db, so
    a bundled db under a fresh install's master key yields a different value —
    that is why fetch-data forces a refresh after unpacking.
    """
    sk_file = MEILI_DIR / "search_key"
    if sk_file.exists() and not force:
        return True
    started = None
    if not _port_open(_MEILI_PORT):
        started = _spawn_meili(master)
        if not _wait_health():
            return False
    try:
        keys = _http(f"http://{MEILI_ADDR}/keys", key=master)
        found = next((k["key"] for k in keys.get("results", [])
                      if k.get("actions") == ["search"]), None)
        if not found:
            made = _http(f"http://{MEILI_ADDR}/keys",
                         data={"description": "search-only, injected by the proxy",
                               "actions": ["search"], "indexes": ["*"],
                               "expiresAt": None}, key=master)
            found = made["key"]
        sk_file.write_text(found)
        return True
    except urllib.error.HTTPError:
        # something else answers on the port, or it rejects our master key —
        # never crash setup over it; serve degrades and status explains
        return False
    finally:
        if started:
            started.terminate()


# ---------------------------------------------------------------- commands

def cmd_setup(args: argparse.Namespace) -> int:
    """Binary + keys. Idempotent; safe to re-run."""
    BIN_DIR.mkdir(parents=True, exist_ok=True)
    MEILI_DIR.mkdir(parents=True, exist_ok=True)

    if not _MEILI_BIN.exists():
        asset = _meili_asset()
        url = (f"https://github.com/meilisearch/meilisearch/releases/download/"
               f"{MEILI_VERSION}/{asset}")
        print(f"downloading {asset} {MEILI_VERSION} ...")
        try:
            _download(url, _MEILI_BIN)
        except (urllib.error.URLError, OSError) as exc:
            _MEILI_BIN.unlink(missing_ok=True)
            print(f"download failed ({exc}) — check the network and re-run "
                  "`bellwether setup`", file=sys.stderr)
            return 1
        if not _IS_WIN:
            _MEILI_BIN.chmod(0o755)
        print(f"  -> {_MEILI_BIN}")
    else:
        print(f"binary present: {_MEILI_BIN}")

    mk_file = MEILI_DIR / "master_key"
    if not mk_file.exists():
        mk_file.write_text(secrets.token_urlsafe(32))
        print("generated master key")
    master = mk_file.read_text().strip()

    if (MEILI_DIR / "search_key").exists():
        print("search key present")
    elif _refresh_search_key(master):
        print("search key written")
    else:
        print("meilisearch did not come up; run setup again", file=sys.stderr)
        return 1
    print("\nsetup complete — next: `bellwether fetch-data --file <bundle>` "
          "(or build the data with the pipeline), then `bellwether serve`")
    return 0


def cmd_bundle(args: argparse.Namespace) -> int:
    """Release engineering, build machine only: pack what an install serves.

    The search db must not be copied under a live engine, so the engine is
    stopped for the (fast) tar step and restarted before the (slow) gzip.
    """
    if _IS_WIN:
        print("bundle runs on the build machine (POSIX)", file=sys.stderr)
        return 1
    files: list[Path] = []
    for g in BUNDLE_GLOBS:
        hits = sorted(ROOT.glob(g))
        if not hits:
            print(f"  missing (skipped): {g}")
        files += hits
    out_dir = ROOT / "dist"
    out_dir.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d")
    tar_path = out_dir / f"bellwether-data-{stamp}.tar"

    was_up = _port_open(_MEILI_PORT)
    if was_up:
        subprocess.run(["pkill", "-f", "meilisearch"], check=False)
        for _ in range(50):
            if not _port_open(_MEILI_PORT):
                break
            time.sleep(0.2)
        print("engine stopped for a consistent copy")
    try:
        rel = [str(p.relative_to(ROOT)) for p in files]
        subprocess.run(["tar", "cf", str(tar_path), *rel], cwd=ROOT, check=True)
    finally:
        if was_up:
            mk = (MEILI_DIR / "master_key").read_text().strip()
            kw: dict = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL,
                        "start_new_session": True}
            subprocess.Popen([str(_MEILI_BIN), "--db-path", str(MEILI_DIR / "db"),
                              "--http-addr", MEILI_ADDR, "--master-key", mk,
                              "--no-analytics"], **kw)
            print("engine restarted")
    print("compressing ...")
    subprocess.run(["gzip", "-f", str(tar_path)], check=True)
    gz = tar_path.with_suffix(".tar.gz")
    print(f"-> {gz}  ({gz.stat().st_size/1e6:,.0f} MB)")
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    """Unpack a data bundle into this install, then re-derive the search key."""
    import tarfile
    if args.file:
        src = Path(args.file)
    elif args.url:
        dest = ROOT / "dist"
        dest.mkdir(exist_ok=True)
        src = dest / Path(args.url).name
        print(f"downloading {args.url} ...")
        try:
            _download(args.url, src)
        except (urllib.error.URLError, OSError) as exc:
            src.unlink(missing_ok=True)
            print(f"download failed ({exc}) — check the network and re-run",
                  file=sys.stderr)
            return 1
    else:
        print("pass --file <bundle.tar.gz> or --url <https://...>", file=sys.stderr)
        return 1
    print(f"unpacking {src.name} into {ROOT} ...")
    with tarfile.open(src) as tf:
        names = tf.getnames()
        # a bundled search db REPLACES any local one — merging two LMDB
        # directories file-by-file is corruption, not an update
        if any(n.startswith("data/meili/db") for n in names) \
                and (MEILI_DIR / "db").exists():
            import shutil
            shutil.rmtree(MEILI_DIR / "db")
        tf.extractall(ROOT, filter="data")
    mk_file = MEILI_DIR / "master_key"
    if mk_file.exists():
        # the bundled db carries key uids from the build machine; values
        # re-derive under THIS install's master key
        ok = _refresh_search_key(mk_file.read_text().strip(), force=True)
        print("search key refreshed" if ok
              else "engine not available — rerun `bellwether setup` before serving")
    else:
        print("no master key yet — run `bellwether setup`")
    print("done — `bellwether serve`")
    return 0


def cmd_mcp(args: argparse.Namespace) -> int:
    from .mcp import serve_stdio
    return serve_stdio()


def cmd_serve(args: argparse.Namespace) -> int:
    from . import server

    if not Path(server.DOCS, "index.html").exists():
        print(f"nothing to serve: {server.DOCS}/index.html missing "
              "(build with `python3 -m icml.site` or unpack a release)",
              file=sys.stderr)
        return 1

    child = None
    if not args.no_meili and not _port_open(_MEILI_PORT):
        mk_file = MEILI_DIR / "master_key"
        if _MEILI_BIN.exists() and mk_file.exists() and (MEILI_DIR / "db").exists():
            child = _spawn_meili(mk_file.read_text().strip())
            ok = _wait_health()
            print(f"meilisearch: {'up' if ok else 'FAILED (site degrades to shipped indexes)'}")
        else:
            print("meilisearch: not set up (run `bellwether setup`); "
                  "site degrades to shipped indexes")
    else:
        print("meilisearch: already running" if _port_open(_MEILI_PORT) else "meilisearch: skipped")

    # SIGTERM (service stop, timeout) must clean up the engine child exactly
    # like Ctrl+C does — SystemExit propagates through serve_forever's frame
    import signal
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

    srv = server.make_server(args.host, args.port)
    print(f"\n  local:    http://127.0.0.1:{args.port}")
    if args.host == "0.0.0.0":
        print(f"  network:  http://{_lan_ip()}:{args.port}   <- other devices on this network")
    print("\nCtrl+C stops everything")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
        if child:
            child.terminate()
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    site = Path(ROOT, "reports", "index.html")
    print(f"site built:     {'yes' if site.exists() else 'no'}")
    print(f"meili binary:   {'yes' if _MEILI_BIN.exists() else 'no  (bellwether setup)'}")
    print(f"meili index:    {'yes' if (MEILI_DIR / 'db').exists() else 'no  (scripts/search_index.py)'}")
    print(f"meili running:  {'yes' if _port_open(_MEILI_PORT) else 'no'}")
    print(f"server on 8001: {'yes' if _port_open(8001) else 'no'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="bellwether", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="serve the site + search on the local network")
    s.add_argument("--port", type=int, default=8001)
    s.add_argument("--host", default="0.0.0.0",
                   help="bind address; 127.0.0.1 disables LAN access")
    s.add_argument("--no-meili", action="store_true",
                   help="do not start the search engine")
    s.set_defaults(fn=cmd_serve)
    p = sub.add_parser("setup", help="fetch the search binary, generate keys")
    p.set_defaults(fn=cmd_setup)
    q = sub.add_parser("status", help="what is running, what data exists")
    q.set_defaults(fn=cmd_status)
    b = sub.add_parser("bundle", help="pack the servable data (build machine)")
    b.set_defaults(fn=cmd_bundle)
    f = sub.add_parser("fetch-data", help="unpack a data bundle into this install")
    f.add_argument("--file", default=None)
    f.add_argument("--url", default=None)
    f.set_defaults(fn=cmd_fetch)
    m = sub.add_parser("mcp", help="MCP server on stdio — connect a coding agent")
    m.set_defaults(fn=cmd_mcp)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
