"""This machine sits behind a TLS-inspecting gateway (swg.kisti.re.kr) whose
root certificate lacks an Authority Key Identifier. Python >= 3.13 turns on
VERIFY_X509_STRICT for new SSL contexts — and urllib3 >= 2.2 re-adds it after
building its own — so every client rejects that chain ("Missing Authority
Key Identifier") while curl accepts it.

Verification stays ON — trust and hostname are still checked. Only the
strict RFC 5280 extension checks are relaxed, and at the one place every
library must pass: the handshake. `SSLContext.wrap_socket` / `wrap_bio` are
wrapped to clear the flag just before wrapping, whatever built the context
(httpx/httpcore, requests/urllib3, huggingface_hub, tiktoken, anyio). The
class itself is not replaced — ssl.py's properties look it up by name and
would recurse. Import this before anything that opens a TLS connection;
budget.py does.
"""
import ssl

if not getattr(ssl, "_bellwether_relaxed", False):
    _wrap_socket, _wrap_bio = ssl.SSLContext.wrap_socket, ssl.SSLContext.wrap_bio

    def _relax(ctx):
        try:
            ctx.verify_flags &= ~ssl.VERIFY_X509_STRICT
        except (ValueError, AttributeError):
            pass

    def _wrap_socket_relaxed(self, *a, **kw):
        _relax(self)
        return _wrap_socket(self, *a, **kw)

    def _wrap_bio_relaxed(self, *a, **kw):
        _relax(self)
        return _wrap_bio(self, *a, **kw)

    ssl.SSLContext.wrap_socket = _wrap_socket_relaxed
    ssl.SSLContext.wrap_bio = _wrap_bio_relaxed

    _orig_default = ssl.create_default_context

    def _relaxed_default(*a, **kw):
        ctx = _orig_default(*a, **kw)
        _relax(ctx)
        return ctx

    ssl.create_default_context = _relaxed_default
    ssl._create_default_https_context = _relaxed_default
    ssl._bellwether_relaxed = True
