"""Python >= 3.13 verifies TLS chains with VERIFY_X509_STRICT, and the chain
this machine sees for api.anthropic.com / api.semanticscholar.org fails it
("Missing Authority Key Identifier") while curl accepts it. Verification
stays ON — hostname and trust are still checked — only the strict RFC 5280
extension checks are relaxed, process-wide, before httpx/litellm build their
contexts. Import this before anything that opens a TLS connection.
"""
import ssl

_orig = ssl.create_default_context


def _relaxed(*a, **kw):
    ctx = _orig(*a, **kw)
    ctx.verify_flags &= ~ssl.VERIFY_X509_STRICT
    return ctx


if not getattr(ssl, "_bellwether_relaxed", False):
    ssl.create_default_context = _relaxed
    ssl._create_default_https_context = _relaxed
    ssl._bellwether_relaxed = True
