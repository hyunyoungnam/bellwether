# Audit suite

Four suites, one command:

```bash
scripts/audit/run.sh                      # against http://127.0.0.1:8001
```

| Suite | What it covers | Form |
|---|---|---|
| `api_audit.py` | every HTTP route (including bad input), the Meilisearch proxy, all 10 MCP tools, and the verifier's accept/reject behaviour | prints PASS/FAIL, exits non-zero on any failure |
| `ui_audit.html` | the chat shell: language, theme, agent choice, conversations, cite chips, copy, address bar, mobile rail | drives the real page in an iframe |
| `browse_audit.html` | `/browse`: landing, search, the cite-chip deep link, the back button | same |
| `live_audit.html` | two real agent runs — one end to end, one stopped — and cleans up after itself | same; takes ~2 minutes |

The three HTML suites are same-origin harnesses: they script the real page
inside an iframe and assert on what a reader would see. `firefox --screenshot`
fires on the load event, so each harness holds that event open with
`/slow?s=N` until its run finishes — **the screenshot is the report**
(`reports/.preview/audit-*.png`).

Reading module state from the harness needs `window.BW` (chat.html); `let`
bindings are not window properties, so there is no other way to assert on
`LANG`, `BUSY`, `CHAT`, or `PAPERS`.
