# 0010. API write protection: open on localhost, token or read-only elsewhere

**Status:** Accepted (M8)

## Context
The explorer API (M7) had no authentication. That was fine for `supplygraph serve` on
127.0.0.1. A deployed server is different: its write endpoints decide sanctions matches
(`/api/actions/{id}/decision`) and spend API credit (`/api/ask`). Full user management (OIDC,
roles) is out of scope for a single-team tool.

## Decision
Writes run in one of three modes, chosen by `write_mode(host, token)`:

| Bound to | `SUPPLYGRAPH_API_TOKEN` | Mode | Writes |
|---|---|---|---|
| localhost | unset | `open` | allowed (local single-analyst use) |
| anything | set | `token` | need `Authorization: Bearer <token>` (compared in constant time) |
| non-local | unset | `off` | refused with 403: the server is read-only |

Reads are never authenticated by the app. If the data must not be public, put the server
behind network access control (ingress auth, VPN, port-forward).

- The UI reads `/api/config`. It shows a token field in `token` mode and a read-only notice in
  `off` mode.
- The token lives in browser `sessionStorage` (one tab), never `localStorage`.
- The Helm chart takes the token from a Secret.

## Alternatives considered
- **OIDC / per-user accounts.** Right for a multi-team product, too heavy here. The deciding
  analyst's name is still recorded per decision, but it is self-asserted; the token only proves
  the caller holds the shared secret.
- **Open writes behind an ingress.** That puts safety in a separate config the app can't see.
  Failing closed (`off`) is the safer default.

## Consequences
- A shared token can't tell analysts apart. If attribution must be trustworthy, supersede this
  record with per-user auth.
- Forgetting the token yields a working read-only deployment, not an open one.
