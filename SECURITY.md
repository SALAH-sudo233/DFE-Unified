# Security and responsible handling

This repository intentionally excludes SSH helpers, host-tunnel configuration,
passwords, tokens, private keys, credential encodings, heartbeat scripts, and
remote administration utilities used during research operations.

Before committing changes, run:

```bash
python scripts/verify_repository.py
```

Do not commit external dataset credentials, licensed dataset content, local
environment files, or server connection details. Report a suspected secret by
contacting the repository owner privately; do not open a public issue containing
the value. If a credential is committed, revoke it first, then remove it from
Git history and rotate any dependent credentials.
