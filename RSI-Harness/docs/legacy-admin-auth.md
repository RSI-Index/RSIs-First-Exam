# Legacy Judge administrator credentials

This page covers `python -m rsi_loop serve` and the corresponding legacy
`python -m rsi_loop run` client. The current `rsi-harness run` command uses its
own per-run submission tokens and does not need this setting. GitHub Actions
credentials and separately managed remote Judge service API keys are also
independent of this credential.

For local legacy use, the server and host client automatically share
`~/.rsi-loop/admin-secret` when they run as the same effective user with the
same home directory. The first command creates a random credential in a
private directory (mode `0700`) and file (mode `0600`). Concurrent starts use
the same complete credential. Importing the package does not create or read
this file. The credential is not passed into task containers.

For a remote server, a different host user, containers, or `sudo`, explicitly
configure the same credential on the server and legitimate host clients using
**one** of these environment variables:

- `RSI_ADMIN_SECRET`: a secret supplied by your deployment's secret manager.
- `RSI_ADMIN_SECRET_FILE`: the absolute path to an existing secret file. The
  file must be owned by the process user and have mode `0600`; its containing
  directory must also be owned by that user and deny group/other access. No
  component of the path may be a symlink.

For example, if the deployment mounts an existing private credential file:

```bash
export RSI_ADMIN_SECRET_FILE=/run/rsi-secrets/legacy-judge-admin
python -m rsi_loop serve --host 127.0.0.1
# In another shell, with the same setting:
python -m rsi_loop run --task YOUR_TASK --agent YOUR_AGENT
```

Use a cryptographically random value with at least 32 printable ASCII
characters and no whitespace (maximum 4096 characters). A file may have one
trailing newline. Blank values, conflicting variables, missing explicit files,
and insecure permissions are errors; they do not trigger a fallback. Do not
put the credential in the repository, task configuration, run logs, or shell
command arguments. Use TLS for a remote Judge connection and limit access to
trusted host clients.

The server resolves the credential when the application starts; each host run
resolves it once before creating run resources. To rotate a credential, update
the managed value/file, restart the server, and start clients with the new
value. For the automatic local default, replace the private file while clients
and the server are stopped, or remove it so the next command generates a new
credential.

## Migration from the fixed credential

The former embedded credential has been removed. Treat any deployment that
used it as having an exposed credential. Upgrading repository files alone does
not invalidate a server that is already running: restart it and upgrade its
host clients together. Machines that rely on separate local defaults will
generate different credentials, so remote clients must use explicit shared
configuration.

Registration and automatic submissions retain their request-body credential
fields. Host access to full history now sends `X-RSI-Admin-Secret` as a request
header, never as a URL query parameter. Custom clients using the old
`admin_secret` query parameter must switch to the header; ordinary agent
history remains scoped by its session token. Do not configure reverse proxies
to log this header or credential-bearing request bodies.
