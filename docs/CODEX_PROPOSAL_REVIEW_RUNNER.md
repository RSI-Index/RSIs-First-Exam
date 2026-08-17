# Coding Plan proposal-review runner

The Discussion rubric workflow uses a dedicated self-hosted runner and a
ChatGPT-managed Codex login. It does not use `OPENAI_API_KEY`.

## Requirements

- Use a trusted Linux x64 machine dedicated to this private repository.
- Register the runner with the `rsi-proposal-review` custom label in addition to
  GitHub's standard `self-hosted`, `linux`, and `x64` labels.
- Run the GitHub Actions service as a dedicated unprivileged account.
- Install the reviewed Codex CLI version 0.147.0 for that account. GPT-5.6 models
  cannot use a ChatGPT-managed login through older Codex clients.
- Preinstall the reviewed `uv` version 0.6.5, GitHub CLI (`gh`), and Python 3 for
  the runner account. The workflow checks these tools and versions before
  reading or posting Discussion content.
- Keep the machine online whenever Discussion reviews should run.

Do not attach this runner to a public repository or a workflow that executes
untrusted pull-request code. The proposal reviewer passes Discussion content to
a tool-disabled judge, but the runner still contains a renewable login
credential.

Restrict who can change the default branch and workflow files. If the
organization supports runner groups, limit this runner group to this repository
and only the proposal-review workflow. For stronger credential separation,
place the credential and Codex process behind a separately owned, root-managed
service with a narrow review-only interface; do not give a general-purpose
Actions account read access to that service account's home. That service boundary
is optional infrastructure and is not created by this repository.

## One-time credential setup

Create the persistent Codex home with permissions limited to the runner account.
Replace `rsi-runner` with that account's actual user and group:

```bash
sudo install -d -m 700 -o rsi-runner -g rsi-runner /var/lib/rsi-codex-judge
sudo install -d -m 700 -o rsi-runner -g rsi-runner /var/tmp/rsi-codex-judge
```

Install the reviewed CLI release and confirm the exact version:

```bash
npm install --global @openai/codex@0.147.0
codex --version
```

Install the reviewed `uv` release using the host's managed provisioning method
(for example, `pipx install uv==0.6.5`) and confirm `uv --version` reports
`0.6.5`. The committed `checks/rubric_review.py.lock` fixes all Python packages
and artifact hashes; CI uses `uv run --locked` and fails instead of re-resolving
the dependency graph.

As the runner account, start the official ChatGPT login flow with this dedicated
home:

```bash
CODEX_HOME=/var/lib/rsi-codex-judge codex login
chmod 600 /var/lib/rsi-codex-judge/auth.json
```

Complete the browser/device authorization with the Coding Plan account. Never
print, commit, upload, or store `auth.json` as a GitHub Actions artifact or
repository secret. The Codex CLI refreshes the login in place, which is why this
directory and the runner filesystem must persist across jobs.

The workflow fixes `CODEX_HOME` to `/var/lib/rsi-codex-judge`, uses
`/var/tmp/rsi-codex-judge` for ephemeral judge files, and fails before reviewing
if the CLI or private credential is missing or has unsafe permissions.

The judge invocation ignores user configuration and exec rules; disables shell,
web, browser, computer-use, code-mode, image-generation, app, plugin,
workspace-dependency, skill-discovery, and image-inspection tools; and receives
a scrubbed environment without `GITHUB_TOKEN`, `GH_TOKEN`, or API keys. Input
images still arrive as bounded initial attachments; the model cannot invoke an
image-inspection tool against arbitrary host paths.

## Runner registration

In the private repository, open **Settings → Actions → Runners → New
self-hosted runner** and follow GitHub's generated installation commands on the
dedicated machine. Add the label during configuration:

```bash
./config.sh --url https://github.com/RSI-Index/RSI-Index-Public \
  --token ONE_TIME_REGISTRATION_TOKEN \
  --labels rsi-proposal-review
```

Install and start the runner as a service using GitHub's generated service
commands. Registration tokens are short-lived; obtain a fresh one from the
repository settings instead of saving it in this repository.

## Maintenance and rotation

- Upgrade Codex CLI deliberately, pin the reviewed version in provisioning, and
  run the reviewer tests and smoke test before rollout.
- Keep `/var/lib/rsi-codex-judge` on persistent storage and backed by restrictive
  filesystem permissions.
- If authentication fails, run `codex login` again with the same `CODEX_HOME`.
- If the host or credential may have been exposed, revoke the ChatGPT session,
  remove the old `auth.json`, and perform a fresh login.
- Remove the runner from repository settings before decommissioning the host.
