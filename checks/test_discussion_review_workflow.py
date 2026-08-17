from __future__ import annotations

from pathlib import Path

WORKFLOW = (
    Path(__file__).parent.parent / ".github" / "workflows" / "discussion-review.yml"
)
SCRIPT_LOCK = Path(__file__).parent / "rubric_review.py.lock"


def test_discussion_review_uses_persistent_coding_plan_runner():
    workflow = WORKFLOW.read_text()

    assert "runs-on: [self-hosted, linux, x64, rsi-proposal-review]" in workflow
    assert "CODEX_HOME: /var/lib/rsi-codex-judge" in workflow
    assert "CODEX_JUDGE_TMPDIR: /var/tmp/rsi-codex-judge" in workflow
    assert "for REQUIRED_COMMAND in codex gh python3 uv" in workflow
    assert 'command -v "$REQUIRED_COMMAND"' in workflow
    assert "REQUIRED_CODEX_VERSION=0.147.0" in workflow
    assert '"$CODEX_VERSION" != "$REQUIRED_CODEX_VERSION"' in workflow
    assert "REQUIRED_UV_VERSION=0.6.5" in workflow
    assert '"$UV_VERSION" != "$REQUIRED_UV_VERSION"' in workflow
    assert '"$CODEX_HOME/auth.json"' in workflow
    assert "persist-credentials: false" in workflow
    assert "actions/checkout@v4" not in workflow
    assert "astral-sh/setup-uv@v4" not in workflow
    assert "OPENAI_API_KEY" not in workflow
    assert "cancel-in-progress: true" in workflow
    assert "uv run --locked checks/rubric_review.py" in workflow

    lockfile = SCRIPT_LOCK.read_text()
    assert 'name = "httpx"' in lockfile
    assert "sha256:" in lockfile


def test_reviewer_documentation_uses_coding_plan_login():
    documentation = (
        Path(__file__).parent.parent / "docs" / "TASK_PROPOSAL_RUBRIC_REVIEW.md"
    ).read_text()

    assert "OpenAI Responses API" not in documentation
    assert "export OPENAI_API_KEY" not in documentation
    assert "Codex CLI" in documentation
    assert "CODEX_HOME" in documentation
