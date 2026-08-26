from __future__ import annotations

from pathlib import Path

import yaml


WORKFLOW = Path(__file__).parent.parent / ".github/workflows/migrate-private-rubric.yml"
CHECKOUT_SHA = "fbc6f3992d24b796d5a048ff273f7fcc4a7b6c09"
APP_TOKEN_SHA = "bcd2ba49218906704ab6c1aa796996da409d3eb1"


def load_workflow() -> tuple[dict, str]:
    raw = WORKFLOW.read_text(encoding="utf-8")
    return yaml.load(raw, Loader=yaml.BaseLoader), raw


def test_migration_is_manual_and_minimally_privileged():
    workflow, _ = load_workflow()

    assert workflow["on"] == {"workflow_dispatch": ""}
    assert workflow["permissions"] == {"contents": "read"}
    assert workflow["jobs"]["migrate"]["runs-on"] == "ubuntu-latest"


def test_migration_reads_only_the_private_rubric_and_writes_only_skills():
    workflow, _ = load_workflow()
    steps = {step["name"]: step for step in workflow["jobs"]["migrate"]["steps"]}

    assert steps["Checkout private rubric"] == {
        "name": "Checkout private rubric",
        "uses": f"actions/checkout@{CHECKOUT_SHA}",
        "with": {
            "repository": "Zhuofeng-Li/RSI-Index-Rubrics",
            "ref": "main",
            "token": "${{ secrets.RUBRIC_REPO_TOKEN }}",
            "path": "source-rubric",
            "sparse-checkout": "task-proposal.md",
            "sparse-checkout-cone-mode": "false",
            "persist-credentials": "false",
        },
    }
    assert steps["Create Skills migration token"] == {
        "name": "Create Skills migration token",
        "id": "skills-token",
        "uses": f"actions/create-github-app-token@{APP_TOKEN_SHA}",
        "with": {
            "client-id": "${{ vars.RSI_DISPATCH_APP_CLIENT_ID }}",
            "private-key": "${{ secrets.RSI_DISPATCH_APP_PRIVATE_KEY }}",
            "owner": "RSI-Index",
            "repositories": "RSI-Skills",
            "permission-contents": "write",
        },
    }
    assert steps["Checkout Skills main"] == {
        "name": "Checkout Skills main",
        "uses": f"actions/checkout@{CHECKOUT_SHA}",
        "with": {
            "repository": "RSI-Index/RSI-Skills",
            "ref": "main",
            "token": "${{ steps.skills-token.outputs.token }}",
            "path": "target-skills",
            "persist-credentials": "false",
        },
    }


def test_migration_validates_exact_bytes_without_printing_private_content():
    _, raw = load_workflow()
    run = yaml.load(raw, Loader=yaml.BaseLoader)["jobs"]["migrate"]["steps"][-1][
        "run"
    ]

    assert "test -f \"$source\"" in run
    assert "test ! -L \"$source\"" in run
    assert "1048576" in run
    assert "install -m 0644 \"$source\"" in run
    assert "cmp --silent" in run
    assert "sha256sum" in run
    assert "automation/migrate-task-proposal-rubric" in run
    assert "GIT_ASKPASS" in run
    assert 'git -C target-skills push origin "HEAD:refs/heads/$branch"' in run
    assert "--force" not in run
    assert "cat \"$source\"" not in run
    assert "RUBRIC_REPO_TOKEN" not in run
