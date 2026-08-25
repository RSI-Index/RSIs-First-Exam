from __future__ import annotations

from pathlib import Path

import yaml


WORKFLOW = Path(__file__).parent.parent / ".github/workflows/discussion-task-dispatch.yml"
CHECKOUT_SHA = "fbc6f3992d24b796d5a048ff273f7fcc4a7b6c09"
APP_TOKEN_SHA = "bcd2ba49218906704ab6c1aa796996da409d3eb1"


def load_workflow() -> tuple[dict, str]:
    raw = WORKFLOW.read_text(encoding="utf-8")
    return yaml.load(raw, Loader=yaml.BaseLoader), raw


def steps_by_name(workflow: dict) -> dict[str, dict]:
    steps = workflow["jobs"]["dispatch"]["steps"]
    return {step["name"]: step for step in steps}


def test_dispatch_workflow_accepts_only_created_discussion_comments():
    workflow, _ = load_workflow()

    assert workflow["on"] == {"discussion_comment": {"types": ["created"]}}


def test_dispatch_workflow_is_public_only_and_minimally_privileged():
    workflow, _ = load_workflow()
    steps = workflow["jobs"]["dispatch"]["steps"]
    checkout = next(step for step in steps if step.get("uses", "").startswith("actions/checkout@"))

    assert workflow["permissions"] == {"contents": "read"}
    assert workflow["jobs"]["dispatch"]["runs-on"] == "ubuntu-latest"
    assert checkout == {
        "name": "Checkout public default branch",
        "uses": f"actions/checkout@{CHECKOUT_SHA}",
        "with": {
            "ref": "${{ github.event.repository.default_branch }}",
            "persist-credentials": "false",
        },
    }
    assert len([step for step in steps if step.get("uses", "").startswith("actions/checkout@")]) == 1


def test_dispatch_workflow_gates_token_and_dispatch_on_identifier_payload():
    workflow, _ = load_workflow()
    steps = workflow["jobs"]["dispatch"]["steps"]
    named_steps = steps_by_name(workflow)
    gate_index = next(index for index, step in enumerate(steps) if step["name"] == "Validate event")
    token_index = next(index for index, step in enumerate(steps) if step["name"] == "Create dispatch App token")

    assert "checks/discussion_task_dispatch.py" in named_steps["Validate event"]["run"]
    assert gate_index < token_index
    assert named_steps["Create dispatch App token"] == {
        "name": "Create dispatch App token",
        "id": "app-token",
        "if": "steps.gate.outputs.should_dispatch == 'true'",
        "uses": f"actions/create-github-app-token@{APP_TOKEN_SHA}",
        "with": {
            "app-id": "${{ vars.RSI_DISPATCH_APP_CLIENT_ID }}",
            "private-key": "${{ secrets.RSI_DISPATCH_APP_PRIVATE_KEY }}",
            "owner": "RSI-Index",
            "repositories": "RSI-Skills",
            "permission-contents": "write",
        },
    }
    for name in ("Build dispatch request", "Dispatch privately"):
        assert named_steps[name]["if"] == "steps.gate.outputs.should_dispatch == 'true'"


def test_dispatch_workflow_posts_a_parser_built_identifier_only_request():
    workflow, _ = load_workflow()
    named_steps = steps_by_name(workflow)
    build_request = named_steps["Build dispatch request"]["run"]
    dispatch = named_steps["Dispatch privately"]

    assert "json.load" in build_request
    assert "dispatch-payload.json" in build_request
    assert "json.dump" in build_request
    assert "repository-dispatch.json" in build_request
    assert "discussion_task_command" in build_request
    assert dispatch["env"] == {"GH_TOKEN": "${{ steps.app-token.outputs.token }}"}
    assert "gh api --method POST /repos/RSI-Index/RSI-Skills/dispatches" in dispatch["run"]
    assert "--input repository-dispatch.json" in dispatch["run"]


def test_dispatch_workflow_never_handles_private_or_untrusted_content():
    _, raw = load_workflow()
    lower = raw.lower()

    for forbidden in (
        "openai/codex-action",
        "openai_api_key",
        "rsi-task-state",
        "upload-artifact",
        "github.event.comment.body",
        "github.event.discussion.body",
        "generated task",
    ):
        assert forbidden not in lower
    assert "repository:" not in lower
