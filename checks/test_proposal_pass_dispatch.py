from __future__ import annotations

import json
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).with_name("proposal_pass_dispatch.py")
PROPOSAL_SHA256 = "60fa597b0bea79505b8ff9d3a8602bea53ed5cb301d5fe2379c54354faec013f"


@pytest.fixture
def dispatch():
    spec = importlib.util.spec_from_file_location("proposal_pass_dispatch", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def proposal_event(*, action: str = "created") -> dict:
    return {
        "action": action,
        "repository": {"full_name": "RSI-Index/RSI-Index-Public"},
        "discussion": {
            "number": 128,
            "node_id": "D_kw128",
            "updated_at": "2026-08-30T09:59:00Z",
            "title": "Train a bounded reasoning model",
            "body": "Proposal body with private-looking review content.",
            "category": {"name": "Task Ideas"},
            "user": {"id": 42, "login": "proposal-author"},
        },
        "sender": {"id": 42, "login": "proposal-author"},
        "installation": {"id": 777},
    }


def review_body(
    *,
    status: str = "completed",
    decision: str = "Pass",
    proposal_sha256: str = PROPOSAL_SHA256,
    schema: int = 2,
) -> str:
    return (
        "<!-- rubric-review-bot -->\n"
        f"<!-- rubric-review-status:{status} -->\n"
        '<!-- rsi-proposal-review:{"decision":"'
        + decision
        + '","discussion_node_id":"D_kw128","proposal_sha256":"'
        + proposal_sha256
        + f'","schema":{schema}}} -->'
    )


def live_authority(*, comments: list[dict] | None = None) -> list[dict]:
    if comments is None:
        comments = [
            {
                "id": "DC_pass",
                "body": review_body(),
                "author": {"login": "rsi-review-app"},
                "viewerDidAuthor": True,
                "createdAt": "2026-08-30T10:00:00Z",
                "updatedAt": "2026-08-30T10:01:00Z",
            }
        ]
    return [
        {
            "data": {
                "viewer": {"login": "rsi-review-app"},
                "repository": {
                    "nameWithOwner": "RSI-Index/RSI-Index-Public",
                    "discussion": {
                        "id": "D_kw128",
                        "number": 128,
                        "lastEditedAt": None,
                        "title": "Train a bounded reasoning model",
                        "body": "Proposal body with private-looking review content.",
                        "category": {"name": "Task Ideas"},
                        "comments": {
                            "nodes": comments,
                            "pageInfo": {
                                "hasNextPage": False,
                                "endCursor": None,
                            },
                        },
                    },
                },
            }
        }
    ]


@pytest.mark.parametrize("action", ["created", "edited"])
def test_builds_exact_identifier_only_payload_for_reviewed_discussion(dispatch, action):
    event = proposal_event(action=action)

    payload = dispatch.build_proposal_pass_payload(event, "DC_pass")

    assert payload == {
        "source_repository": "RSI-Index/RSI-Index-Public",
        "discussion_number": 128,
        "discussion_node_id": "D_kw128",
        "triggering_comment_node_id": "DC_pass",
        "trigger_kind": "proposal_pass",
    }
    serialized = json.dumps(payload, sort_keys=True)
    for forbidden in (
        event["discussion"]["title"],
        event["discussion"]["body"],
        event["discussion"]["user"]["login"],
        "review content",
        "rubric text",
        json.dumps(event, sort_keys=True),
    ):
        assert forbidden not in serialized


def test_verified_live_current_completed_pass_builds_one_identifier_only_payload(dispatch):
    payload = dispatch.build_verified_proposal_pass_payload(
        proposal_event(), "DC_pass", live_authority()
    )

    assert payload == {
        "source_repository": "RSI-Index/RSI-Index-Public",
        "discussion_number": 128,
        "discussion_node_id": "D_kw128",
        "triggering_comment_node_id": "DC_pass",
        "trigger_kind": "proposal_pass",
    }


def test_verified_live_current_edited_pass_binds_the_edit_timestamp(dispatch):
    event = proposal_event(action="edited")
    live = live_authority()
    live[0]["data"]["repository"]["discussion"]["lastEditedAt"] = event[
        "discussion"
    ]["updated_at"]

    assert dispatch.build_verified_proposal_pass_payload(event, "DC_pass", live) is not None


def test_verified_pass_rejects_completed_old_run_when_newer_progress_exists(dispatch):
    comments = live_authority()[0]["data"]["repository"]["discussion"]["comments"][
        "nodes"
    ]
    comments[0]["updatedAt"] = "2026-08-30T10:03:00Z"
    comments.append(
        {
            "id": "DC_new_run",
            "body": (
                "<!-- rubric-review-bot -->\n"
                "<!-- rubric-review-status:running -->"
            ),
            "author": {"login": "rsi-review-app"},
            "viewerDidAuthor": True,
            "createdAt": "2026-08-30T10:02:00Z",
            "updatedAt": "2026-08-30T10:02:00Z",
        }
    )

    assert (
        dispatch.build_verified_proposal_pass_payload(
            proposal_event(), "DC_pass", live_authority(comments=comments)
        )
        is None
    )


def test_verified_pass_orders_fractional_rfc3339_generation_timestamps(dispatch):
    comments = live_authority()[0]["data"]["repository"]["discussion"]["comments"][
        "nodes"
    ]
    comments[0]["createdAt"] = "2026-08-30T10:02:00Z"
    comments.append(
        {
            "id": "DC_new_run",
            "body": (
                "<!-- rubric-review-bot -->\n"
                "<!-- rubric-review-status:running -->"
            ),
            "author": {"login": "rsi-review-app"},
            "viewerDidAuthor": True,
            "createdAt": "2026-08-30T10:02:00.500Z",
            "updatedAt": "2026-08-30T10:02:00.500Z",
        }
    )

    assert (
        dispatch.build_verified_proposal_pass_payload(
            proposal_event(), "DC_pass", live_authority(comments=comments)
        )
        is None
    )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda live: live[0]["data"]["repository"]["discussion"].update(
            body="Edited proposal body"
        ),
        lambda live: live[0]["data"]["repository"]["discussion"].update(
            lastEditedAt="2026-08-30T10:02:00Z"
        ),
        lambda live: live[0]["data"]["repository"]["discussion"]["comments"][
            "nodes"
        ][0].update(body=review_body(status="superseded")),
        lambda live: live[0]["data"]["repository"]["discussion"]["comments"][
            "nodes"
        ][0].update(body=review_body(proposal_sha256="f" * 64)),
        lambda live: live[0]["data"]["repository"]["discussion"]["comments"][
            "nodes"
        ][0].update(body=review_body(decision="Reject")),
        lambda live: live[0]["data"]["repository"]["discussion"]["comments"][
            "nodes"
        ][0]["author"].update(login="forged-review-app"),
        lambda live: live[0]["data"]["repository"]["discussion"]["comments"][
            "nodes"
        ][0].update(viewerDidAuthor=False),
    ],
)
def test_verified_pass_fails_closed_for_noncurrent_live_authority(dispatch, mutate):
    live = live_authority()
    mutate(live)

    assert (
        dispatch.build_verified_proposal_pass_payload(
            proposal_event(), "DC_pass", live
        )
        is None
    )


def test_verified_pass_rejects_wrong_review_comment_id(dispatch):
    assert (
        dispatch.build_verified_proposal_pass_payload(
            proposal_event(), "DC_other", live_authority()
        )
        is None
    )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda event: event.update(action="deleted"),
        lambda event: event["repository"].update(full_name="other/repository"),
        lambda event: event["discussion"]["category"].update(name="General"),
        lambda event: event["discussion"].pop("number"),
        lambda event: event["discussion"].update(number="128"),
        lambda event: event["discussion"].update(number=True),
        lambda event: event["discussion"].update(number=0),
        lambda event: event["discussion"].update(node_id="D bad"),
        lambda event: event["discussion"].update(node_id=""),
    ],
)
def test_rejects_wrong_incomplete_or_malformed_discussion_event(dispatch, mutate):
    event = proposal_event()
    mutate(event)

    assert dispatch.build_proposal_pass_payload(event, "DC_pass") is None


@pytest.mark.parametrize("comment_id", [None, 1, True, [], {}, "", "DC bad", "DC/pass"])
def test_rejects_non_string_or_malformed_review_comment_node_id(dispatch, comment_id):
    assert dispatch.build_proposal_pass_payload(proposal_event(), comment_id) is None


@pytest.mark.parametrize(
    "event",
    [None, [], {}, {"action": "created"}, {"action": "created", "repository": []}],
)
def test_malformed_untrusted_event_shapes_return_none_without_raising(dispatch, event):
    assert dispatch.build_proposal_pass_payload(event, "DC_pass") is None


def test_cli_atomically_writes_compact_sorted_payload(tmp_path):
    event_path = tmp_path / "event.json"
    live_path = tmp_path / "live.json"
    output_path = tmp_path / "dispatch.json"
    event_path.write_text(json.dumps(proposal_event()), encoding="utf-8")
    live_path.write_text(json.dumps(live_authority()), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(event_path),
            "DC_pass",
            str(live_path),
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "GITHUB_OUTPUT": ""},
    )

    assert result.stdout == "true\n"
    assert result.stderr == ""
    assert output_path.read_text(encoding="utf-8") == (
        '{"discussion_node_id":"D_kw128","discussion_number":128,'
        '"source_repository":"RSI-Index/RSI-Index-Public",'
        '"trigger_kind":"proposal_pass",'
        '"triggering_comment_node_id":"DC_pass"}'
    )
    assert not list(tmp_path.glob(".dispatch.json.*"))


def test_cli_writes_null_for_rejected_event(tmp_path):
    event_path = tmp_path / "event.json"
    live_path = tmp_path / "live.json"
    output_path = tmp_path / "dispatch.json"
    event = proposal_event()
    event["discussion"]["category"]["name"] = "General"
    event_path.write_text(json.dumps(event), encoding="utf-8")
    live_path.write_text(json.dumps(live_authority()), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(event_path),
            "DC_pass",
            str(live_path),
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout == "false\n"
    assert result.stderr == ""
    assert output_path.read_text(encoding="utf-8") == "null"
