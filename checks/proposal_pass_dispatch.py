from __future__ import annotations

import json
import os
import re
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from proposal_review_marker import canonical_proposal_hash, parse_review_marker

_SOURCE_REPOSITORY = "RSI-Index/RSI-Index-Public"
_DISCUSSION_ACTIONS = frozenset({"created", "edited"})
_NODE_ID = re.compile(r"[A-Za-z0-9_-]+\Z")
_REVIEW_BOT_MARKER = "<!-- rubric-review-bot -->"
_REVIEW_STATUS = re.compile(
    r"<!-- rubric-review-status:(running|superseded|completed|failed) -->"
)


def _node_id(value: object) -> bool:
    return isinstance(value, str) and _NODE_ID.fullmatch(value) is not None


def _timestamp(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


def build_proposal_pass_payload(
    event: object, review_comment_node_id: object
) -> dict[str, object] | None:
    """Build the identifier-only private dispatch payload for a published Pass."""
    if not isinstance(event, dict) or event.get("action") not in _DISCUSSION_ACTIONS:
        return None
    repository = event.get("repository")
    discussion = event.get("discussion")
    if not isinstance(repository, dict) or not isinstance(discussion, dict):
        return None
    if repository.get("full_name") != _SOURCE_REPOSITORY:
        return None
    category = discussion.get("category")
    if not isinstance(category, dict) or category.get("name") != "Task Ideas":
        return None

    discussion_number = discussion.get("number")
    discussion_node_id = discussion.get("node_id")
    if type(discussion_number) is not int or discussion_number <= 0:
        return None
    if not _node_id(discussion_node_id) or not _node_id(review_comment_node_id):
        return None

    return {
        "source_repository": _SOURCE_REPOSITORY,
        "discussion_number": discussion_number,
        "discussion_node_id": discussion_node_id,
        "triggering_comment_node_id": review_comment_node_id,
        "trigger_kind": "proposal_pass",
    }


def build_verified_proposal_pass_payload(
    event: object,
    review_comment_node_id: object,
    live_pages: object,
) -> dict[str, object] | None:
    """Build a payload only for the current live App-authored completed Pass."""
    payload = build_proposal_pass_payload(event, review_comment_node_id)
    if payload is None or not isinstance(event, dict):
        return None
    event_discussion = event.get("discussion")
    if not isinstance(event_discussion, dict):
        return None
    event_title = event_discussion.get("title")
    event_body = event_discussion.get("body")
    event_updated_at = event_discussion.get("updated_at")
    if (
        not isinstance(event_title, str)
        or not isinstance(event_body, str)
        or not isinstance(event_updated_at, str)
        or not event_updated_at
    ):
        return None
    expected_last_edited_at = (
        None if event.get("action") == "created" else event_updated_at
    )
    if not isinstance(live_pages, list) or not live_pages:
        return None

    viewer_login: str | None = None
    comments: list[dict] = []
    seen_comment_ids: set[str] = set()
    for page in live_pages:
        if not isinstance(page, dict):
            return None
        data = page.get("data")
        if not isinstance(data, dict):
            return None
        viewer = data.get("viewer")
        repository = data.get("repository")
        if not isinstance(viewer, dict) or not isinstance(repository, dict):
            return None
        page_viewer_login = viewer.get("login")
        if not isinstance(page_viewer_login, str) or not page_viewer_login:
            return None
        if viewer_login is None:
            viewer_login = page_viewer_login
        elif viewer_login != page_viewer_login:
            return None
        if repository.get("nameWithOwner") != _SOURCE_REPOSITORY:
            return None
        discussion = repository.get("discussion")
        if not isinstance(discussion, dict):
            return None
        category = discussion.get("category")
        if (
            discussion.get("id") != payload["discussion_node_id"]
            or discussion.get("number") != payload["discussion_number"]
            or discussion.get("title") != event_title
            or discussion.get("body") != event_body
            or discussion.get("lastEditedAt") != expected_last_edited_at
            or not isinstance(category, dict)
            or category.get("name") != "Task Ideas"
        ):
            return None
        connection = discussion.get("comments")
        if not isinstance(connection, dict):
            return None
        nodes = connection.get("nodes")
        page_info = connection.get("pageInfo")
        if not isinstance(nodes, list) or not isinstance(page_info, dict):
            return None
        for comment in nodes:
            if not isinstance(comment, dict):
                return None
            comment_id = comment.get("id")
            if not _node_id(comment_id) or comment_id in seen_comment_ids:
                return None
            seen_comment_ids.add(comment_id)
            comments.append(comment)

    final_page_info = live_pages[-1]["data"]["repository"]["discussion"]["comments"][
        "pageInfo"
    ]
    if final_page_info.get("hasNextPage") is not False:
        return None

    generations: list[tuple[datetime, str, dict, str]] = []
    for comment in comments:
        author = comment.get("author")
        body = comment.get("body")
        created_at = comment.get("createdAt")
        updated_at = comment.get("updatedAt")
        created_timestamp = _timestamp(created_at)
        updated_timestamp = _timestamp(updated_at)
        if (
            not isinstance(author, dict)
            or author.get("login") != viewer_login
            or comment.get("viewerDidAuthor") is not True
            or not isinstance(body, str)
            or _REVIEW_BOT_MARKER not in body
            or created_timestamp is None
            or updated_timestamp is None
        ):
            continue
        statuses = _REVIEW_STATUS.findall(body)
        if len(statuses) != 1:
            continue
        generations.append((created_timestamp, comment["id"], comment, statuses[0]))

    if not generations:
        return None
    _, current_id, current_comment, current_status = max(generations)
    if current_id != review_comment_node_id or current_status != "completed":
        return None
    current_body = current_comment["body"]
    marker = parse_review_marker(
        current_body,
        expected_discussion_node_id=payload["discussion_node_id"],
    )
    if (
        marker is None
        or marker.schema != 2
        or marker.decision != "Pass"
        or marker.proposal_sha256 != canonical_proposal_hash(event_title, event_body)
    ):
        return None
    return payload


def _write_json_atomically(path: Path, value: object) -> None:
    path = path.absolute()
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(
                value,
                output,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def main(argv: list[str] | None = None) -> None:
    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) != 4:
        raise SystemExit(
            "usage: proposal_pass_dispatch.py "
            "EVENT_JSON REVIEW_COMMENT_NODE_ID LIVE_JSON OUTPUT_JSON"
        )

    event_path = Path(arguments[0])
    review_comment_node_id = arguments[1]
    live_path = Path(arguments[2])
    output_path = Path(arguments[3])
    try:
        event = json.loads(event_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        event = None
    try:
        live_pages = json.loads(live_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        live_pages = None

    payload = build_verified_proposal_pass_payload(
        event, review_comment_node_id, live_pages
    )
    _write_json_atomically(output_path, payload)
    print("true" if payload is not None else "false")


if __name__ == "__main__":
    main()
