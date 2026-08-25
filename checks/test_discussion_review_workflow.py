from pathlib import Path

import yaml


WORKFLOW = Path(__file__).parent.parent / ".github/workflows/discussion-review.yml"
SETUP_UV_SHA = "38f3f104447c67c051c4a08e39b64a148898af3a"


def parsed_steps() -> list[dict]:
    workflow = yaml.load(WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    return workflow["jobs"]["review"]["steps"]


def test_edit_lookup_paginates_all_discussion_comments_for_bot_marker():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    lookup = workflow.split("EXISTING_COMMENT_ID=$(gh api graphql", 1)[1].split(
        "if [ -n \"$EXISTING_COMMENT_ID\" ]", 1
    )[0]

    assert "$endCursor: String" in lookup
    assert "comments(first: 100, after: $endCursor)" in lookup
    assert "pageInfo { hasNextPage endCursor }" in lookup
    assert "--paginate --slurp" in lookup
    assert "[.[].data.repository.discussion.comments.nodes[]" in lookup
    assert 'select(.body | contains("<!-- rubric-review-bot -->"))' in lookup


def test_review_workflow_pins_actions_and_drops_checkout_credentials():
    steps = parsed_steps()
    checkouts = [
        step for step in steps if step.get("uses", "").startswith("actions/checkout@")
    ]
    setup_uv = next(step for step in steps if step.get("name") == "Install uv")

    assert len(checkouts) == 2
    assert [
        checkout.get("with", {}).get("persist-credentials") for checkout in checkouts
    ] == ["false", "false"]
    assert setup_uv["uses"] == f"astral-sh/setup-uv@{SETUP_UV_SHA}"


def test_review_workflow_passes_github_expressions_through_step_environment():
    for step in parsed_steps():
        assert "${{" not in step.get("run", "")


def test_review_workflow_binds_zizmor_flagged_values_as_step_environment():
    steps = {step.get("name"): step for step in parsed_steps()}
    react = steps["React with eyes"]
    format_comment = steps["Format and post or update comment"]

    assert react["env"]["DISCUSSION_NODE_ID"] == "${{ github.event.discussion.node_id }}"
    assert 'id="$DISCUSSION_NODE_ID"' in react["run"]
    assert format_comment["env"]["DECISION"] == "${{ steps.review.outputs.decision }}"
    assert format_comment["env"].get("REPOSITORY_OWNER") == "${{ github.repository_owner }}"
    assert format_comment["env"].get("REPOSITORY_NAME") == "${{ github.event.repository.name }}"
    assert 'owner="$REPOSITORY_OWNER"' in format_comment["run"]
    assert 'repo="$REPOSITORY_NAME"' in format_comment["run"]
