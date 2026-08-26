from pathlib import Path

import yaml

WORKFLOW = Path(__file__).parent.parent / ".github/workflows/discussion-review.yml"
ROOT = WORKFLOW.parent.parent.parent
CHECKOUT_SHA = "fbc6f3992d24b796d5a048ff273f7fcc4a7b6c09"
APP_TOKEN_SHA = "bcd2ba49218906704ab6c1aa796996da409d3eb1"
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
    assert "author { login }" in lookup
    assert 'select(.author.login == $bot)' in lookup
    assert '--arg bot "$BOT_LOGIN"' in lookup
    assert 'select(.body | contains("<!-- rubric-review-bot -->"))' in lookup


def test_review_workflow_pins_actions_and_drops_checkout_credentials():
    steps = parsed_steps()
    checkouts = [
        step for step in steps if step.get("uses", "").startswith("actions/checkout@")
    ]
    setup_uv = next(step for step in steps if step.get("name") == "Install uv")

    assert len(checkouts) == 2
    assert [checkout["uses"] for checkout in checkouts] == [
        f"actions/checkout@{CHECKOUT_SHA}",
        f"actions/checkout@{CHECKOUT_SHA}",
    ]
    assert [
        checkout.get("with", {}).get("persist-credentials") for checkout in checkouts
    ] == ["false", "false"]
    assert setup_uv["uses"] == f"astral-sh/setup-uv@{SETUP_UV_SHA}"


def test_review_workflow_reads_the_private_skills_rubric_with_an_app_token():
    steps = {step.get("name"): step for step in parsed_steps()}

    assert steps["Create private Skills read token"] == {
        "name": "Create private Skills read token",
        "id": "skills-token",
        "uses": f"actions/create-github-app-token@{APP_TOKEN_SHA}",
        "with": {
            "client-id": "${{ vars.RSI_DISPATCH_APP_CLIENT_ID }}",
            "private-key": "${{ secrets.RSI_DISPATCH_APP_PRIVATE_KEY }}",
            "owner": "RSI-Index",
            "repositories": "RSI-Skills",
            "permission-contents": "read",
        },
    }
    assert steps["Checkout private discussion rubric"] == {
        "name": "Checkout private discussion rubric",
        "uses": f"actions/checkout@{CHECKOUT_SHA}",
        "with": {
            "repository": "RSI-Index/RSI-Skills",
            "ref": "main",
            "token": "${{ steps.skills-token.outputs.token }}",
            "path": "private-skills",
            "sparse-checkout": "rubrics/task-proposal.md",
            "sparse-checkout-cone-mode": "false",
            "persist-credentials": "false",
        },
    }
    assert steps["Run rubric review"]["env"]["RUBRIC_FILE"] == (
        "${{ github.workspace }}/private-skills/rubrics/task-proposal.md"
    )


def test_review_reactions_and_comments_use_a_repository_scoped_app_token():
    steps = {step.get("name"): step for step in parsed_steps()}

    assert steps["Create public Discussion App token"] == {
        "name": "Create public Discussion App token",
        "id": "discussion-token",
        "uses": f"actions/create-github-app-token@{APP_TOKEN_SHA}",
        "with": {
            "client-id": "${{ vars.RSI_DISPATCH_APP_CLIENT_ID }}",
            "private-key": "${{ secrets.RSI_DISPATCH_APP_PRIVATE_KEY }}",
            "owner": "RSI-Index",
            "repositories": "RSI-Index-Public",
            "permission-discussions": "write",
        },
    }
    assert steps["React with eyes"]["env"]["GH_TOKEN"] == (
        "${{ steps.discussion-token.outputs.token }}"
    )
    assert steps["Format and post or update comment"]["env"]["GH_TOKEN"] == (
        "${{ steps.discussion-token.outputs.token }}"
    )
    assert steps["Format and post or update comment"]["env"]["BOT_LOGIN"] == (
        "${{ format('{0}[bot]', steps.discussion-token.outputs.app-slug) }}"
    )
    assert "DISCUSSION_TOKEN" not in WORKFLOW.read_text(encoding="utf-8")


def test_public_repository_does_not_ship_or_migrate_the_private_rubric():
    workflows = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((ROOT / ".github/workflows").glob("*.yml"))
    )

    assert not (ROOT / "rubrics/task-proposal.md").exists()
    assert not (ROOT / ".github/workflows/migrate-private-rubric.yml").exists()
    assert "Zhuofeng-Li/RSI-Index-Rubrics" not in workflows
    assert "RUBRIC_REPO_TOKEN" not in workflows


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


def test_review_workflow_renders_and_appends_the_canonical_marker():
    steps = {step.get("name"): step for step in parsed_steps()}
    render_marker = steps["Render canonical proposal review marker"]
    format_comment = steps["Format and post or update comment"]

    assert render_marker["env"] == {
        "DECISION": "${{ steps.review.outputs.decision }}",
        "DISCUSSION_TITLE": "${{ github.event.discussion.title }}",
        "DISCUSSION_BODY": "${{ github.event.discussion.body }}",
        "DISCUSSION_NODE_ID": "${{ github.event.discussion.node_id }}",
    }
    assert render_marker["run"].strip() == (
        "marker=$(python3 checks/proposal_review_marker.py)\n"
        "printf 'marker=%s\\n' \"$marker\" >> \"$GITHUB_OUTPUT\""
    )
    assert format_comment["env"]["REVIEW_MARKER"] == "${{ steps.marker.outputs.marker }}"

    comment_template = format_comment["run"].split("BODY=$(cat << COMMENT_EOF", 1)[1].split(
        "COMMENT_EOF", 1
    )[0]
    assert "${REVIEW_MARKER}" in comment_template
    assert format_comment["run"].count('-f body="$BODY"') == 2


def test_review_workflow_replaces_withheld_output_before_publication():
    steps = {step.get("name"): step for step in parsed_steps()}
    run_review = steps["Run rubric review"]["run"]

    assert 'publication_guard = result.get("publication_guard")' in run_review
    assert 'if publication_guard == "withheld":' in run_review
    assert 'decision = "require human review"' in run_review
    assert "Automated review output was withheld" in run_review
    assert 'elif publication_guard != "pass":' in run_review
