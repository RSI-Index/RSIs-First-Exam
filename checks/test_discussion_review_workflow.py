from pathlib import Path


WORKFLOW = Path(__file__).parent.parent / ".github/workflows/discussion-review.yml"


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
