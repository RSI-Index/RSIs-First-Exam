from pathlib import Path


ROOT = Path(__file__).parent.parent
RUBRIC = ROOT / "rubrics/task-proposal.md"
WORKER = ROOT / "tools/rubric-review-service/src/index.ts"
REVIEW_DOC = ROOT / "docs/TASK_PROPOSAL_RUBRIC_REVIEW.md"
SERVICE_README = ROOT / "tools/rubric-review-service/README.md"


def test_public_rubric_keeps_expertise_gate_and_non_blocking_frontier_review():
    rubric = RUBRIC.read_text().lower()

    assert "contributor expertise alignment" in rubric
    assert "eight proposal gates" in rubric
    assert "current frontier relevance" in rubric
    assert "frontier evidence is non-blocking" in rubric
    assert "### 2. frontier activity" not in rubric
    assert "at least three" not in rubric
    assert "two independent" not in rubric
    assert "rolling six-month" in rubric
    assert "web search" in rubric
    assert "require human review" in rubric
    assert "| gate | status | evidence |" in rubric
    hard_gate_table = rubric.split("hard gate review:", 1)[1]
    assert "| frontier activity |" not in hard_gate_table

def test_worker_enables_high_context_web_search_and_supplies_review_date():
    worker = WORKER.read_text()

    assert 'type: "web_search"' in worker
    assert 'search_context_size: "high"' in worker
    assert 'tool_choice: "required"' in worker
    assert "review_date_utc" in worker


def test_review_docs_disclose_web_search_and_required_identity_field():
    for path in (REVIEW_DOC, SERVICE_README):
        documentation = path.read_text().lower()
        assert "web search" in documentation
        assert "contributor full name" in documentation
