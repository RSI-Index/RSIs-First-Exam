from pathlib import Path


ROOT = Path(__file__).parent.parent
TEMPLATE = ROOT / ".agents/skills/proposal-agent/references/proposal-template.md"
SKILL = ROOT / ".agents/skills/proposal-agent/SKILL.md"
RESEARCH = ROOT / ".agents/skills/proposal-agent/references/repository-research.md"
RUBRICS = (
    ROOT / ".agents/skills/proposal-agent/references/task-proposal-rubric.md",
    ROOT / "rubrics/task-proposal.md",
)
WORKER = ROOT / "tools/rubric-review-service/src/index.ts"
REVIEW_DOC = ROOT / "docs/TASK_PROPOSAL_RUBRIC_REVIEW.md"
SERVICE_README = ROOT / "tools/rubric-review-service/README.md"


def test_proposal_contract_collects_contributor_full_name():
    template = TEMPLATE.read_text()
    skill = SKILL.read_text()

    assert "| Contributor | Full name |" in template
    assert "contributor's full name" in skill.lower()


def test_research_contract_keeps_frontier_evidence_non_blocking():
    research = RESEARCH.read_text().lower()
    frontier = research.split("## frontier evidence", 1)[1].split("\n## ", 1)[0]

    assert "public expertise evidence" in research
    assert "rolling six-month window" in frontier
    assert "preference" in frontier
    assert "x topic activity" in frontier
    assert "at least three" not in frontier
    assert "two independent" not in frontier
    assert "hard gate" not in frontier


def test_both_rubric_copies_keep_expertise_gate_and_non_blocking_frontier_review():
    for rubric_path in RUBRICS:
        rubric = rubric_path.read_text().lower()
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
