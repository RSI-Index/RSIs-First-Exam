from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROPOSAL_AGENT = ROOT / ".agents" / "skills" / "proposal-agent"
SKILL = PROPOSAL_AGENT / "SKILL.md"
TEMPLATE = PROPOSAL_AGENT / "references" / "proposal-template.md"
RUBRIC = PROPOSAL_AGENT / "references" / "task-proposal-rubric.md"


def test_proposal_agent_allows_justified_fixed_protocol_retraining():
    skill = SKILL.read_text(encoding="utf-8").lower()
    template = TEMPLATE.read_text(encoding="utf-8").lower()
    rubric = RUBRIC.read_text(encoding="utf-8").lower()
    normalized_rubric = " ".join(rubric.split())

    for source in (skill, rubric):
        body = " ".join(source.split())
        assert "prefer direct evaluation" in body
        assert "evaluation-time retraining under a fixed protocol" in body
        assert "why direct evaluation cannot answer" in body
        assert "evaluator-owned source code" in body
        assert "run budget" in body
        assert "metric capture" in body
        assert "matched baseline/candidate execution" in body
        assert "retry" in body or "retries" in body
        assert "candidate-failure" in body
        assert "fabricated outputs" in body

    assert "configuration or manifest" in template
    assert "retraining justification" in template
    assert "fixed evaluation execution contract" in template
    assert "matched baseline/candidate execution under the same evaluator-owned protocol" in normalized_rubric
    assert "except the declared candidate-versus-baseline treatment" in normalized_rubric
    assert "must not rerun the candidate's training recipe" not in rubric
