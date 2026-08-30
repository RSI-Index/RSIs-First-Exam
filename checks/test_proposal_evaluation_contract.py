from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROPOSAL_AGENT = ROOT / ".agents" / "skills" / "proposal-agent"
SKILL = PROPOSAL_AGENT / "SKILL.md"
TEMPLATE = PROPOSAL_AGENT / "references" / "proposal-template.md"
RUBRIC = PROPOSAL_AGENT / "references" / "task-proposal-rubric.md"
README = ROOT / "README.md"


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


def test_round_six_routes_unresolved_task_contract_decisions_before_rendering():
    skill = " ".join(SKILL.read_text(encoding="utf-8").lower().split())

    assert "simulate handing the complete proposal directly" in skill
    assert "harbor-task-agent" in skill
    for contract_area in (
        "research question",
        "baseline",
        "evaluation or reward",
        "action space",
        "data boundary",
        "network boundary",
        "compute contract",
    ):
        assert contract_area in skill
    assert "route the gap back to its owning round" in skill
    assert "must not render or write the proposal" in skill
    assert "without a separate assumption confirmation" in skill
    assert "repository paths" in skill
    assert "implementation details remain agent-owned" in skill


def test_existing_template_rows_close_the_default_task_generation_contract():
    skill = " ".join(SKILL.read_text(encoding="utf-8").lower().split())
    template = " ".join(TEMPLATE.read_text(encoding="utf-8").lower().split())

    generation_defaults = (
        "candidate-producing compute runs in work",
        "judge reloads and evaluates the complete materialized candidate snapshot",
        "candidate-only evaluation",
        "solution materializes the traceable baseline without training or evaluation",
        "candidate-invalid artifacts are unscored",
        "shared base/work/judge snapshot model",
    )
    for required_default in generation_defaults:
        assert required_default in skill
        assert required_default in template

    for required_default in (
        "web search separately from each external service",
        "additional-data access",
        "concrete existing delivery/access interface",
    ):
        assert required_default in template


def test_reference_rubric_rejects_failed_task_generation_readiness():
    rubric = " ".join(RUBRIC.read_text(encoding="utf-8").lower().split())

    assert "evaluate all nine gates before deciding" in rubric
    assert "### 9. task-generation readiness" in rubric
    assert "task-generation readiness | pass / fail" in rubric
    assert "a task-generation readiness failure forces `reject`" in rubric
    assert "at least one of the nine proposal gates fails" in rubric


def test_readme_describes_one_command_task_creation_with_optional_answers():
    readme = README.read_text(encoding="utf-8")
    normalized = " ".join(readme.lower().split())

    assert "about 1 hour" in normalized
    assert "send `/task` once" in normalized
    assert "genuine task-defining question" in normalized
    assert "`/task <answer or guidance>`" in readme
    assert "automatically continues" in normalized
    assert "`/task confirm`" not in readme

    without_gpu = readme.split("**Without GPUs**", 1)[1].split("**With GPUs**", 1)[0]
    with_gpu = readme.split("**With GPUs**", 1)[1].split("### 4.", 1)[0]
    assert "private task repository" in without_gpu.lower()
    assert "clone" in without_gpu.lower()
    assert "validator" not in without_gpu.lower()
    assert "trajectory" not in without_gpu.lower()
    assert "clone" in with_gpu.lower()
    assert "harbor-task-validator" in with_gpu
