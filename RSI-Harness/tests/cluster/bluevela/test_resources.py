from __future__ import annotations

from pathlib import Path

import pytest

from rsi_harness.cluster.bluevela.adapter import derive_resources
from rsi_harness.cluster.config import load_cluster_profile
from rsi_harness.errors import SetupError
from rsi_harness.models import CompileOptions, GPURequirement
from rsi_harness.task.compiler import HarborTaskCompiler

TARGET = (
    "linkedin__liger-kernel.c856fbab."
    "test_fused_neighborhood_attention.78217be4.lv2"
)


def _target_definition():
    root = Path(__file__).resolve().parents[3]
    return HarborTaskCompiler().compile(
        root / "sample_tasks" / TARGET,
        CompileOptions(agent_name="codex"),
    )


def test_target_resources_are_derived_from_current_task_configuration() -> None:
    resources = derive_resources(
        _target_definition(),
        load_cluster_profile("bluevela", {"USER": "alice"}),
    )

    assert resources.work_gpus == 2
    assert resources.verifier_gpus == 2
    assert resources.total_gpus == 4
    assert resources.cpu_slots == 8
    assert resources.memory_mb == 65536
    assert resources.local_tmp_mb == 15360
    assert resources.build_walltime == "02:00"
    assert resources.run_walltime == "02:15"


def test_task_build_timeout_expands_cluster_build_walltime() -> None:
    definition = _target_definition()
    definition = definition.model_copy(
        update={
            "service": definition.service.model_copy(
                update={"build_timeout_seconds": 21600.0}
            )
        }
    )

    resources = derive_resources(
        definition,
        load_cluster_profile("bluevela", {"USER": "alice"}),
    )

    assert resources.build_walltime == "06:15"


def test_task_storage_is_used_as_node_local_tmp_requirement() -> None:
    definition = _target_definition()
    definition = definition.model_copy(
        update={
            "service": definition.service.model_copy(
                update={"storage_mb": 512000}
            )
        }
    )

    resources = derive_resources(
        definition,
        load_cluster_profile("bluevela", {"USER": "alice"}),
    )

    assert resources.local_tmp_mb == 512000


def test_all_gpu_declaration_requires_numeric_profile_override() -> None:
    definition = _target_definition().model_copy(
        update={"gpu_requirement": GPURequirement(count="all")}
    )

    with pytest.raises(SetupError, match="numeric override"):
        derive_resources(
            definition,
            load_cluster_profile("bluevela", {"USER": "alice"}),
        )


@pytest.mark.parametrize("verifier_gpus", (4, 8))
def test_single_node_reuses_work_gpus_when_phase_sum_exceeds_capacity(
    verifier_gpus: int,
) -> None:
    definition = _target_definition()
    definition = definition.model_copy(
        update={
            "gpu_requirement": GPURequirement(count=8),
            "verifier": definition.verifier.model_copy(
                update={"gpu_count": verifier_gpus}
            ),
        }
    )

    resources = derive_resources(
        definition,
        load_cluster_profile("bluevela", {"USER": "alice"}),
    )

    assert resources.work_gpus == 8
    assert resources.verifier_gpus == verifier_gpus
    assert resources.total_gpus == 8


def test_phase_gpu_limit_is_enforced() -> None:
    definition = _target_definition().model_copy(
        update={"gpu_requirement": GPURequirement(count=9)}
    )

    with pytest.raises(SetupError, match="single-node capacity"):
        derive_resources(
            definition,
            load_cluster_profile("bluevela", {"USER": "alice"}),
        )
