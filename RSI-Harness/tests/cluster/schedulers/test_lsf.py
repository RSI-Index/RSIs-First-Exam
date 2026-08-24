from __future__ import annotations

import subprocess
from collections.abc import Sequence
from pathlib import Path

import pytest

from rsi_harness.cluster.schedulers.lsf import (
    LSFJobSpec,
    LSFScheduler,
)
from rsi_harness.errors import InfrastructureError


class FakeRunner:
    def __init__(self, responses: Sequence[subprocess.CompletedProcess[str]]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, argv: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
        self.calls.append(argv)
        return self.responses.pop(0)


def _completed(
    stdout: str = "", stderr: str = "", returncode: int = 0
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess((), returncode, stdout, stderr)


def _spec(tmp_path: Path, *, gpu_count: int = 4) -> LSFJobSpec:
    script = tmp_path / "job.sh"
    script.write_text("#!/bin/bash\nexit 0\n")
    script.chmod(0o700)
    return LSFJobSpec(
        name="mt-rsi-target-alice",
        queue="normal",
        group="grp_models",
        cpu_slots=8,
        memory_mb=65536,
        walltime="02:15",
        stdout_path=(tmp_path / "lsf.%J.out").resolve(),
        stderr_path=(tmp_path / "lsf.%J.err").resolve(),
        script_path=script.resolve(),
        gpu_count=gpu_count,
        local_tmp_mb=512000,
    )


def test_gpu_submit_argv_is_explicit_and_single_host(tmp_path: Path) -> None:
    scheduler = LSFScheduler()

    argv = scheduler.render_submit(_spec(tmp_path))

    assert argv == (
        "bsub",
        "-J",
        "mt-rsi-target-alice",
        "-G",
        "grp_models",
        "-q",
        "normal",
        "-n",
        "8",
        "-M",
        "65536M",
        "-W",
        "02:15",
        "-R",
        "select[tmp>=512000] span[hosts=1]",
        "-gpu",
        "num=4:mode=exclusive_process",
        "-o",
        str(tmp_path / "lsf.%J.out"),
        "-e",
        str(tmp_path / "lsf.%J.err"),
        str(tmp_path / "job.sh"),
    )


def test_submit_parses_exact_job_id(tmp_path: Path) -> None:
    runner = FakeRunner(
        [_completed("Job <678901> is submitted to queue <normal>.\n")]
    )
    scheduler = LSFScheduler(runner=runner)

    assert scheduler.submit(_spec(tmp_path, gpu_count=0)) == "678901"


def test_submit_rejects_non_lsf_output(tmp_path: Path) -> None:
    scheduler = LSFScheduler(runner=FakeRunner([_completed("submitted\n")]))

    with pytest.raises(InfrastructureError, match="job ID"):
        scheduler.submit(_spec(tmp_path))


def test_wait_returns_done_exit_code_after_run_state() -> None:
    states: list[str] = []
    runner = FakeRunner([_completed("RUN -\n"), _completed("DONE 0\n")])
    scheduler = LSFScheduler(runner=runner, sleeper=lambda _: None)

    result = scheduler.wait("678901", poll_seconds=0, on_state=states.append)

    assert result.job_id == "678901"
    assert result.state == "DONE"
    assert result.exit_code == 0
    assert states == ["RUN", "DONE"]


def test_interrupt_cancels_only_the_recorded_job() -> None:
    runner = FakeRunner(
        [_completed("RUN -\n"), _completed("Job <678901> is being terminated\n")]
    )

    def interrupt(_: float) -> None:
        raise KeyboardInterrupt

    scheduler = LSFScheduler(runner=runner, sleeper=interrupt)

    with pytest.raises(KeyboardInterrupt):
        scheduler.wait("678901", poll_seconds=1)

    assert runner.calls[-1] == ("bkill", "678901")


def test_exact_live_job_name_is_rejected_before_submission() -> None:
    runner = FakeRunner([_completed("other RUN\nmt-rsi-target-alice PEND\n")])
    scheduler = LSFScheduler(runner=runner)

    with pytest.raises(InfrastructureError, match="already active"):
        scheduler.require_name_available("mt-rsi-target-alice")

    assert runner.calls == [
        ("bjobs", "-noheader", "-o", "job_name stat"),
    ]
