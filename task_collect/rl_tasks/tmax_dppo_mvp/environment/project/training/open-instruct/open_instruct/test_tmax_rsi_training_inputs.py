from __future__ import annotations

import socket
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from open_instruct.tmax_rsi.runtime_ports import reserve_vllm_server_socket
from open_instruct.tmax_rsi.training_inputs import (
    apply_training_reward_shaping,
    load_training_inputs,
    resolve_scheduler_horizon,
)


def test_vllm_server_socket_is_reserved_inside_attempt_port_block() -> None:
    busy = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    busy.bind(("127.0.0.1", 0))
    base = busy.getsockname()[1]
    if base >= 65534:
        busy.close()
        pytest.skip("ephemeral port is too close to the end of the TCP range")

    with patch.dict(
        "os.environ",
        {"TMAX_VLLM_PORT_BASE": str(base), "TMAX_VLLM_PORT_COUNT": "2"},
    ):
        reserved = reserve_vllm_server_socket()
    try:
        assert reserved.getsockname() == ("127.0.0.1", base + 1)
        contender = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            with pytest.raises(OSError):
                contender.bind(("127.0.0.1", base + 1))
        finally:
            contender.close()
    finally:
        reserved.close()
        busy.close()


def test_vllm_api_server_keeps_reserved_socket_through_uvicorn_startup() -> None:
    source = (Path(__file__).parent / "vllm_utils.py").read_text()

    assert "reserve_vllm_server_socket" in source
    assert "serve(sockets=[server_socket])" in source


def test_short_run_preserves_two_hundred_step_scheduler_horizon() -> None:
    assert resolve_scheduler_horizon(run_steps=7, requested_horizon=200) == 200
    assert resolve_scheduler_horizon(run_steps=200, requested_horizon=None) == 200
    with pytest.raises(ValueError, match="smaller than run_steps"):
        resolve_scheduler_horizon(run_steps=20, requested_horizon=10)


def test_grpo_config_source_exposes_scheduler_horizon_and_training_reward_paths() -> None:
    source = (Path(__file__).parent / "grpo_utils.py").read_text()
    fast_source = (Path(__file__).parent / "grpo_fast.py").read_text()

    assert "scheduler_horizon_steps: int | None = None" in source
    assert "training_reward_file: str | None = None" in source
    assert 'training_artifact_root: str = "/app/training"' in source
    assert fast_source.count(
        "resolve_scheduler_horizon(args.num_training_steps, args.scheduler_horizon_steps)"
    ) >= 3
    assert "training_reward_file=args.training_reward_file" in fast_source
    assert "training_artifact_root=args.training_artifact_root" in fast_source


def test_training_inputs_must_stay_below_training_artifact_root() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "training"
        data = root / "data"
        prompt = root / "prompt" / "system.txt"
        reward = root / "reward" / "reward.py"
        data.mkdir(parents=True)
        prompt.parent.mkdir(parents=True)
        reward.parent.mkdir(parents=True)
        (data / "mix.json").write_text('{"public": 1.0}\n')
        prompt.write_text("training prompt\n")
        reward.write_text("def shape_rewards(context):\n    return context['base_rewards']\n")

        inputs = load_training_inputs(
            artifact_root=root,
            data_dir=data,
            prompt_file=prompt,
            reward_file=reward,
        )

        assert inputs.data_dir == data.resolve()
        assert inputs.prompt_file == prompt.resolve()
        assert inputs.reward_file == reward.resolve()
        assert all(len(value) == 64 for value in inputs.hashes.values())

        outside = Path(temporary) / "outside.txt"
        outside.write_text("outside\n")
        with pytest.raises(ValueError, match="below training artifact root"):
            load_training_inputs(root, data, outside, reward)


def test_training_input_symlinks_are_rejected() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "training"
        data = root / "data"
        prompt = root / "prompt" / "system.txt"
        reward = root / "reward" / "reward.py"
        data.mkdir(parents=True)
        prompt.parent.mkdir(parents=True)
        reward.parent.mkdir(parents=True)
        target = root / "target.txt"
        target.write_text("prompt\n")
        prompt.symlink_to(target)
        reward.write_text("def shape_rewards(context):\n    return context['base_rewards']\n")

        with pytest.raises(ValueError, match="symlink"):
            load_training_inputs(root, data, prompt, reward)


def test_training_reward_hook_receives_public_rollout_state_and_must_return_finite_values() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "training"
        root.mkdir(parents=True)
        reward = root / "reward.py"
        reward.write_text(
            "def shape_rewards(context):\n"
            "    return [base + 0.1 * state.get('valid_tool_calls', 0) "
            "for base, state in zip(context['base_rewards'], context['rollout_states'])]\n"
        )

        shaped = apply_training_reward_shaping(
            reward_file=reward,
            artifact_root=root,
            base_rewards=[0.0, 1.0],
            decoded_responses=["a", "b"],
            finish_reasons=["stop", "stop"],
            rollout_states=[{"valid_tool_calls": 2}, {"valid_tool_calls": 0}],
        )

        assert shaped == pytest.approx([0.2, 1.0])

        reward.write_text("def shape_rewards(context):\n    return [float('nan')] * len(context['base_rewards'])\n")
        with pytest.raises(ValueError, match="finite"):
            apply_training_reward_shaping(
                reward_file=reward,
                artifact_root=root,
                base_rewards=[0.0],
                decoded_responses=["a"],
                finish_reasons=["stop"],
                rollout_states=[{}],
            )


def test_public_reward_pipeline_calls_training_hook_after_base_reward() -> None:
    source = (Path(__file__).parent / "ground_truth_utils.py").read_text()

    assert "from pathlib import Path" in source
    assert "training_reward_file: str | None = None" in source
    assert "apply_training_reward_shaping(" in source
    assert 'metrics["objective/training_reward_shaped"]' in source
