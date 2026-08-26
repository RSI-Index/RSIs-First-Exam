from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from blaunch_proxy import Broker, safe_client_environment  # noqa: E402


class BrokerSecurityTest(unittest.TestCase):
    def make_broker(self, temporary: str) -> Broker:
        root = Path(temporary)
        runner = root / "run_tmax_attempt.sh"
        blaunch = root / "blaunch"
        runner.write_text("#!/bin/sh\n")
        blaunch.write_text("#!/bin/sh\n")
        return Broker(root / "proxy", blaunch, runner, root / "full-run")

    def test_only_allocated_host_and_allowlisted_runner_are_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, patch.dict(
            os.environ,
            {"LSB_MCPU_HOSTS": "n1 8 n2 8", "LSB_JOBID": "123"},
        ):
            broker = self.make_broker(temporary)
            accepted = broker.validate_arguments(
                ["-z", "n1", "/task-tools/run_tmax_attempt.sh", "/app/output/attempts/rl-007/run_contract.json"]
            )
            self.assertEqual(accepted[2], str(broker.runner))
            with self.assertRaisesRegex(ValueError, "outside this allocation"):
                broker.validate_arguments(["-z", "other-host", "/task-tools/run_tmax_attempt.sh"])
            with self.assertRaisesRegex(ValueError, "runner"):
                broker.validate_arguments(["-z", "n1", "/bin/bash"])

    def test_client_cannot_forward_credentials_or_scheduler_identity(self) -> None:
        filtered = safe_client_environment(
            {
                "TMAX_ATTEMPT_ID": "rl-007",
                "RSI_RL_LEARNING_RATE": "1e-6",
                "OPENAI_API_KEY": "secret",
                "LSB_JOBID": "forged",
            }
        )

        self.assertEqual(filtered["TMAX_ATTEMPT_ID"], "rl-007")
        self.assertEqual(filtered["RSI_RL_LEARNING_RATE"], "1e-6")
        self.assertNotIn("OPENAI_API_KEY", filtered)
        self.assertNotIn("LSB_JOBID", filtered)


if __name__ == "__main__":
    unittest.main()
