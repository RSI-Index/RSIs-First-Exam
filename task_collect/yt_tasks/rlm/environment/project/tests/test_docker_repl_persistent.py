"""Tests for DockerREPL persistence + compaction features.

These mirror tests/test_local_repl_persistent.py but verify behavior by
executing code inside the container (docker locals are repr strings, not a live
host dict). Requires a working Docker daemon; skipped otherwise.
"""

import shutil
import subprocess

import pytest

from rlm.environments import SupportsPersistence


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=15).returncode == 0
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _docker_available(), reason="Docker not available")


@pytest.fixture
def repl():
    from rlm.environments.docker_repl import DockerREPL

    r = DockerREPL()
    yield r
    r.cleanup()


@pytest.fixture
def repl_compaction():
    from rlm.environments.docker_repl import DockerREPL

    r = DockerREPL(compaction=True)
    yield r
    r.cleanup()


class TestDockerProtocol:
    def test_satisfies_persistence_protocol(self, repl):
        assert isinstance(repl, SupportsPersistence)


class TestDockerMultiContext:
    def test_add_context_versioning_and_access(self, repl):
        repl.add_context("First", 0)
        repl.add_context("Second", 1)
        assert repl.get_context_count() == 2
        r = repl.execute_code("print(f'{context_0}|{context_1}|{context}')")
        assert r.stdout.strip() == "First|Second|First", (r.stdout, r.stderr)

    def test_add_context_auto_increment(self, repl):
        assert repl.add_context("A") == 0
        assert repl.add_context("B") == 1
        assert repl.get_context_count() == 2
        r = repl.execute_code("print(context_1)")
        assert r.stdout.strip() == "B"

    def test_context_alias_points_to_first(self, repl):
        repl.add_context("First")
        repl.add_context("Second")
        r = repl.execute_code("print(context == context_0)")
        assert r.stdout.strip() == "True"

    def test_dict_context(self, repl):
        repl.add_context({"k": "v", "n": 5}, 0)
        r = repl.execute_code("print(context_0['k'], context_0['n'])")
        assert r.stdout.strip() == "v 5"


class TestDockerMultiHistory:
    def test_add_history_versioning(self, repl):
        h0 = [{"role": "user", "content": "hi"}]
        h1 = [{"role": "assistant", "content": "yo"}]
        assert repl.add_history(h0) == 0
        assert repl.add_history(h1) == 1
        assert repl.get_history_count() == 2
        r = repl.execute_code("print(history_0[0]['content'], history_1[0]['content'])")
        assert r.stdout.strip() == "hi yo"

    def test_history_alias_points_to_first(self, repl):
        repl.add_history([{"role": "user", "content": "first"}])
        r = repl.execute_code("print(history[0]['content'])")
        assert r.stdout.strip() == "first"

    def test_add_history_deep_copy(self, repl):
        msgs = [{"role": "user", "content": "orig"}]
        repl.add_history(msgs)
        # Mutating the caller's list must not affect the stored history.
        msgs[0]["content"] = "mutated"
        r = repl.execute_code("print(history_0[0]['content'])")
        assert r.stdout.strip() == "orig"


class TestDockerUpdateHandlerAddress:
    def test_update_handler_address(self, repl):
        repl.update_handler_address(("127.0.0.1", 6000))
        assert repl.lm_handler_address == ("127.0.0.1", 6000)
        assert repl.proxy_server.lm_handler_address == ("127.0.0.1", 6000)


class TestDockerCompaction:
    def test_history_seeded_as_list(self, repl_compaction):
        r = repl_compaction.execute_code("print(type(history).__name__, len(history))")
        assert r.stdout.strip() == "list 0"

    def test_append_compaction_entry(self, repl_compaction):
        repl_compaction.append_compaction_entry([{"role": "user", "content": "turn1"}])
        repl_compaction.append_compaction_entry({"type": "summary", "content": "did stuff"})
        r = repl_compaction.execute_code(
            "print(len(history)); print(history[1]['type']); print(history[1]['content'])"
        )
        lines = r.stdout.strip().split("\n")
        assert lines == ["2", "summary", "did stuff"], (r.stdout, r.stderr)

    def test_append_ignored_without_compaction(self, repl):
        # Non-compaction env: append is a no-op and must not error.
        repl.append_compaction_entry({"type": "summary", "content": "x"})

    def test_history_survives_model_overwrite(self, repl_compaction):
        repl_compaction.append_compaction_entry({"type": "summary", "content": "keep"})
        # Model clobbers history mid-cell...
        repl_compaction.execute_code("history = 'corrupted'")
        # ...next append must restore the canonical accumulated list.
        repl_compaction.append_compaction_entry({"type": "summary", "content": "keep2"})
        r = repl_compaction.execute_code("print(len(history)); print(history[-1]['content'])")
        assert r.stdout.strip().split("\n") == ["2", "keep2"], (r.stdout, r.stderr)
