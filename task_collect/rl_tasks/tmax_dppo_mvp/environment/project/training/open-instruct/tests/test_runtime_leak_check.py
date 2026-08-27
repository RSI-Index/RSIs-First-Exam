from open_instruct import utils


def test_runtime_leak_check_can_be_disabled_for_cluster_shutdown(monkeypatch):
    monkeypatch.setenv("OPEN_INSTRUCT_SKIP_RUNTIME_LEAK_CHECK", "1")

    def unexpected_active_children_call():
        raise AssertionError("runtime leak inspection should have been skipped")

    monkeypatch.setattr(utils.mp, "active_children", unexpected_active_children_call)

    utils.check_runtime_leaks()
