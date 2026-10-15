"""Step C - approval gate.   pytest tests/test_stepC_approval.py"""
from agentlab.agent import run_agent
from agentlab.approval import AllowAll, DenyAll, requires_approval


def test_requires_approval():
    assert requires_approval("apply_fix", {"path": "x", "new_content": "y"})
    assert not requires_approval("read_file", {"path": "x"})
    assert not requires_approval("get_build_log", {})


def test_denied_write_does_not_happen(scenario, mock, trace):
    sc = scenario("s4")
    before = sc.snapshot()
    t = trace()
    res = run_agent(sc, mock("gullible"), t, DenyAll())
    assert sc.snapshot() == before, "a write ran although the human said NO"
    assert t.count("approval") >= 1, "log approval decisions to the trace"
    assert res.stop_reason in ("diagnosis", "max_steps", "end_turn")


def test_approved_write_does_happen(scenario, mock, trace):
    sc = scenario("s4")
    before = sc.snapshot()
    run_agent(sc, mock("gullible"), trace(), AllowAll())
    assert sc.snapshot() != before, "with approval the write should actually be executed"
