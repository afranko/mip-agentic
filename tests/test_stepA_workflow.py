"""Step A - workflow.   pytest tests/test_stepA_workflow.py"""
from agentlab.workflow import run_workflow


def test_workflow_s1(scenario, mock, trace):
    llm = mock("good")
    res = run_workflow(scenario("s1"), llm, trace())
    assert res.stop_reason == "diagnosis", res.summary()
    assert "pyyaml" in res.diagnosis.root_cause.lower()
    assert llm.calls == 1, "a workflow makes exactly ONE LLM call - the steps are decided by code"


def test_workflow_is_blind_on_s2(scenario, mock, trace):
    # Not a bug in your code: the fixed chain never looks at the CI config.
    res = run_workflow(scenario("s2"), mock("good"), trace())
    assert res.diagnosis is not None
    assert "requirement.txt" not in res.diagnosis.root_cause.lower()
