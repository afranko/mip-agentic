"""Step B - agent loop.   pytest tests/test_stepB_agent.py"""
import json

from agentlab.agent import run_agent
from agentlab.approval import DenyAll


def test_agent_s1(scenario, mock, trace):
    llm = mock("good")
    res = run_agent(scenario("s1"), llm, trace(), DenyAll())
    assert res.stop_reason == "diagnosis", res.summary()
    assert "pyyaml" in res.diagnosis.root_cause.lower()
    assert llm.calls >= 3, "the agent should have used several tools"


def test_agent_finds_real_cause_on_s2(scenario, mock, trace):
    res = run_agent(scenario("s2"), mock("good"), trace(), DenyAll())
    assert res.stop_reason == "diagnosis", res.summary()
    assert "requirement.txt" in res.diagnosis.root_cause.lower()


def test_agent_s3_reads_ci_config(scenario, mock, trace):
    res = run_agent(scenario("s3"), mock("good"), trace(), DenyAll())
    assert "3.9" in res.diagnosis.root_cause


def test_agent_stops_at_max_steps(scenario, mock, trace):
    llm = mock("looper")
    res = run_agent(scenario("s1"), llm, trace(), DenyAll(), max_steps=4)
    assert res.stop_reason == "max_steps", res.summary()
    assert llm.calls <= 4


def test_trace_records_tool_calls(scenario, mock, trace):
    t = trace()
    run_agent(scenario("s1"), mock("good"), t, DenyAll())
    events = [json.loads(l)["event"] for l in t.path.read_text().splitlines()]
    assert "tool_result" in events and "stop" in events, f"log tool calls/results to the trace. Got: {set(events)}"


def test_agent_survives_broken_json_arguments(scenario, mock, trace):
    # gpt-oss sometimes writes tool arguments that are not valid JSON. Send the error back; do not crash.
    res = run_agent(scenario("s1"), mock("sloppy"), trace(), DenyAll())
    assert res.stop_reason == "diagnosis", res.summary()
