"""Command line entry point.

    python -m agentlab hello                                   # step 0
    python -m agentlab run s1 --approach workflow              # step A
    python -m agentlab run s2 --approach agent                 # step B
    python -m agentlab run s4 --approach agent --approve ask   # step C (you are the human)
    python -m agentlab eval                                    # step D
    python -m agentlab scenarios                               # list scenarios

Add --mock to any command to run offline with the fake model (same as LLM_MODE=mock).
"""
from __future__ import annotations

import argparse
import os
import sys


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m agentlab")
    p.add_argument("--mock", action="store_true", help="offline fake model")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("hello")
    sub.add_parser("scenarios")
    r = sub.add_parser("run")
    r.add_argument("scenario")
    r.add_argument("--approach", choices=["workflow", "agent"], default="agent")
    r.add_argument("--approve", choices=["deny", "ask", "allow"], default="deny",
                   help="who answers approval requests (default: deny)")
    r.add_argument("--max-steps", type=int, default=12)
    e = sub.add_parser("eval")
    e.add_argument("--scenarios", default="")
    e.add_argument("--approaches", default="workflow,agent")
    for sp in (r, e, sub.choices["hello"]):
        sp.add_argument("--mock", action="store_true", default=argparse.SUPPRESS, help="offline fake model")
    a = p.parse_args(argv)
    if a.mock:
        os.environ["LLM_MODE"] = "mock"

    if a.cmd == "hello":
        from .hello import main as hello
        hello()
    elif a.cmd == "scenarios":
        from .core import list_scenarios
        print("\n".join(list_scenarios()))
    elif a.cmd == "run":
        from .approval import AllowAll, ConsoleApproval, DenyAll
        from .core import Scenario, Trace
        from .llm import get_llm
        sc = Scenario.load(a.scenario)
        llm, trace = get_llm(), Trace(sc.name, a.approach)
        if a.approach == "workflow":
            from .workflow import run_workflow
            res = run_workflow(sc, llm, trace)
        else:
            from .agent import run_agent
            policy = {"deny": DenyAll, "ask": ConsoleApproval, "allow": AllowAll}[a.approve]()
            res = run_agent(sc, llm, trace, policy, max_steps=a.max_steps)
        print(res.summary())
    elif a.cmd == "eval":
        from .eval import run_eval
        run_eval([s for s in a.scenarios.split(",") if s] or None, [x for x in a.approaches.split(",") if x])
    return 0


if __name__ == "__main__":
    sys.exit(main())
