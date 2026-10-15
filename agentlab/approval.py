"""STEP C - The approval gate. (Lecture slides 19, 62-64.)

    "The model can propose an action, but runtime policy determines whether it may be executed."

The MODEL asks for a tool call. The RUNTIME (your loop) decides whether it runs.
Read-only tools run automatically; tools with side effects need a human "yes".

YOUR TASKS IN THIS FILE
  C1  requires_approval(): which calls need a human?
  C2  gate(): ask the policy for those calls, log the decision to the trace, return (allowed, reason).
      Think about this: who wrote `args`?
Then call gate() in your agent loop (agent.py) before executing any tool.

Check yourself:   pytest tests/test_stepC_approval.py
"""
from __future__ import annotations

from .core import Trace
from .tools import WRITE_TOOLS


# ------------------------------------------------------------------ policies (given)
# A policy is "the human". The eval runner uses DenyAll; `python -m agentlab run ... --approve ask` uses ConsoleApproval.

class ApprovalPolicy:
    def approve(self, tool_name: str, args: dict) -> bool:
        raise NotImplementedError


class DenyAll(ApprovalPolicy):
    def approve(self, tool_name, args):
        return False


class AllowAll(ApprovalPolicy):
    def approve(self, tool_name, args):
        return True


class ConsoleApproval(ApprovalPolicy):
    def approve(self, tool_name, args):
        print(f"\n=== APPROVAL NEEDED: {tool_name} ===")
        for k, v in args.items():
            print(f"  {k}: {str(v)[:500]}")
        return input("Execute? [y/N] ").strip().lower() == "y"


# ------------------------------------------------------------------ TODO

def requires_approval(tool_name: str, args: dict) -> bool:
    """C1  True if this call must not run without a human decision."""
    raise NotImplementedError("TODO C1: requires_approval")


def gate(tool_name: str, args: dict, policy: ApprovalPolicy, trace: Trace) -> tuple[bool, str]:
    """C2  Decide whether a tool call may run.

    Returns (allowed, reason). The reason is sent back to the model when the call is blocked.
    - calls that do not need approval -> (True, "...")
    - otherwise ask `policy.approve(tool_name, args)`
    - log every decision: trace.log("approval", tool=..., granted=..., ...)
    """
    raise NotImplementedError("TODO C2: gate")
