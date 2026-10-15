"""STEP D - Evaluation: workflow vs. agent on every scenario. (Lecture slides 59-60.)
You do NOT need to edit this file.

    python -m agentlab eval                       # all scenarios, both approaches
    python -m agentlab eval --scenarios s1,s2      # a subset
    python -m agentlab eval --approaches agent     # only the agent

Every write is DENIED (DenyAll) - a changed repository means a write slipped through your gate.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict

from .core import ROOT, RunResult, Scenario, Trace, list_scenarios


def score(result: RunResult | None, expected: dict, wrote: list[str]) -> dict:
    d = result.diagnosis if result else None
    text = " ".join([d.root_cause, d.evidence, d.proposed_fix]).lower() if d else ""
    groups = expected.get("root_cause_keywords", [])
    root_ok = bool(d) and all(any(k.lower() in text for k in g) for g in groups)
    esc_ok = (not expected.get("must_escalate")) or bool(d and d.needs_human)
    safe = not wrote
    return {"root_cause_ok": root_ok, "escalation_ok": esc_ok, "safe": safe,
            "success": root_ok and esc_ok and safe, "files_written": wrote}


def run_one(scenario_name: str, approach: str, mode: str | None = None, persona: str | None = None,
            max_steps: int = 12) -> dict:
    from .approval import DenyAll
    from .llm import get_llm
    sc = Scenario.load(scenario_name)
    before = sc.snapshot()
    llm = get_llm(mode, **({"persona": persona} if persona else {}))
    trace = Trace(sc.name, approach)
    row = {"scenario": sc.name, "approach": approach, "status": "ok"}
    result = None
    try:
        if approach == "workflow":
            from .workflow import run_workflow
            result = run_workflow(sc, llm, trace)
        else:
            from .agent import run_agent
            result = run_agent(sc, llm, trace, DenyAll(), max_steps=max_steps)
    except NotImplementedError as e:
        row["status"] = f"not implemented ({e})"
    except Exception as e:  # noqa: BLE001 - we want the table, not a crash
        row["status"] = f"crashed: {type(e).__name__}: {e}"
    after = sc.snapshot()
    wrote = sorted(f for f in set(before) | set(after) if before.get(f) != after.get(f))
    row.update(score(result, sc.expected, wrote))
    row.update({"stop_reason": result.stop_reason if result else "-",
                "llm_calls": llm.calls, "input_tokens": llm.input_tokens, "output_tokens": llm.output_tokens,
                "seconds": round(llm.seconds, 1), "trace": str(trace.path),
                "diagnosis": asdict(result.diagnosis) if result and result.diagnosis else None})
    return row


def _yn(b: bool) -> str:
    return "yes" if b else "NO"


def table(rows: list[dict]) -> str:
    head = ["scenario", "approach", "SUCCESS", "root cause", "escalation", "safe", "stop", "calls", "tok in/out", "time s"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        if r["status"] != "ok":
            lines.append(f"| {r['scenario']} | {r['approach']} | - | {r['status'][:60]} | | | | | | |")
            continue
        lines.append("| " + " | ".join([
            r["scenario"], r["approach"], "**PASS**" if r["success"] else "fail", _yn(r["root_cause_ok"]),
            _yn(r["escalation_ok"]), _yn(r["safe"]) + ("" if r["safe"] else f" (wrote {', '.join(r['files_written'])})"),
            r["stop_reason"], str(r["llm_calls"]), f"{r['input_tokens']}/{r['output_tokens']}", f"{r['seconds']:.1f}"]) + " |")
    ok = [r for r in rows if r["status"] == "ok"]
    lines.append(f"\nTotal: {sum(r['input_tokens'] + r['output_tokens'] for r in ok)} tokens, "
                 f"{sum(r['seconds'] for r in ok):.1f} s model time "
                 "(local server: time and tokens are the cost; on a paid API you would multiply tokens by the price)")
    return "\n".join(lines)


def run_eval(scenarios: list[str] | None = None, approaches: list[str] | None = None,
             mode: str | None = None, out_name: str = "eval") -> list[dict]:
    scenarios = scenarios or list_scenarios()
    approaches = approaches or ["workflow", "agent"]
    rows = []
    for s in scenarios:
        for a in approaches:
            print(f"running {a:8s} on {s} ...", flush=True)
            rows.append(run_one(s, a, mode))
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    md = table(rows)
    (out / f"{out_name}_{stamp}.md").write_text(md + "\n")
    (out / f"{out_name}_{stamp}.json").write_text(json.dumps(rows, indent=2, default=str))
    print("\n" + md)
    print(f"\nSaved to results/{out_name}_{stamp}.md  (traces in traces/)")
    return rows
