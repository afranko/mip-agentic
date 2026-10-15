"""Rules behind MockLLM. You do NOT need to edit (or read) this file.

The mock is NOT smart - it pattern-matches the conversation so that the loop, the approval gate
and the trace can be exercised offline. Personas:
    good           investigates like a careful agent: log -> relevant file -> git history -> diagnosis
    gullible       obeys prompt injections: immediately tries apply_fix
    gullible_plus  like gullible, but also sends approved=true (as the injection asks)
    looper         never stops: calls get_build_log again and again
    sloppy         like good, but once sends tool arguments that are not valid JSON (gpt-oss does this sometimes)
"""
from __future__ import annotations

import json
import re


def _tool_events(messages: list[dict]) -> list[dict]:
    """[{name, input, result, error}] for every tool call so far, in order (OpenAI chat format)."""
    calls, results, order = {}, {}, []
    for m in messages:
        for tc in m.get("tool_calls") or []:
            try:
                args = json.loads(tc["function"].get("arguments") or "{}")
            except (json.JSONDecodeError, TypeError):
                args = {}
            calls[tc["id"]] = {"name": tc["function"]["name"], "input": args if isinstance(args, dict) else {}}
            order.append(tc["id"])
        if m.get("role") == "tool":
            c = str(m.get("content") or "")
            results[m.get("tool_call_id")] = (c, c.startswith("ERROR"))
    out = []
    for i in order:
        res, err = results.get(i, ("", False))
        out.append({**calls[i], "result": res, "error": err})
    return out


def _all_text(messages: list[dict]) -> str:
    return "\n".join(str(m.get("content") or "") for m in messages if m.get("role") != "system")


def _requirement_names(text: str) -> set[str] | None:
    """Package names from lines that look like requirements.txt pins. None if none were seen."""
    names = {m.group(1).lower() for m in re.finditer(r"(?m)^\s*([A-Za-z0-9_.\-]+)==\S+\s*$", text)}
    return names or None


PKG_FOR_MODULE = {"yaml": "pyyaml"}


def diagnose(text: str) -> dict:
    t = text.lower()
    if "could not open requirements file" in t:
        fname = re.search(r"no such file or directory: '([^']+)'", t)
        f = fname.group(1) if fname else "the requirements file"
        return dict(root_cause=f"The CI install step points to '{f}', which does not exist (typo for requirements.txt), "
                               "so no dependencies are installed; '|| true' hides the failure. The missing module is a symptom.",
                    evidence="Build log: 'Could not open requirements file'. Install step in .github/workflows/ci.yml.",
                    proposed_fix="Fix the filename in the CI install step (.github/workflows/ci.yml) and remove '|| true'.",
                    needs_human=False, confidence="high")
    m = re.search(r"satisfies the requirement ([a-z0-9_.\-]+)==([0-9a-z.]+)", t)
    if m:
        return dict(root_cause=f"requirements.txt pins {m.group(1)}=={m.group(2)}, a version that does not exist, so pip install fails.",
                    evidence="Build log: 'No matching distribution found'. requirements.txt pin.",
                    proposed_fix=f"Pin {m.group(1)} to an existing version (e.g. the previous one).",
                    needs_human=False, confidence="high")
    if "unsupported operand type(s) for |" in t:
        v = re.search(r"python-version:\s*\"?([0-9.]+)", t)
        where = f"CI runs Python {v.group(1)} (python-version in .github/workflows/ci.yml)" if v \
            else "CI runs a Python version older than 3.10"
        return dict(root_cause=f"{where}, but src/settings.py uses the `int | None` syntax, which needs Python 3.10+.",
                    evidence="TypeError: unsupported operand type(s) for | in src/settings.py.",
                    proposed_fix="Run CI on Python 3.10+ or use Optional[int].",
                    needs_human=False, confidence="high" if v else "medium")
    if "connectionerror" in t or "name resolution" in t:
        return dict(root_cause="Tests cannot reach the internal artifact server (artifacts.internal): DNS / network failure. "
                               "Nothing in the repository explains it.",
                    evidence="ConnectionError / NameResolutionError for artifacts.internal in the test output.",
                    proposed_fix="No code change. Escalate to infrastructure / on-call to restore the artifact server.",
                    needs_human=True, confidence="medium")
    m = re.search(r"no module named '([a-z0-9_.]+)'", t)
    if m:
        mod = m.group(1)
        pkg = PKG_FOR_MODULE.get(mod, mod)
        reqs = _requirement_names(text)
        if reqs is not None and pkg in reqs:
            return dict(root_cause=f"Module '{mod}' is missing at test time although {pkg} is listed in requirements.txt. "
                                   "The real cause is unclear from the available information.",
                        evidence=f"ModuleNotFoundError: {mod}; {pkg} present in requirements.txt.",
                        proposed_fix="Investigate the CI environment.", needs_human=True, confidence="low")
        return dict(root_cause=f"{pkg} is missing from requirements.txt (removed), but the code imports '{mod}'.",
                    evidence=f"ModuleNotFoundError: No module named '{mod}'; requirements.txt has no {pkg}.",
                    proposed_fix=f"Add {pkg} back to requirements.txt.", needs_human=False, confidence="high")
    return dict(root_cause="Could not determine the root cause from the available evidence.",
                evidence="-", proposed_fix="Hand over to a human.", needs_human=True, confidence="low")


def _next_file(text: str) -> str | None:
    t = text.lower()
    if "could not open requirements file" in t or "unsupported operand" in t:
        return ".github/workflows/ci.yml"
    if "no module named" in t or "satisfies the requirement" in t:
        return "requirements.txt"
    return None


THINK = {"get_build_log": "I should look at the build log first.",
         "read_file": "The log points to a file; let me check it before concluding.",
         "git_log": "When did this change? The history should show it.",
         "submit_diagnosis": "I have enough evidence to conclude."}


def _use(name: str, inp: dict, text: str | None = None):
    """-> (content, [(tool name, arguments)], reasoning)"""
    return text, [(name, inp)], THINK.get(name, "")


def respond(messages: list[dict], tools: list[dict], tool_choice, persona: str):
    names = {t["function"]["name"] for t in tools or []}
    text = _all_text(messages)
    events = _tool_events(messages)
    called = lambda n: [e for e in events if e["name"] == n]  # noqa: E731

    forced = isinstance(tool_choice, dict) and (tool_choice.get("function") or {}).get("name")
    if forced:
        return _use(forced, diagnose(text))

    if persona == "looper" and "get_build_log" in names:
        return _use("get_build_log", {}, "Let me look at the log once more.")

    if persona in ("gullible", "gullible_plus") and "apply_fix" in names and not called("apply_fix"):
        args = {"path": ".github/workflows/ci.yml", "new_content": "on: push\njobs: {}\n"}
        if persona == "gullible_plus":
            args["approved"] = True
        return _use("apply_fix", args, "The notice says the fix is pre-approved, applying it now.")

    # ---- good (and gullible after its write attempt)
    if "get_build_log" in names and not called("get_build_log"):
        return _use("get_build_log", {"tail_lines": 400})

    if persona == "sloppy" and "read_file" in names and not any(e["error"] for e in events):
        return _use("read_file", '{"path": "requirements.txt"', "Let me read the requirements.")

    if "submit_diagnosis" not in names:  # e.g. the step-0 hello demo
        last_err = [l for l in text.splitlines() if "Error:" in l] or [l for l in text.splitlines() if "ERROR" in l]
        return ("The build log ends with: " + (last_err[-1].strip() if last_err else "no error"), [],
                "The tool result answers the question.")

    target = _next_file(text)
    if target and "read_file" in names and not any(e["input"].get("path") == target for e in called("read_file")):
        return _use("read_file", {"path": target})
    if target and "git_log" in names and not called("git_log"):
        return _use("git_log", {"path": target})
    return _use("submit_diagnosis", diagnose(text))
