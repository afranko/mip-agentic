"""Shared infrastructure. You do NOT need to edit this file.

Contents:
  - Scenario      : one simulated CI failure, copied into a private temp workspace
  - Diagnosis     : the structured answer an approach must produce
  - RunResult     : what every approach (workflow / agent) returns
  - Trace         : JSONL execution trace (lecture slide 61: "log the path")
  - validate_args : tiny JSON-schema checker for tool arguments
  - message_to_dict, tool_message, parse_arguments : OpenAI chat-format message helpers
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent

TASK = (
    "The CI build of our repository is failing. Investigate why. "
    "Find the root cause, back it up with evidence (log lines, files, commits), "
    "and propose a fix. Finish by calling submit_diagnosis."
)


class ToolError(Exception):
    """Raised by a tool when the call cannot be executed. The message is sent back to the model."""


# --------------------------------------------------------------------------- scenarios

def _scenario_dirs() -> list[Path]:
    dirs = [ROOT / "scenarios"]
    for extra in os.environ.get("AGENTLAB_SCENARIO_DIRS", "").split(os.pathsep):
        if extra:
            dirs.append(Path(extra))
    return dirs


def list_scenarios() -> list[str]:
    names = []
    for d in _scenario_dirs():
        if d.is_dir():
            names += sorted(p.name for p in d.iterdir() if (p / "build.log").exists())
    return names


@dataclass
class Scenario:
    name: str
    source: Path      # original folder (never modified)
    workspace: Path   # private temp copy; tools read and write here

    @classmethod
    def load(cls, name_or_path: str | Path) -> "Scenario":
        p = Path(name_or_path)
        candidates = [p] if p.is_dir() else []
        for d in _scenario_dirs():
            candidates += [d / str(name_or_path)]
            candidates += sorted(d.glob(f"{name_or_path}*"))  # allow "s1" for "s1_missing_dep"
        for c in candidates:
            if (c / "build.log").exists():
                ws = Path(tempfile.mkdtemp(prefix=f"agentlab_{c.name}_"))
                shutil.copytree(c, ws, dirs_exist_ok=True)
                return cls(name=c.name, source=c, workspace=ws)
        raise FileNotFoundError(f"Unknown scenario '{name_or_path}'. Available: {', '.join(list_scenarios())}")

    # --- data access (used by the tools) ---
    @property
    def repo(self) -> Path:
        return self.workspace / "repo"

    def build_log(self) -> str:
        return (self.workspace / "build.log").read_text()

    def git_log(self) -> list[dict]:
        return json.loads((self.workspace / "git_log.json").read_text())

    @property
    def expected(self) -> dict:
        f = self.source / "expected.json"
        return json.loads(f.read_text()) if f.exists() else {}

    def resolve(self, rel_path: str) -> Path:
        """Turn a repo-relative path into an absolute path INSIDE the repo (sandbox).

        Raises ToolError for anything that would escape the repo ("../", absolute paths, symlinks).
        """
        if not isinstance(rel_path, str) or not rel_path.strip():
            raise ToolError("path must be a non-empty string")
        root = self.repo.resolve()
        target = (root / rel_path.lstrip("/\\")).resolve() if not os.path.isabs(rel_path) else Path(rel_path).resolve()
        if target != root and root not in target.parents:
            raise ToolError(f"Access denied: '{rel_path}' is outside the repository.")
        return target

    def repo_files(self) -> list[str]:
        return sorted(str(p.relative_to(self.repo)).replace("\\", "/")
                      for p in self.repo.rglob("*") if p.is_file())

    def snapshot(self) -> dict[str, str]:
        """Hash of every repo file. Compare before/after a run to detect writes."""
        return {f: hashlib.sha256((self.repo / f).read_bytes()).hexdigest() for f in self.repo_files()}


# --------------------------------------------------------------------------- results

@dataclass
class Diagnosis:
    root_cause: str
    evidence: str
    proposed_fix: str
    needs_human: bool = False
    confidence: str = "medium"

    @classmethod
    def from_tool_input(cls, d: dict) -> "Diagnosis":
        return cls(
            root_cause=str(d.get("root_cause", "")),
            evidence=str(d.get("evidence", "")),
            proposed_fix=str(d.get("proposed_fix", "")),
            needs_human=bool(d.get("needs_human", False)),
            confidence=str(d.get("confidence", "medium")),
        )


@dataclass
class RunResult:
    scenario: str
    approach: str
    diagnosis: Diagnosis | None
    stop_reason: str          # "diagnosis" | "max_steps" | "end_turn" | "budget" | "error"
    steps: int                # number of LLM calls
    input_tokens: int
    output_tokens: int
    seconds: float            # time spent waiting for the model (local server: this is the "cost")
    trace_path: str
    error: str | None = None

    @classmethod
    def build(cls, scenario: Scenario, approach: str, llm: Any, trace: "Trace",
              diagnosis: Diagnosis | None, stop_reason: str, error: str | None = None) -> "RunResult":
        trace.log("stop", stop_reason=stop_reason, error=error,
                  diagnosis=asdict(diagnosis) if diagnosis else None)
        return cls(scenario=scenario.name, approach=approach, diagnosis=diagnosis,
                   stop_reason=stop_reason, steps=llm.calls, input_tokens=llm.input_tokens,
                   output_tokens=llm.output_tokens, seconds=round(llm.seconds, 2),
                   trace_path=str(trace.path), error=error)

    def summary(self) -> str:
        d = self.diagnosis
        lines = [f"[{self.approach} | {self.scenario}] stop_reason={self.stop_reason}  llm_calls={self.steps}  "
                 f"tokens in/out={self.input_tokens}/{self.output_tokens}  model time={self.seconds:.1f}s"]
        if d:
            lines += [f"  root cause : {d.root_cause}", f"  evidence   : {d.evidence}",
                      f"  fix        : {d.proposed_fix}", f"  needs human: {d.needs_human}  (confidence: {d.confidence})"]
        if self.error:
            lines.append(f"  error: {self.error}")
        lines.append(f"  trace: {self.trace_path}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- trace

class Trace:
    """Append-only JSONL trace. One line per event. Open it in any editor."""

    def __init__(self, scenario: str, approach: str, out_dir: Path | None = None):
        out = Path(out_dir or ROOT / "traces")
        out.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.path = out / f"{stamp}_{scenario}_{approach}_{os.getpid()}_{int(time.time()*1000) % 100000}.jsonl"
        self.events: list[dict] = []

    def log(self, event: str, **data: Any) -> None:
        rec = {"t": round(time.time(), 3), "event": event, **_shorten(data)}
        self.events.append(rec)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

    def count(self, event: str) -> int:
        return sum(1 for e in self.events if e["event"] == event)


def _shorten(x: Any, limit: int = 2000) -> Any:
    if isinstance(x, str):
        return x if len(x) <= limit else x[:limit] + f"...[+{len(x) - limit} chars]"
    if isinstance(x, dict):
        return {k: _shorten(v, limit) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_shorten(v, limit) for v in x]
    return x


# --------------------------------------------------------------------------- message helpers
# OpenAI Chat Completions format (what vLLM, Ollama, llama.cpp, LM Studio ... all speak):
#   {"role": "system" | "user", "content": "..."}
#   {"role": "assistant", "content": "..." or None, "tool_calls": [{"id", "type": "function",
#                                                   "function": {"name", "arguments": "<JSON STRING>"}}]}
#   {"role": "tool", "tool_call_id": "...", "content": "..."}        one message PER tool call

_KEEP = ("role", "content", "tool_calls", "reasoning", "reasoning_content", "thinking")


def _get(obj: Any, key: str) -> Any:
    return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)


def message_to_dict(msg: Any) -> dict:
    """The assistant message from the response -> a plain dict you can append to `messages`.

    Keeps the model's reasoning text (gpt-oss returns it in a separate field; the field name depends on
    the server) so that the model sees its own earlier reasoning in the next call.
    """
    if hasattr(msg, "model_dump"):                    # openai SDK object (pydantic, extra fields kept)
        raw = msg.model_dump(exclude_none=True)
    elif isinstance(msg, dict):
        raw = dict(msg)
    else:                                             # mock / manual dataclass
        raw = {k: getattr(msg, k, None) for k in _KEEP}
    out: dict[str, Any] = {"role": raw.get("role") or "assistant", "content": raw.get("content")}
    calls = raw.get("tool_calls") or []
    if calls:
        out["tool_calls"] = [{"id": _get(c, "id"), "type": "function",
                              "function": {"name": _get(_get(c, "function"), "name"),
                                           "arguments": _get(_get(c, "function"), "arguments") or "{}"}}
                             for c in calls]
    for k in ("reasoning", "reasoning_content", "thinking"):
        if raw.get(k):
            out[k] = raw[k]
    return out


def tool_message(tool_call: Any, content: str, is_error: bool = False) -> dict:
    """The message you send back after running ONE tool call (role='tool').

    The chat format has no error flag, so errors are marked in the text - the model reads it and can recover.
    `tool_call` is the tool call object from the response (or its id as a string).
    """
    if isinstance(tool_call, str):
        call_id, name = tool_call, None
    else:
        call_id, name = _get(tool_call, "id"), _get(_get(tool_call, "function"), "name")
    msg = {"role": "tool", "tool_call_id": call_id, "content": ("ERROR: " + content) if is_error else content}
    if name:
        msg["name"] = name          # not required by OpenAI, but some servers (gpt-oss on vLLM) use it
    return msg


def parse_arguments(tool_call: Any) -> dict:
    """tool_call.function.arguments is a JSON *string* written by the model. Parse it.

    Raises ToolError if it is not a JSON object - send that back to the model as an error result.
    """
    raw = _get(_get(tool_call, "function"), "arguments")
    if isinstance(raw, dict):
        return raw
    if raw is None or not str(raw).strip():
        return {}
    try:
        val = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ToolError(f"Arguments are not valid JSON ({e.msg}). Send a JSON object.") from None
    if not isinstance(val, dict):
        raise ToolError("Arguments must be a JSON object.")
    return val


# --------------------------------------------------------------------------- schema validation

_PY_TYPES = {"string": str, "integer": int, "boolean": bool, "number": (int, float),
             "array": list, "object": dict}


def validate_args(schema: dict, args: Any) -> list[str]:
    """Return a list of problems with `args` according to a tool's input_schema ([] = valid).

    Checks: object type, required keys, unknown keys (when additionalProperties is false),
    primitive types and enums. Deliberately small - enough to reject malformed or extra arguments.
    """
    if not isinstance(args, dict):
        return ["arguments must be a JSON object"]
    errors = []
    props = schema.get("properties", {})
    for key in schema.get("required", []):
        if key not in args:
            errors.append(f"missing required argument '{key}'")
    if schema.get("additionalProperties") is False:
        for key in args:
            if key not in props:
                errors.append(f"unknown argument '{key}'")
    for key, value in args.items():
        spec = props.get(key)
        if not spec:
            continue
        py = _PY_TYPES.get(spec.get("type"))
        if py and not isinstance(value, py) or (spec.get("type") == "integer" and isinstance(value, bool)):
            errors.append(f"argument '{key}' must be of type {spec.get('type')}")
        if "enum" in spec and value not in spec["enum"]:
            errors.append(f"argument '{key}' must be one of {spec['enum']}")
    return errors
