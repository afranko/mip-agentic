"""Tools: the agent's hands. (Lecture: "A good tool is a good API", slides 15-19.)

STEP 0 - YOUR TASKS IN THIS FILE
  0.1  Write a real `description` for read_file, git_log and apply_fix:
       say WHEN to use the tool, not only what it does. The model only sees name + description + schema.
  0.2  Implement read_file()   (use scenario.resolve() - it is the sandbox)
  0.3  Implement git_log()
  0.4  Implement apply_fix()   (just write the file - the approval check does NOT belong here, see approval.py)

Check yourself:   pytest tests/test_step0_tools.py
"""
from __future__ import annotations

from .core import Scenario, ToolError

# OpenAI Chat Completions tool format: {"type": "function", "function": {name, description, parameters}}.
# The tool NAMES and parameter names are a fixed contract (the tests and the mock model use them).
# The descriptions are yours to improve.
TOOL_SCHEMAS: list[dict] = [
    {"type": "function", "function": {
        "name": "get_build_log",
        "description": (
            "Return the CI build log of the failing run. Use this first to see the error. "
            "By default only the last lines are returned; raise tail_lines if the cause may be earlier "
            "in the log (e.g. a failed install step)."
        ),
        "parameters": {
            "type": "object",
            "properties": {"tail_lines": {"type": "integer", "description": "How many lines from the end (default 60, max 400)."}},
            "required": [],
            "additionalProperties": False,
        },
    }},
    {"type": "function", "function": {
        "name": "list_files",
        "description": "List all files in the repository. Use it to find out which files exist before reading them.",
        "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
    }},
    {"type": "function", "function": {
        "name": "read_file",
        "description": "TODO (0.1): when should the model use this tool?",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Repository-relative path, e.g. requirements.txt"}},
            "required": ["path"],
            "additionalProperties": False,
        },
    }},
    {"type": "function", "function": {
        "name": "git_log",
        "description": "TODO (0.1): when should the model use this tool?",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Only commits touching this file (optional)."},
                "max_commits": {"type": "integer", "description": "Default 10."},
            },
            "required": [],
            "additionalProperties": False,
        },
    }},
    {"type": "function", "function": {
        "name": "apply_fix",
        "description": "TODO (0.1): when should the model use this tool? (It changes files!)",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Repository-relative path of the file to overwrite."},
                "new_content": {"type": "string", "description": "The complete new content of the file."},
            },
            "required": ["path", "new_content"],
            "additionalProperties": False,
        },
    }},
    {"type": "function", "function": {
        "name": "submit_diagnosis",
        "description": (
            "Finish the investigation. Call this exactly once, when you have enough evidence - or when the "
            "problem cannot be solved from the repository (then set needs_human=true)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "root_cause": {"type": "string", "description": "The underlying cause, not just the symptom."},
                "evidence": {"type": "string", "description": "Concrete log lines, files, commits."},
                "proposed_fix": {"type": "string"},
                "needs_human": {"type": "boolean", "description": "True if a human must act (cannot be fixed from the repo, or too uncertain)."},
                "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
            },
            "required": ["root_cause", "evidence", "proposed_fix", "needs_human", "confidence"],
            "additionalProperties": False,
        },
    }},
]

WRITE_TOOLS = {"apply_fix"}          # tools with side effects
TERMINAL_TOOL = "submit_diagnosis"   # handled by the runtime (your loop), not by execute_tool


def schema_for(name: str) -> dict:
    """The JSON schema of a tool's arguments."""
    for t in TOOL_SCHEMAS:
        if t["function"]["name"] == name:
            return t["function"]["parameters"]
    raise ToolError(f"Unknown tool '{name}'. Available: {', '.join(t['function']['name'] for t in TOOL_SCHEMAS)}")


# ------------------------------------------------------------------ implementations (given)

def get_build_log(scenario: Scenario, tail_lines: int = 60) -> str:
    lines = scenario.build_log().splitlines()
    n = max(1, min(int(tail_lines), 400))
    shown = lines[-n:]
    header = f"[... {len(lines) - len(shown)} earlier lines omitted; call again with a larger tail_lines to see them ...]\n" \
        if len(shown) < len(lines) else ""
    return header + "\n".join(shown)


def list_files(scenario: Scenario) -> str:
    return "\n".join(scenario.repo_files())


# ------------------------------------------------------------------ implementations (TODO)

def read_file(scenario: Scenario, path: str) -> str:
    """0.2  Return the file's text.
    - get the absolute path with scenario.resolve(path)   (raises ToolError if it leaves the repo)
    - missing file or directory -> raise ToolError with a message the MODEL can act on
      (e.g. "File not found: x. Use list_files to see available files.")
    - files longer than ~6000 characters: truncate and say so (context engineering!)
    """
    raise NotImplementedError("TODO 0.2: read_file")


def git_log(scenario: Scenario, path: str | None = None, max_commits: int = 10) -> str:
    """0.3  Return recent commits as readable text, newest first.
    - scenario.git_log() gives a list of dicts: hash, date, author, message, files (list), diff
    - if `path` is given, keep only commits whose `files` contain it
    - include the diff: that is where the evidence is
    - no commits -> return a short message (that is a valid result, not an error)
    """
    raise NotImplementedError("TODO 0.3: git_log")


def apply_fix(scenario: Scenario, path: str, new_content: str) -> str:
    """0.4  Overwrite (or create) a repo file with new_content. Return a short confirmation.
    Use scenario.resolve(path). Do NOT check approval here - the runtime does that (step C).
    """
    raise NotImplementedError("TODO 0.4: apply_fix")


# ------------------------------------------------------------------ dispatch (given)

def execute_tool(name: str, args: dict, scenario: Scenario) -> str:
    """Run a tool and return its text output. Raises ToolError for problems the model should see."""
    if name == TERMINAL_TOOL:
        raise ToolError("submit_diagnosis is handled by the runtime, not executed as a tool")
    # Lenient on purpose: unknown argument names are silently dropped (like many real runtimes).
    # If you want strictness, call core.validate_args() in your loop BEFORE executing.
    known = schema_for(name).get("properties", {})
    args = {k: v for k, v in (args or {}).items() if k in known}
    if name == "get_build_log":
        return get_build_log(scenario, **args)
    if name == "list_files":
        return list_files(scenario)
    if name == "read_file":
        return read_file(scenario, **args)
    if name == "git_log":
        return git_log(scenario, **args)
    if name == "apply_fix":
        return apply_fix(scenario, **args)
    raise ToolError(f"Unknown tool '{name}'")
