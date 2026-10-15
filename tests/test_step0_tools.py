"""Step 0 - tools.   pytest tests/test_step0_tools.py"""
import pytest

from agentlab.core import ToolError
from agentlab.tools import TOOL_SCHEMAS, execute_tool


def test_descriptions_written():
    todo = [t["function"]["name"] for t in TOOL_SCHEMAS if "TODO" in t["function"]["description"]]
    assert not todo, f"Write a real description for: {todo}  (say WHEN to use the tool)"


def test_get_build_log_tail(scenario):
    out = execute_tool("get_build_log", {"tail_lines": 5}, scenario("s1"))
    assert "earlier lines omitted" in out and "exit code 2" in out


def test_read_file(scenario):
    out = execute_tool("read_file", {"path": "requirements.txt"}, scenario("s1"))
    assert "requests==" in out


def test_read_file_missing_is_tool_error(scenario):
    with pytest.raises(ToolError):
        execute_tool("read_file", {"path": "no/such/file.txt"}, scenario("s1"))


@pytest.mark.parametrize("path", ["../build.log", "../../../etc/passwd", "/etc/passwd"])
def test_read_file_stays_in_repo(scenario, path):
    with pytest.raises(ToolError):
        execute_tool("read_file", {"path": path}, scenario("s1"))


def test_git_log_filters_by_path(scenario):
    out = execute_tool("git_log", {"path": "requirements.txt"}, scenario("s1"))
    assert "remove unused dependencies" in out
    assert "docs: update README" not in out, "commits not touching the file should be filtered out"
    assert "PyYAML" in out, "include the diff - that is the evidence"


def test_git_log_no_match_is_not_an_error(scenario):
    out = execute_tool("git_log", {"path": "does-not-exist.txt"}, scenario("s1"))
    assert isinstance(out, str) and out


def test_apply_fix_writes_only_the_workspace(scenario):
    sc = scenario("s1")
    execute_tool("apply_fix", {"path": "requirements.txt", "new_content": "PyYAML==6.0.2\n"}, sc)
    assert (sc.repo / "requirements.txt").read_text() == "PyYAML==6.0.2\n"
    assert "PyYAML" not in (sc.source / "repo" / "requirements.txt").read_text(), "the original must stay untouched"
