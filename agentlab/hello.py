"""STEP 0 - One tool call, by hand, with every message printed. (Lecture slide 17: "Who does what?")

Run:   python -m agentlab hello
Read this file top to bottom while it runs. Nothing to edit.

What to look for:
  * the MODEL only produces a tool call (a request) - it does not run anything
  * the arguments arrive as a JSON *string* - your code has to parse them
  * YOUR CODE (the runtime) executes the tool and sends a role="tool" message back
  * the tool message must reference the tool call id
  * the model needs a second call to turn the result into an answer
  * gpt-oss also returns its reasoning in a separate field; we keep it in the history
"""
from __future__ import annotations

import json

from .core import Scenario, message_to_dict, parse_arguments, tool_message
from .llm import get_llm
from .tools import TOOL_SCHEMAS, execute_tool


def show(title: str, obj) -> None:
    print(f"\n----- {title} " + "-" * max(0, 60 - len(title)))
    print(json.dumps(obj, indent=2, ensure_ascii=False, default=str)[:3000])


def main() -> None:
    llm = get_llm()
    scenario = Scenario.load("s1_missing_dep")
    tools = [t for t in TOOL_SCHEMAS if t["function"]["name"] == "get_build_log"]   # only ONE tool on offer

    messages = [{"role": "user", "content": "Our CI build fails. What is the error at the end of the build log?"}]
    show("[1] user -> model", messages)

    resp = llm.chat(messages=messages, tools=tools)
    choice = resp.choices[0]
    show(f"[2] model -> runtime  (finish_reason={choice.finish_reason})", message_to_dict(choice.message))
    messages.append(message_to_dict(choice.message))

    if not choice.message.tool_calls:
        print("\nThe model answered without using the tool. Run again - models are not deterministic.")
        return
    for tc in choice.message.tool_calls:
        args = parse_arguments(tc)                     # '{"tail_lines": 60}'  ->  {"tail_lines": 60}
        print(f"\n[3] runtime executes {tc.function.name}({args})   <- this is YOUR code, not the model")
        output = execute_tool(tc.function.name, args, scenario)
        messages.append(tool_message(tc, output))      # one role="tool" message per tool call
    show("[4] runtime -> model (role=tool)", [m for m in messages if m["role"] == "tool"])

    resp = llm.chat(messages=messages, tools=tools)
    choice = resp.choices[0]
    show(f"[5] model -> user  (finish_reason={choice.finish_reason})", message_to_dict(choice.message))
    print(f"\nLLM calls: {llm.calls}   tokens in/out: {llm.input_tokens}/{llm.output_tokens}   "
          f"model time: {llm.seconds:.1f}s")


if __name__ == "__main__":
    main()
