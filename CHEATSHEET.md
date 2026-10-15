# Cheat sheet

## One tool round trip (lecture slide 17)

```
[1] system     {"role": "system", "content": SYSTEM}
    user       {"role": "user", "content": "Why does the build fail?"}
[2] assistant  msg = resp.choices[0].message
               msg.tool_calls = [ToolCall(id="call_1", function=Function(name="get_build_log",
                                                                         arguments='{"tail_lines": 60}'))]
               resp.choices[0].finish_reason == "tool_calls"          <- the MODEL only asks
[3] runtime    args = parse_arguments(tc)                              <- arguments is a JSON STRING
               output = execute_tool("get_build_log", args, scenario)  <- YOUR CODE executes
[4] tool       {"role": "tool", "tool_call_id": "call_1", "content": output}   = tool_message(tc, output)
[5] assistant  ... next decision, or the final answer (no tool_calls, finish_reason "stop")
```

Rules that bite:
- Append the assistant message with `messages.append(message_to_dict(msg))` — it keeps the tool calls **and** the
  model's reasoning (gpt-oss returns it in a separate `reasoning` field; the model needs it in the next call).
- Every tool call needs its own `role: "tool"` message with the same id, **right after** the assistant message.
  Two calls → two tool messages. A missing one → the server answers `400`.
- `function.arguments` is text the model wrote. It can be broken JSON → `parse_arguments()` raises `ToolError` →
  send that back as an error: `tool_message(tc, str(e), is_error=True)`. The model will fix it.
- Stop when the message has **no** `tool_calls`. Do not parse the text. (`finish_reason` tells you why, but some
  servers say `"stop"` even with tool calls — trust `msg.tool_calls`.)
- Tool errors are results too: `tool_message(tc, "File not found ...", is_error=True)` — the model can recover.

## The loop

```python
messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": TASK}]
for step in range(max_steps):
    resp = llm.chat(messages=messages, tools=TOOL_SCHEMAS)
    msg = resp.choices[0].message
    messages.append(message_to_dict(msg))
    if not msg.tool_calls:
        return ...end_turn...
    for tc in msg.tool_calls:
        name = tc.function.name
        args = parse_arguments(tc)              # try/except ToolError
        if name == "submit_diagnosis":
            return ...diagnosis...
        # gate -> execute_tool -> messages.append(tool_message(tc, output))
return ...max_steps...
```

## Asking for one specific tool (workflow)

```python
llm.chat(messages=..., tools=[SUBMIT_TOOL],
         tool_choice={"type": "function", "function": {"name": "submit_diagnosis"}})
```
Many local servers cannot enforce this for gpt-oss. `llm.py` then asks in words and, if the model still answers
in text, nudges it once (you will see a second LLM call and an `[llm]` line). Your code must still handle
"no tool call".

## Lecture terms → this API

| Lecture (Anthropic API) | Here (OpenAI Chat Completions) |
|---|---|
| `tool_use` block, `input` (object) | `message.tool_calls[i]`, `function.arguments` (JSON string) |
| `tool_result` block in a user message | `{"role": "tool", "tool_call_id": ..., "content": ...}`, one per call |
| `is_error: true` | no such field: the text says `ERROR: ...` |
| `stop_reason`: `tool_use` / `end_turn` / `max_tokens` | `finish_reason`: `tool_calls` / `stop` / `length` |
| `system=` parameter | first message with `role: "system"` |
| `thinking` block | `reasoning` / `reasoning_content` field |

## Commands

```bash
python -m agentlab run s2 --approach agent              # one run, prints the diagnosis + trace path
python -m agentlab run s4 --approach agent --approve ask
python -m agentlab eval
MOCK_PERSONA=gullible python -m agentlab run s4 --mock  # a fake model that obeys injections
LLM_MODE=manual python -m agentlab run s3               # you are the model
```

Personas for `--mock`: `good`, `gullible`, `gullible_plus`, `looper`, `sloppy` (sends broken JSON once).

## Reading a trace

`traces/*.jsonl`, one event per line: `start`, `llm_response`, `tool_call`, `tool_result`, `approval`, `stop`.
Ask: which observation changed the plan? Where did a tool fail and did the model recover? Why did it stop?

## Cost

The model runs on our own server: no money per token, but **time and tokens** are shared by everyone.
Each run prints its tokens and model time. A process stops itself after 150 LLM calls (`AGENTLAB_MAX_CALLS`).
On a paid API you would multiply the tokens by the price.

## Settings (`.env`)

| Variable | Meaning |
|---|---|
| `AGENTLAB_BASE_URL` | server address, ends in `/v1` |
| `AGENTLAB_MODEL` | model name the server knows (`openai/gpt-oss-20b` on vLLM, `gpt-oss:20b` on Ollama) |
| `AGENTLAB_EFFORT` | gpt-oss reasoning effort: `low` (default), `medium`, `high` |
| `LLM_MODE` | `real`, `mock`, `manual` |
