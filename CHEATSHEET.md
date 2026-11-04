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
- Append the assistant message with `messages.append(message_to_dict(msg))` — it keeps the tool calls
  (and, on local gpt-oss servers, the model's `reasoning` text, which that model needs in the next call).
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
Azure enforces this. (Some local gpt-oss servers cannot: then `llm.py` asks in words and, if needed, nudges the
model once — a second LLM call and an `[llm]` line.) Your code must still handle "no tool call".

## Lecture terms → this API

| Lecture (Anthropic API) | Here (OpenAI Chat Completions) |
|---|---|
| `tool_use` block, `input` (object) | `message.tool_calls[i]`, `function.arguments` (JSON string) |
| `tool_result` block in a user message | `{"role": "tool", "tool_call_id": ..., "content": ...}`, one per call |
| `is_error: true` | no such field: the text says `ERROR: ...` |
| `stop_reason`: `tool_use` / `end_turn` / `max_tokens` | `finish_reason`: `tool_calls` / `stop` / `length` |
| `system=` parameter | first message with `role: "system"` |
| `thinking` block | not returned by Azure (local gpt-oss: `reasoning` field) |
| `max_tokens` | `max_completion_tokens` for GPT-5.x (`llm.py` handles it) |

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

`gpt-5.4-mini`: $0.75 / 1M input, $4.50 / 1M output tokens (reasoning tokens count as output).
One agent run ≈ $0.01–0.03. Each run prints its cost; a process stops itself at $0.50 (`AGENTLAB_MAX_COST_USD`)
or 150 LLM calls. `--mock` is free and shows what the run would have cost.
`[llm] rate limited (429)`: the shared per-minute quota is used up — the client waits and retries.

## Settings (`.env`)

| Variable | Meaning |
|---|---|
| `AGENTLAB_BASE_URL` | Azure: `https://<resource>.openai.azure.com/openai/v1/` (local server: `http://host:8000/v1`) |
| `AGENTLAB_MODEL` | Azure: the deployment name, e.g. `gpt-5.4-mini` |
| `AGENTLAB_API_KEY` | the key |
| `AGENTLAB_EFFORT` | reasoning effort: `none`, `low` (default), `medium`, `high` |
| `LLM_MODE` | `real`, `mock`, `manual` |
