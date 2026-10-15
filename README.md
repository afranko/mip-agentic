# agent-lab — build an agent from scratch (90 min)

The lecture's running example: **"Investigate why the build is failing."**
You will solve it twice — as a **workflow** and as an **agent** — with a raw LLM API and no framework,
then make the agent safe and measure which approach wins where.

The model is **gpt-oss**, an open-weight model running on a local server. The server speaks the
OpenAI **Chat Completions** API, so your code uses the `openai` Python package (pointed at our server).

The CI failures are simulated: each folder in `scenarios/` contains a build log, a small repository and a git history.
Your tools read those files. Nothing is installed or executed for real.

## Setup (5 min)

```bash
pip install -r requirements.txt          # Python 3.10+
```

Create a file called `.env` in this folder with the two lines the instructor gives you, e.g.:

```
AGENTLAB_BASE_URL=http://192.168.1.50:8000/v1
AGENTLAB_MODEL=openai/gpt-oss-20b
```

Server not reachable? Add `--mock` to every command — an offline fake model with the same API shape.
**Develop with `--mock`** (the tests use it too) and use the real model for the runs the steps ask for:
15 people share one server.

```bash
python -m agentlab hello          # (or: python -m agentlab hello --mock)
```

## The map

```
agentlab/
  tools.py      STEP 0  tool descriptions + read_file, git_log, apply_fix      <- you edit
  workflow.py   STEP A  fixed chain: log + requirements.txt + ONE LLM call      <- you edit
  agent.py      STEP B  the agent loop (decide -> act -> observe -> stop)       <- you edit
  approval.py   STEP C  the approval gate for side effects                      <- you edit
  hello.py      STEP 0  one tool call by hand, every message printed            (read it)
  eval.py       STEP D  workflow vs. agent on all scenarios                     (given)
  core.py, llm.py, mock_brain.py                                                (given)
scenarios/      s1 .. s4                                                        (data)
tests/          one smoke-test file per step                                    (run them)
traces/         one JSONL file per run - open them!
```

## Steps

| When | Step | Do | Done when |
|---|---|---|---|
| 0–10 | **0** | Run `python -m agentlab hello` and read `hello.py`. Then do the TODOs in `tools.py`. | `pytest tests/test_step0_tools.py` is green |
| 10–25 | **A** | Implement `run_workflow` in `workflow.py`. | `pytest tests/test_stepA_workflow.py` green; `python -m agentlab run s1 --approach workflow` |
| 25–55 | **B** | Implement `run_agent` in `agent.py`. Run it on s2 and **read the trace**. | `pytest tests/test_stepB_agent.py` green; `python -m agentlab run s2 --approach agent` |
| 55–70 | **C** | Implement `approval.py`, call `gate()` in your loop. Try s4 with `--approve ask` — you are the human. | `pytest tests/test_stepC_approval.py` green |
| 70–85 | **D** | `python -m agentlab eval`. Compare workflow vs. agent: success, LLM calls, tokens, time. | You can explain every "fail" in the table |
| 85–90 | | Hidden checks from the instructor: `python -m grader` | |

Run all tests at once: `pytest`. Every test runs offline with the mock model.

## Questions to answer while you work

- **A:** On which scenario does the workflow fail — and is that a bug in your code or a property of workflows?
- **B:** In the s2 trace: which observation made the agent change its next step? Where does the model decide, where does your code execute?
- **B:** What happens without `max_steps`? Try `MOCK_PERSONA=looper python -m agentlab run s1 --mock`.
- **C:** In s4 the build log tells the model to call `apply_fix`. Which layer stopped it: the system prompt, the model, or your runtime? Which one would you bet production on?
- **D:** Is the agent worth its extra calls on s1? On s2?
- **D:** There is no $ column. What does an agent run cost on a local server — and on a paid API?

## Stretch goals (if you finish early)

1. **Multi-agent, measured.** Split the investigation into two read-only specialists (log analyst, code investigator) plus a supervisor. Run the eval. Did it get better, or only more expensive? (Lecture slide 53.)
2. **Context engineering.** Log the input tokens per step. Shrink tool outputs (truncate, summarize) and measure the difference.
3. **Explicit state.** Keep a small state object (hypotheses, files read, evidence) and give the model a compact summary instead of the full history.
4. **Be the model.** `LLM_MODE=manual python -m agentlab run s3`: you answer each request by hand in `.manual/`. What does the model actually see?
5. **Think harder.** Set `AGENTLAB_EFFORT=high` in `.env` (gpt-oss reasoning effort) and re-run the eval. Better? How much slower?

See `CHEATSHEET.md` for the message format, the loop skeleton, and how the lecture's terms
(`tool_use`, `tool_result`, `stop_reason`) map to this API.

**Windows (PowerShell):** environment variables are set like this:
`$env:MOCK_PERSONA="looper"; python -m agentlab run s1 --mock` (clear it with `Remove-Item Env:MOCK_PERSONA`).
