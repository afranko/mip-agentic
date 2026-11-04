"""LLM access. You do NOT need to edit this file.

    llm = get_llm()          # picks the mode from the environment (see below)
    resp = llm.chat(messages=[...], tools=[...], tool_choice=None)

The response has the shape of the OpenAI Chat Completions API (the openai Python SDK object in real mode):
    choice = resp.choices[0]
    choice.finish_reason          -> "tool_calls" | "stop" | "length"
    choice.message.content        -> text or None
    choice.message.tool_calls     -> None or [tool_call]; tool_call.id, tool_call.function.name,
                                     tool_call.function.arguments  (a JSON *string*)
    resp.usage.prompt_tokens / resp.usage.completion_tokens

Modes (environment variable LLM_MODE):
    real    any OpenAI-compatible Chat Completions endpoint: Azure OpenAI (e.g. gpt-5.4-mini), OpenAI,
            or a local server (vLLM, Ollama, llama.cpp, LM Studio ...) running gpt-oss.
            Default when AGENTLAB_BASE_URL is set.
    mock    Offline rule-based fake model with the same response shape. Default otherwise.
            MOCK_PERSONA = good | gullible | gullible_plus | looper | sloppy
    manual  YOU are the model: requests are written to .manual/, you write the responses.

Settings (environment or the .env file in the repo root):
    AGENTLAB_BASE_URL     Azure:  https://<resource>.openai.azure.com/openai/v1/
                          vLLM:   http://192.168.1.50:8000/v1        Ollama: http://localhost:11434/v1
    AGENTLAB_MODEL        Azure: the DEPLOYMENT name (e.g. gpt-5.4-mini)
                          vLLM: openai/gpt-oss-20b    Ollama: gpt-oss:20b
    AGENTLAB_API_KEY      Azure / OpenAI: the API key.  Local servers: only if started with a key.
    AGENTLAB_EFFORT       reasoning effort: none | low | medium | high (default low)
    AGENTLAB_TIMEOUT      seconds per request (default 180)
    AGENTLAB_MAX_CALLS    LLM calls per process before it stops itself (default 150)
    AGENTLAB_MAX_COST_USD dollars per process before it stops itself (default 0.50; paid APIs only)
    AGENTLAB_PRICE_IN / AGENTLAB_PRICE_OUT   USD per 1M tokens, if the model is not in PRICES below
    AGENTLAB_PROVIDER     azure | openai | local   (default: guessed from the URL)
    AGENTLAB_TOOL_CHOICE  native | emulate   (default: native on Azure/OpenAI, emulate on local servers)
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Any

from .core import ROOT, message_to_dict


def _load_dotenv() -> None:
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()


# What we learned about the server in this process (so each run does not hit the same 400 again).
_LEARNED: dict[str, bool] = {}

# USD per 1M tokens (input, output). Azure Global Standard / OpenAI list price, Nov 2026.
# On Azure the model name is YOUR deployment name - if it differs, set AGENTLAB_PRICE_IN / _OUT.
PRICES = {"gpt-5.4-mini": (0.75, 4.50), "gpt-5.4-nano": (0.20, 1.25)}
HOSTED_FALLBACK_PRICE = (2.00, 10.00)   # unknown model on a paid API: a deliberately high guess


def provider_for(base_url: str) -> str:
    forced = os.environ.get("AGENTLAB_PROVIDER", "").strip().lower()
    if forced in ("azure", "openai", "local"):
        return forced
    u = base_url.lower()
    if "azure.com" in u or "azure-api.net" in u:
        return "azure"
    if "api.openai.com" in u:
        return "openai"
    return "local"


def price_for(model: str, provider: str) -> tuple[float, float]:
    pin, pout = os.environ.get("AGENTLAB_PRICE_IN"), os.environ.get("AGENTLAB_PRICE_OUT")
    if pin and pout:
        return float(pin), float(pout)
    for name, price in PRICES.items():
        if name in model.lower():
            return price
    return HOSTED_FALLBACK_PRICE if provider in ("azure", "openai") else (0.0, 0.0)


class BudgetExceeded(RuntimeError):
    """Raised before a call that would go over AGENTLAB_MAX_CALLS or AGENTLAB_MAX_COST_USD."""


class ManualPending(SystemExit):
    """Manual mode: a request is waiting for you to write the response file."""


# ---------------------------------------------------------------------------- response objects
# Same attribute names as the openai SDK, so your code works with every mode.

@dataclass
class Function:
    name: str
    arguments: str          # JSON string, exactly like the real API


@dataclass
class ToolCall:
    id: str
    function: Function
    type: str = "function"


@dataclass
class Message:
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    role: str = "assistant"
    reasoning: str | None = None      # gpt-oss style: the reasoning text, separate from the answer


@dataclass
class Choice:
    message: Message
    finish_reason: str
    index: int = 0


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class ChatResponse:
    choices: list[Choice]
    usage: Usage = field(default_factory=Usage)
    model: str = "mock"


def make_response(content: str | None, calls: list[tuple[str, dict]], finish: str, prefix: str,
                  usage: Usage, model: str, reasoning: str | None = None) -> ChatResponse:
    tcs = [ToolCall(id=f"call_{prefix}_{j}",
                    function=Function(name=n, arguments=a if isinstance(a, str) else json.dumps(a)))
           for j, (n, a) in enumerate(calls)] or None
    return ChatResponse(choices=[Choice(Message(content=content, tool_calls=tcs, reasoning=reasoning), finish)],
                        usage=usage, model=model)


def normalize_messages(messages: list[Any]) -> list[dict]:
    """Messages (dicts, SDK objects or mock objects) -> plain JSON-able dicts."""
    out = []
    for m in messages:
        if isinstance(m, dict) and m.get("role") != "assistant":
            out.append(dict(m))
        else:
            out.append(message_to_dict(m))
    return out


def _estimate_tokens(obj: Any) -> int:
    return len(json.dumps(obj, default=str, ensure_ascii=False)) // 4


# ---------------------------------------------------------------------------- base

class BaseLLM:
    model = "mock"
    price = (0.0, 0.0)        # USD per 1M tokens (input, output); 0 = free (mock, manual, local server)

    def __init__(self, max_calls: int | None = None, max_cost_usd: float | None = None):
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.seconds = 0.0
        self.max_calls = max_calls if max_calls is not None else int(os.environ.get("AGENTLAB_MAX_CALLS", "150"))
        self.max_cost_usd = max_cost_usd if max_cost_usd is not None else float(
            os.environ.get("AGENTLAB_MAX_COST_USD", "0.50"))

    @property
    def cost_usd(self) -> float:
        return (self.input_tokens * self.price[0] + self.output_tokens * self.price[1]) / 1_000_000

    def chat(self, *, messages: list[Any], tools: list[dict] | None = None,
             tool_choice: Any = None, max_tokens: int = 4096) -> Any:
        if self.calls >= self.max_calls:
            raise BudgetExceeded(f"call limit {self.max_calls} reached (AGENTLAB_MAX_CALLS)")
        if self.price != (0.0, 0.0) and self.cost_usd >= self.max_cost_usd:
            raise BudgetExceeded(f"cost limit ${self.max_cost_usd:.2f} reached (AGENTLAB_MAX_COST_USD)")
        t0 = time.monotonic()
        resp = self._chat(messages=normalize_messages(messages), tools=tools or [], tool_choice=tool_choice,
                          max_tokens=max_tokens)
        self.seconds += time.monotonic() - t0
        self.calls += 1
        self.input_tokens += resp.usage.prompt_tokens
        self.output_tokens += resp.usage.completion_tokens
        return resp

    def _chat(self, **kw) -> Any:  # pragma: no cover
        raise NotImplementedError


# ---------------------------------------------------------------------------- real

def _forced_name(tool_choice: Any) -> str | None:
    if isinstance(tool_choice, dict):
        return (tool_choice.get("function") or {}).get("name") or tool_choice.get("name")
    return None


def _clean_name(name: str, known: set[str]) -> str:
    """gpt-oss sometimes leaks its internal format into the function name
    ("functions.read_file", "read_file<|channel|>commentary"). Repair it if it maps to a known tool."""
    if name in known:
        return name
    cand = re.split(r"<\||\s", name.strip())[0]
    cand = cand.split(".")[-1] if cand.startswith("functions.") else cand
    return cand if cand in known else name


class RealLLM(BaseLLM):
    def __init__(self, model: str | None = None, **kw):
        super().__init__(**kw)
        try:
            import openai
        except ImportError:
            sys.exit("The 'openai' package is missing. Run: pip install -r requirements.txt")
        self.base_url = os.environ.get("AGENTLAB_BASE_URL", "").strip()
        if not self.base_url:
            sys.exit("AGENTLAB_BASE_URL is not set. Put it in a .env file in the repo root "
                     "(e.g. AGENTLAB_BASE_URL=http://192.168.1.50:8000/v1), or run with --mock to work offline.")
        self._openai = openai
        self.provider = provider_for(self.base_url)
        hosted = self.provider in ("azure", "openai")
        key = os.environ.get("AGENTLAB_API_KEY") or ("" if hosted else "local")
        if hosted and not key:
            sys.exit("AGENTLAB_API_KEY is not set. Put the key from the instructor in the .env file, "
                     "or run with --mock to work offline.")
        # Paid APIs answer "429 rate limit" when many people share one key: retry patiently.
        self.client = openai.OpenAI(base_url=self.base_url, api_key=key, max_retries=8 if hosted else 3,
                                    timeout=float(os.environ.get("AGENTLAB_TIMEOUT", "180")))
        self.model = model or os.environ.get("AGENTLAB_MODEL", "gpt-5.4-mini" if hosted else "openai/gpt-oss-20b")
        self.price = price_for(self.model, self.provider)
        if hosted and self.price == HOSTED_FALLBACK_PRICE and not _LEARNED.get("warned_price"):
            _LEARNED["warned_price"] = True
            print(f"[llm] no price known for '{self.model}'; estimating with ${self.price[0]}/${self.price[1]} per 1M "
                  "tokens (set AGENTLAB_PRICE_IN / AGENTLAB_PRICE_OUT)", file=sys.stderr)
        eff = os.environ.get("AGENTLAB_EFFORT", "low").strip().lower()
        self.effort = None if eff in ("", "off") or _LEARNED.get("no_effort") else eff
        if self.effort == "none" and not hosted:
            self.effort = None                      # gpt-oss has no "none"; just do not send it
        self.tool_choice_mode = "emulate" if _LEARNED.get("emulate") else \
            os.environ.get("AGENTLAB_TOOL_CHOICE", "native" if hosted else "emulate").strip().lower()
        # Hosted APIs never return reasoning text and may reject unknown fields -> do not send them back.
        self.send_reasoning = not hosted and not _LEARNED.get("no_reasoning")
        self.send_tool_name = not hosted            # "name" on tool messages: only local servers use it
        # Newer OpenAI / Azure models (GPT-5.x) reject max_tokens and want max_completion_tokens.
        self.tok_param = _LEARNED.get("tok_param") or ("max_completion_tokens" if hosted else "max_tokens")
        self.repairs = 0          # how often we had to repair a broken tool name / tool choice

    # -- one HTTP request, with the compatibility fallbacks local servers need
    def _create(self, kw: dict) -> Any:
        oa = self._openai
        for _ in range(4):
            try:
                return self.client.chat.completions.create(**kw)
            except oa.BadRequestError as e:
                msg = str(e).lower()
                if _is_content_filter(e):
                    return self._filtered(kw, e)
                other = {"max_tokens": "max_completion_tokens", "max_completion_tokens": "max_tokens"}
                cur = next((k for k in other if k in kw), None)
                if cur and cur in msg and other[cur] not in kw:
                    print(f"[llm] server rejected {cur}; using {other[cur]}", file=sys.stderr)
                    kw[other[cur]] = kw.pop(cur)
                    self.tok_param = _LEARNED["tok_param"] = other[cur]
                    continue
                if "system" in msg and "role" in msg and any(m.get("role") == "system" for m in kw["messages"]):
                    print("[llm] server rejected role 'system'; sending it as 'developer'", file=sys.stderr)
                    kw["messages"] = [{**m, "role": "developer"} if m.get("role") == "system" else m
                                      for m in kw["messages"]]
                    continue
                if "reasoning_effort" in kw and "effort" in msg:
                    print("[llm] server rejected reasoning_effort; sending without it", file=sys.stderr)
                    self.effort = None
                    _LEARNED["no_effort"] = True
                    kw.pop("reasoning_effort")
                    continue
                if self.send_reasoning and any(k in m for m in kw["messages"]
                                               for k in ("reasoning", "reasoning_content", "thinking")):
                    print("[llm] server rejected reasoning fields in the history; dropping them", file=sys.stderr)
                    self.send_reasoning = False
                    _LEARNED["no_reasoning"] = True
                    kw["messages"] = _strip_reasoning(kw["messages"])
                    continue
                if "tool_choice" in kw and "tool_choice" in msg:
                    print("[llm] server rejected tool_choice; emulating it", file=sys.stderr)
                    self.tool_choice_mode = "emulate"
                    _LEARNED["emulate"] = True
                    kw.pop("tool_choice")
                    forced = _forced_name(kw.get("tool_choice")) or self._forced
                    if forced:
                        kw["messages"] = _with_rule(kw["messages"], forced)
                    continue
                raise
            except (oa.AuthenticationError, oa.PermissionDeniedError):
                if self.provider == "azure":
                    sys.exit("Azure rejected the API key (401/403). Check AGENTLAB_API_KEY in .env "
                             "(Azure portal -> your resource -> Keys and Endpoint).")
                sys.exit("The server rejected the API key (401). Set AGENTLAB_API_KEY in .env to the key "
                         "the server was started with.")
            except oa.NotFoundError as e:
                if self.provider == "azure":
                    sys.exit(f"404 from Azure: {e}\nAGENTLAB_MODEL must be the DEPLOYMENT name (Foundry -> Deployments), "
                             f"now '{self.model}'. AGENTLAB_BASE_URL must end in /openai/v1/ , now {self.base_url}")
                sys.exit(f"404 from the server: {e}\nIs AGENTLAB_BASE_URL right (it usually ends in /v1) and does the "
                         f"server know the model '{self.model}'? Check with:  curl {self.base_url.rstrip('/')}/models")
            except oa.RateLimitError as e:
                # The SDK already retried with backoff. Wait a bit longer, then give up on this run.
                waited = getattr(self, "_rl_waits", 0)
                if waited < 3:
                    self._rl_waits = waited + 1
                    print(f"[llm] rate limited (429) - many people share the quota; waiting 20 s", file=sys.stderr)
                    time.sleep(20)
                    continue
                raise RuntimeError(f"rate limit (429) persists - try again in a minute, or use --mock. {e}") from None
            except oa.APIConnectionError:
                sys.exit(f"Cannot reach the model server at {self.base_url} (connection failed or timed out).\n"
                         "Same network / VPN? Is the server running? Or work offline with --mock.")
        raise RuntimeError("request kept failing after compatibility fallbacks")

    def _chat(self, *, messages, tools, tool_choice, max_tokens):
        if not self.send_reasoning:
            messages = _strip_reasoning(messages)
        if not self.send_tool_name:
            messages = [{k: v for k, v in m.items() if not (m.get("role") == "tool" and k == "name")} for m in messages]
        self._rl_waits = 0
        kw: dict[str, Any] = {"model": self.model, "messages": messages, self.tok_param: max_tokens}
        if tools:
            kw["tools"] = tools
        if self.effort:
            kw["reasoning_effort"] = self.effort
        forced = self._forced = _forced_name(tool_choice)
        if tool_choice and not (forced and self.tool_choice_mode == "emulate"):
            kw["tool_choice"] = tool_choice
        if forced and self.tool_choice_mode == "emulate":
            # Many local servers ignore or reject a NAMED tool_choice for gpt-oss (vLLM: only "auto";
            # Ollama: not supported). We ask for it in words and check the answer instead.
            kw["messages"] = _with_rule(messages, forced)
        resp = self._create(kw)
        known = {t["function"]["name"] for t in tools}
        if resp.choices[0].finish_reason == "content_filter":
            return resp
        if forced and not _has_call(resp, forced, known):
            # One retry with a nudge (counts as an extra LLM call - visible in the numbers).
            self.repairs += 1
            print(f"[llm] the model did not call {forced}; asking once more", file=sys.stderr)
            self.calls += 1
            self._add_usage(resp, messages)
            prev = message_to_dict(resp.choices[0].message)
            prev["content"] = prev.get("content") or ""
            kw["messages"] = kw["messages"] + [prev, {"role": "user", "content":
                                                      f"Call the function `{forced}` now with your answer."}]
            resp = self._create(kw)
        self._repair_names(resp, known)
        if tools and looks_like_unparsed_tool_call(resp.choices[0].message) and not _LEARNED.get("warned_parser"):
            _LEARNED["warned_parser"] = True
            print(PARSER_WARNING, file=sys.stderr)
        if not resp.usage or not (resp.usage.prompt_tokens or resp.usage.completion_tokens):
            # some servers report no / zero usage -> rough estimate, so the numbers are not empty
            resp.usage = Usage(_estimate_tokens(messages) + _estimate_tokens(tools),
                               _estimate_tokens(message_to_dict(resp.choices[0].message)))
        return resp

    def _filtered(self, kw: dict, err: Exception) -> Any:
        """Azure's content filter refused the PROMPT (HTTP 400). Turn it into an answer the loop can handle."""
        print("[llm] Azure content filter blocked this request (often: a prompt injection in the data). "
              "Returned as finish_reason='content_filter'.", file=sys.stderr)
        return make_response("[blocked by the Azure content filter - the request was not processed]", [],
                             "content_filter", "cf", Usage(_estimate_tokens(kw["messages"]), 0), self.model)

    def _add_usage(self, resp, messages):
        u = resp.usage
        if u and (u.prompt_tokens or u.completion_tokens):
            self.input_tokens += u.prompt_tokens
            self.output_tokens += u.completion_tokens
        else:
            self.input_tokens += _estimate_tokens(messages)

    def _repair_names(self, resp, known: set[str]) -> None:
        for tc in resp.choices[0].message.tool_calls or []:
            fixed = _clean_name(tc.function.name, known)
            if fixed != tc.function.name:
                print(f"[llm] repaired tool name {tc.function.name!r} -> {fixed!r}", file=sys.stderr)
                self.repairs += 1
                tc.function.name = fixed


def _with_rule(messages: list[dict], forced: str) -> list[dict]:
    """Ask for the function in words. Added to the FIRST system message: some chat templates reject a
    system message later in the conversation."""
    rule = f"Respond ONLY by calling the function `{forced}`. Do not answer in text."
    if messages and messages[0].get("role") == "system":
        return [{**messages[0], "content": f"{messages[0].get('content') or ''}\n\n{rule}"}] + messages[1:]
    return [{"role": "system", "content": rule}] + messages


PARSER_WARNING = (
    "[llm] WARNING: the model wrote a tool call as plain TEXT (JSON in `content`, no `tool_calls`).\n"
    "      The SERVER is not parsing gpt-oss tool calls - this is a server setting, not your code.\n"
    "      vLLM: restart with   --tool-call-parser openai --enable-auto-tool-choice\n"
    "      llama.cpp: start llama-server with --jinja.   Ollama / LM Studio: update to a recent version.\n"
    "      Until it is fixed: work with --mock.")


def looks_like_unparsed_tool_call(msg: Any) -> bool:
    """No tool_calls, but the text is a bare JSON object (or contains harmony markers) -> the server did not
    turn the model's tool call into a tool_calls entry."""
    if getattr(msg, "tool_calls", None):
        return False
    text = (getattr(msg, "content", None) or "").strip()
    if "<|call|>" in text or "<|channel|>" in text or "to=functions." in text:
        return True
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    if not (text.startswith("{") and text.endswith("}")):
        return False
    try:
        return isinstance(json.loads(text), dict)
    except json.JSONDecodeError:
        return False


def _has_call(resp, name: str, known: set[str]) -> bool:
    return any(_clean_name(tc.function.name, known) == name for tc in resp.choices[0].message.tool_calls or [])


def _is_content_filter(e: Exception) -> bool:
    body = getattr(e, "body", None)
    code = ""
    if isinstance(body, dict):
        code = str(body.get("code") or (body.get("error") or {}).get("code") or "")
    return code == "content_filter" or "content_filter" in str(e) or "content management policy" in str(e)


def _strip_reasoning(messages: list[dict]) -> list[dict]:
    return [{k: v for k, v in m.items() if k not in ("reasoning", "reasoning_content", "thinking")} for m in messages]


def _offline_price() -> tuple[float, float]:
    """Mock / manual mode: show what the run WOULD cost on the configured paid model (default gpt-5.4-mini).
    With a local server configured, there is no price."""
    base = os.environ.get("AGENTLAB_BASE_URL", "")
    if base and provider_for(base) == "local":
        return (0.0, 0.0)
    return price_for(os.environ.get("AGENTLAB_MODEL") or "gpt-5.4-mini", "azure")


# ---------------------------------------------------------------------------- mock

class MockLLM(BaseLLM):
    """Deterministic offline model. Reacts to the conversation, not a canned script."""

    model = "mock"
    LOOP_GUARD = 60

    def __init__(self, persona: str | None = None, **kw):
        super().__init__(**kw)
        self.persona = persona or os.environ.get("MOCK_PERSONA", "good")
        self.price = _offline_price()

    def _chat(self, *, messages, tools, tool_choice, max_tokens):
        if self.calls >= self.LOOP_GUARD:
            raise RuntimeError(f"MockLLM loop guard: {self.LOOP_GUARD} calls - your loop never stops")
        from .mock_brain import respond
        content, calls, reasoning = respond(messages, tools, tool_choice, self.persona)
        finish = "tool_calls" if calls else "stop"
        usage = Usage(_estimate_tokens(messages) + _estimate_tokens(tools),
                      _estimate_tokens([content, calls, reasoning]))
        return make_response(content, calls, finish, f"mock{self.calls}", usage, "mock", reasoning)


# ---------------------------------------------------------------------------- manual

class ManualLLM(BaseLLM):
    """You play the model. Call i writes .manual/<session>/req_i.json and stops; you write resp_i.json and re-run
    the same command. Earlier responses are replayed, so the run continues where it stopped.

    resp_i.json:  {"content": "optional text",
                   "tool_calls": [{"name": "read_file", "arguments": {"path": "requirements.txt"}}]}
    (no tool_calls = a final text answer)
    """

    model = "manual"

    def __init__(self, session: str | None = None, **kw):
        super().__init__(**kw)
        self.price = _offline_price()
        self.dir = ROOT / ".manual" / (session or os.environ.get("MANUAL_SESSION", "default"))
        self.dir.mkdir(parents=True, exist_ok=True)

    def _chat(self, *, messages, tools, tool_choice, max_tokens):
        i = self.calls
        resp_f, req_f = self.dir / f"resp_{i:03d}.json", self.dir / f"req_{i:03d}.json"
        if resp_f.exists():
            data = json.loads(resp_f.read_text(encoding="utf-8"))
            calls = [(c["name"], c.get("arguments") or {}) for c in data.get("tool_calls") or []]
            usage = Usage(_estimate_tokens(messages), _estimate_tokens(data))
            return make_response(data.get("content"), calls, "tool_calls" if calls else "stop", f"m{i}", usage, "manual")
        req_f.write_text(json.dumps({"tool_choice": tool_choice,
                                     "tools": [{"name": t["function"]["name"],
                                                "description": t["function"].get("description", ""),
                                                "parameters": t["function"].get("parameters")} for t in tools],
                                     "messages": messages}, indent=2, ensure_ascii=False, default=str),
                         encoding="utf-8")
        raise ManualPending(f"[manual] wrote {req_f}. Write {resp_f.name} next to it, then re-run.\n"
                            '  format: {"content": "...", "tool_calls": [{"name": "...", "arguments": {...}}]}')


# ---------------------------------------------------------------------------- factory

_BANNER = False


def get_llm(mode: str | None = None, **kw) -> BaseLLM:
    mode = (mode or os.environ.get("LLM_MODE") or ("real" if os.environ.get("AGENTLAB_BASE_URL") else "mock")).lower()
    if mode == "real":
        return RealLLM(**kw)
    if mode == "mock":
        llm = MockLLM(**kw)
        global _BANNER
        if not _BANNER:
            print(f"*** MOCK MODE (persona={llm.persona}) - offline fake model, no server calls ***", file=sys.stderr)
            _BANNER = True
        return llm
    if mode == "manual":
        return ManualLLM(**kw)
    raise ValueError(f"Unknown LLM_MODE '{mode}' (use real, mock or manual)")
