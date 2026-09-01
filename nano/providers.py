from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


from pydantic import BaseModel

_RETRYABLE_STATUS = {408, 429, 529}
# Plus every 5xx: a proxy in front of a slow model returns 504 (and Cloudflare
# 520-524) for exactly the blips 500/502/503 cover. The SDK maps all of them to
# one InternalServerError whose class name matches no substring check, so
# listing codes individually silently dropped the most likely gateway failure.


def _request_timeout(max_tokens: int) -> float:
    """Read timeout for one non-streaming completion.

    Nothing arrives until the whole body is ready, so the ceiling has to fit
    the largest generation the caller allows: at a conservative ~50 tok/s a
    16k-token write needs ~330s, and a fixed 120s made those fail on every
    retry, deterministically. Still bounded, because one stuck request must
    not eat a whole task's wall clock.
    """
    return float(min(600, max(120, 30 + max_tokens / 50)))


def _max_tokens_default() -> int:
    """Output-token ceiling, env-tunable (NANO_MAX_TOKENS). 8192 is safe
    everywhere; the benchmark gateway accepts far higher, and every file
    write past ~6KB at 8192 costs a whole continuation round trip."""
    try:
        return int(os.environ.get("NANO_MAX_TOKENS", 8192))
    except ValueError:
        return 8192


def _call_with_retry(fn, attempts: int = 3):
    """Retry transient API failures (rate limits, overload, dropped
    connections) with exponential backoff. Non-transient errors and the
    final attempt raise. One unlucky 429 must not zero out a whole task."""
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as e:
            status = getattr(e, "status_code", None)
            # Client-side timeouts (APITimeoutError) carry no status_code and
            # no "Connection" in their concrete class name - same transient
            # class of failure, same retry.
            name = type(e).__name__
            transient = (status in _RETRYABLE_STATUS
                         or (isinstance(status, int) and status >= 500)
                         or "Connection" in name or "Timeout" in name)
            if not transient or attempt == attempts - 1:
                raise
            time.sleep(2 ** attempt)


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any]


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int = 0


class StepResult(BaseModel):
    text: str | None
    tool_calls: list[ToolCall]
    stop_reason: str  # end_turn | tool_use | max_tokens
    usage: Usage


@runtime_checkable
class Provider(Protocol):
    model: str

    def step(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        system: str,
    ) -> StepResult: ...


def _ensure_block_list(content: Any) -> list[dict[str, Any]]:
    """Normalize a message's content to a list-of-blocks form so we can
    attach cache_control to the last block."""
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return [dict(b) for b in content]


@dataclass
class AnthropicProvider:
    model: str
    client: Any = None  # injectable for tests; defaults to anthropic.Anthropic()
    max_tokens: int = field(default_factory=_max_tokens_default)

    def __post_init__(self) -> None:
        if self.client is None:
            import anthropic
            # Timeout scaled to the output ceiling (see _request_timeout); the
            # SDK default of 600s x its own retries lets one stuck request eat
            # a whole task. max_retries=0 because _call_with_retry already
            # retries with backoff - stacking the two multiplies worst-case
            # wall clock by ~3x for no extra resilience.
            self.client = anthropic.Anthropic(
                timeout=_request_timeout(self.max_tokens), max_retries=0)

    def step(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        system: str,
    ) -> StepResult:
        sys_param = [{"type": "text", "text": system,
                      "cache_control": {"type": "ephemeral"}}]
        msgs = [{"role": m["role"], "content": _ensure_block_list(m["content"])}
                for m in messages]
        # Cache-mark the last user turn (spec §3.5: system + second-to-last user
        # turn — by the time step() runs, the "second-to-last" is the most recent
        # user message before the assistant turn we're about to generate).
        for m in reversed(msgs):
            if m["role"] == "user" and m["content"]:
                m["content"][-1]["cache_control"] = {"type": "ephemeral"}
                break

        resp = _call_with_retry(lambda: self.client.messages.create(
            model=self.model,
            system=sys_param,
            messages=msgs,
            tools=tools,
            max_tokens=self.max_tokens,
        ))

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in resp.content:
            btype = getattr(block, "type", None)
            if btype == "text":
                text_parts.append(block.text)
            elif btype == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.id, name=block.name, arguments=dict(block.input)))

        usage = Usage(
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            cache_read_tokens=getattr(resp.usage, "cache_read_input_tokens", 0) or 0,
        )

        return StepResult(
            text="\n".join(text_parts) if text_parts else None,
            tool_calls=tool_calls,
            stop_reason=resp.stop_reason,
            usage=usage,
        )


_OAI_FINISH_REASON = {
    "stop": "end_turn",
    "tool_calls": "tool_use",
    "length": "max_tokens",
}


def _normalize_for_openai(msg: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert one internal message into one or more OpenAI chat.completions
    messages. Assistant messages may carry both content blocks and tool_calls;
    user messages may carry tool_result blocks that must split into role='tool'
    messages, one per tool result."""
    role = msg["role"]
    content = msg.get("content")

    if role == "assistant":
        out: dict[str, Any] = {"role": "assistant"}
        if isinstance(content, list):
            out["content"] = "\n".join(
                b["text"] for b in content if b.get("type") == "text"
            ) or None
            # Derive tool_calls from the content blocks themselves - the single
            # source of truth. A separate copy would diverge when history is
            # mutated (e.g. a giant tool arg truncated), silently re-inflating
            # this request.
            tool_uses = [b for b in content if b.get("type") == "tool_use"]
            if tool_uses:
                out["tool_calls"] = [{
                    "id": b["id"],
                    "type": "function",
                    "function": {"name": b["name"],
                                 "arguments": json.dumps(b["input"])},
                } for b in tool_uses]
        else:
            out["content"] = content
        return [out]

    if role == "user" and isinstance(content, list) and any(
            b.get("type") == "tool_result" for b in content):
        # Split tool_result blocks into individual role="tool" messages.
        # Any plain text blocks become a separate role="user" message.
        out_msgs: list[dict[str, Any]] = []
        text_parts: list[str] = []
        for b in content:
            if b.get("type") == "tool_result":
                out_msgs.append({
                    "role": "tool",
                    "tool_call_id": b["tool_use_id"],
                    "content": b["content"],
                })
            elif b.get("type") == "text":
                text_parts.append(b["text"])
        if text_parts:
            # AFTER the tool messages: OpenAI requires role:"tool" replies to
            # immediately follow the assistant message that carried the
            # tool_calls; user text between them splits the pair and 400s.
            out_msgs.append({"role": "user", "content": "\n".join(text_parts)})
        return out_msgs

    return [msg]


@dataclass
class OpenAIProvider:
    model: str
    client: Any = None
    base_url: str | None = None
    max_completion_tokens: int = field(default_factory=_max_tokens_default)

    def __post_init__(self) -> None:
        if self.client is None:
            import openai
            # Timeout + retry policy: see AnthropicProvider.__post_init__.
            kw: dict[str, Any] = {
                "timeout": _request_timeout(self.max_completion_tokens),
                "max_retries": 0,
            }
            self.client = openai.OpenAI(base_url=self.base_url, **kw) \
                if self.base_url else openai.OpenAI(**kw)

    def step(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        system: str,
    ) -> StepResult:
        oai_messages: list[dict[str, Any]] = [
            {"role": "system", "content": system}
        ]
        for m in messages:
            oai_messages.extend(_normalize_for_openai(m))
        oai_tools = [{
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            },
        } for t in tools]

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": oai_messages,
            "max_completion_tokens": self.max_completion_tokens,
        }
        if oai_tools:
            kwargs["tools"] = oai_tools

        def _create():
            resp = self.client.chat.completions.create(**kwargs)
            # A flaky gateway can 200 with no choices; classify it as the
            # transient connection problem it is so the retry loop covers it.
            if not getattr(resp, "choices", None):
                raise ConnectionError("gateway returned no choices")
            return resp

        resp = _call_with_retry(_create)
        choice = resp.choices[0]
        msg = choice.message

        tool_calls: list[ToolCall] = []
        for tc in (msg.tool_calls or []):
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_raw": tc.function.arguments}
            # Valid JSON of the wrong shape (null, a list, a bare string) would
            # blow up ToolCall's dict field. Wrap it so dispatch can return a
            # fixable error instead of crashing the run.
            if not isinstance(args, dict):
                args = {"_raw": tc.function.arguments}
            # Some OpenAI-compatible gateways omit the id. It pairs the call
            # with its tool_result, so it must be unique across the WHOLE
            # conversation (an index restarts every response and collides
            # across turns) - and a missing one must not crash the run.
            call_id = tc.id or f"call_{uuid.uuid4().hex[:12]}"
            tool_calls.append(ToolCall(id=call_id, name=tc.function.name,
                                       arguments=args))

        # cached_tokens (when the gateway reports it) answers whether prompt
        # caching is happening at all on this path - the benchmark runs
        # entirely through here, and a zero here is a latency finding, not
        # cosmetics.
        details = getattr(resp.usage, "prompt_tokens_details", None)
        usage = Usage(
            input_tokens=getattr(resp.usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(resp.usage, "completion_tokens", 0) or 0,
            cache_read_tokens=getattr(details, "cached_tokens", 0) or 0,
        )

        return StepResult(
            text=msg.content,
            tool_calls=tool_calls,
            # A null finish_reason (flaky gateway turn) must still be a string:
            # StepResult.stop_reason is typed str, and a validation error here
            # ends the run before the agent's empty-turn guard can handle it.
            stop_reason=_OAI_FINISH_REASON.get(choice.finish_reason,
                                               choice.finish_reason or "unknown"),
            usage=usage,
        )
