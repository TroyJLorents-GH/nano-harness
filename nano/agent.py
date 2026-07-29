from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .providers import Provider, StepResult, ToolCall
from .tools import TOOLS, BashTool, ToolError, dispatch


@dataclass
class AgentResult:
    final_text: str | None
    stop_reason: str  # end_turn | unverified | max_iterations | max_tokens | error
    iterations: int
    total_input_tokens: int
    total_output_tokens: int
    total_cache_read_tokens: int
    transcript: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class Agent:
    provider: Provider
    system: str
    max_iterations: int = 30
    max_input_tokens: int = 200_000
    truncation_char_budget: int = 120_000  # ~30k tokens of tool_result content
    verify: bool = True  # gate "done" behind tool evidence (see max_pushbacks)
    max_pushbacks: int = 8  # toolless "done"s challenged before giving in
    on_event: Callable[[dict[str, Any]], None] | None = None
    bash: BashTool | None = None

    def __post_init__(self) -> None:
        if self.bash is None:
            self.bash = BashTool()

    def _emit(self, event: dict[str, Any]) -> None:
        # Observers are telemetry, never load-bearing. _emit runs inside the
        # loop's try, so without this guard a crashing printer (rich markup in
        # build output, a broken pipe, a bad terminal encoding) would surface
        # as stop_reason="error" and throw away a working run.
        if self.on_event:
            try:
                self.on_event(event)
            except Exception:  # noqa: BLE001 - telemetry must not fail the task
                pass

    def run(self, task: str) -> AgentResult:
        messages: list[dict[str, Any]] = [{"role": "user", "content": task}]
        transcript: list[dict[str, Any]] = [{"type": "user", "content": task}]
        total_in = total_out = total_cache = 0
        iteration = 0
        used_tools = False
        pushbacks_left = self.max_pushbacks if self.verify else 0
        challenged = False  # has any "done" been pushed back yet?
        tools_since_nudge = False  # successful tool evidence since last pushback

        try:
          while True:
            iteration += 1
            if iteration > self.max_iterations:
                return AgentResult(
                    final_text=None, stop_reason="max_iterations",
                    iterations=iteration - 1,
                    total_input_tokens=total_in, total_output_tokens=total_out,
                    total_cache_read_tokens=total_cache, transcript=transcript,
                )

            self._truncate_if_needed(messages, transcript)
            sr: StepResult = self.provider.step(messages, TOOLS, self.system)
            total_in += sr.usage.input_tokens
            total_out += sr.usage.output_tokens
            total_cache += sr.usage.cache_read_tokens

            transcript.append({
                "type": "assistant", "text": sr.text,
                "tool_calls": [tc.model_dump() for tc in sr.tool_calls],
                "stop_reason": sr.stop_reason, "usage": sr.usage.model_dump(),
            })
            self._emit({"type": "assistant", "text": sr.text,
                        "tool_calls": sr.tool_calls})
            # Running totals every step: a run killed from outside (timeout)
            # must not take its token accounting down with it.
            self._emit({"type": "stats", "iteration": iteration,
                        "input_tokens": total_in, "output_tokens": total_out})

            messages.append(self._assistant_message(sr))

            # The cap is on per-step context size, not cumulative spend: every
            # step resends the whole conversation, so capping the sum would
            # silently end long tasks after a handful of steps. Cap breach takes
            # priority over natural completion even if the model said end_turn.
            if sr.usage.input_tokens >= self.max_input_tokens:
                return AgentResult(
                    final_text=sr.text, stop_reason="max_tokens",
                    iterations=iteration,
                    total_input_tokens=total_in, total_output_tokens=total_out,
                    total_cache_read_tokens=total_cache, transcript=transcript,
                )

            # Output cut off mid-response (often mid-tool-call JSON): nudge a
            # continuation instead of misreporting success.
            if sr.stop_reason == "max_tokens" and not sr.tool_calls:
                nudge = ("Your previous response was cut off by the output "
                         "token limit. Continue: re-issue the incomplete tool "
                         "call in full, or finish your answer.")
                messages.append({"role": "user", "content": nudge})
                transcript.append({"type": "user", "content": nudge})
                continue

            if sr.stop_reason == "end_turn":
                # Verify pass: models grade their own work generously, and
                # some end a turn merely *describing* their next action. A
                # "done" is only accepted when backed by *successful* tool
                # evidence since the last challenge; toolless (or failed-only)
                # dones get pushed back until max_pushbacks runs out. Skipped
                # when no tool was ever used.
                # Don't spend the last iteration on a pushback - a challenge
                # the model can't answer would return max_iterations and throw
                # away the summary it just produced.
                if used_tools and pushbacks_left > 0 and (
                        self.max_iterations - iteration) > 0 and (
                        not challenged or not tools_since_nudge):
                    pushbacks_left -= 1
                    challenged = True
                    tools_since_nudge = False
                    remaining = self.max_iterations - iteration
                    nudge = ("Your turn ended without a completed, verified "
                             f"result. You have {remaining} iterations left - "
                             "do not stop to describe what you plan to do; do "
                             "it now with tool calls, then re-read the original "
                             "task and prove each requirement is met by "
                             "running the relevant code or tests. Only when "
                             "everything passes, finish with your summary.")
                    messages.append({"role": "user", "content": nudge})
                    transcript.append({"type": "user", "content": nudge})
                    continue
                # Accepting a "done" the gate couldn't (or didn't need to)
                # challenge: if it has successful tool evidence behind it,
                # that's a genuine end_turn. If the gate simply ran out of
                # pushbacks or iterations, don't fail open - keep the text
                # but report "unverified" so the caller knows.
                verified = (not self.verify or not used_tools
                            or tools_since_nudge)
                return AgentResult(
                    final_text=sr.text,
                    stop_reason="end_turn" if verified else "unverified",
                    iterations=iteration,
                    total_input_tokens=total_in, total_output_tokens=total_out,
                    total_cache_read_tokens=total_cache, transcript=transcript,
                )

            # Stopped with no tool calls but not a clean end_turn: refusal,
            # content_filter, or an unmapped provider reason. Never report that
            # as success - surface the reason so the caller (and CLI exit code)
            # knows the run did not complete.
            if not sr.tool_calls:
                return AgentResult(
                    final_text=sr.text, stop_reason=sr.stop_reason,
                    iterations=iteration,
                    total_input_tokens=total_in, total_output_tokens=total_out,
                    total_cache_read_tokens=total_cache, transcript=transcript,
                )

            used_tools = True
            tool_results = self._execute_tool_calls(sr.tool_calls, transcript)
            # Only a *successful* tool counts as verification evidence - a
            # failed-only round must not satisfy the verify gate.
            if any(not r["is_error"] for r in tool_results):
                tools_since_nudge = True
            messages.append({"role": "user", "content": tool_results})
        except Exception as e:  # noqa: BLE001 - any failure becomes a result, not a crash
            transcript.append({"type": "error", "message": f"{type(e).__name__}: {e}"})
            return AgentResult(
                final_text=f"agent error: {type(e).__name__}: {e}",
                stop_reason="error", iterations=iteration,
                total_input_tokens=total_in, total_output_tokens=total_out,
                total_cache_read_tokens=total_cache, transcript=transcript,
            )

    def _truncate_if_needed(self, messages: list[dict[str, Any]],
                            transcript: list[dict[str, Any]]) -> None:
        def total_chars() -> int:
            n = 0
            for m in messages:
                c = m.get("content")
                if isinstance(c, str):
                    n += len(c)
                elif isinstance(c, list):
                    for b in c:
                        n += len(b.get("text", "")) + len(b.get("content", ""))
                        # A tool_use block's args live under `input` - a huge
                        # edit_file `new` value hides here and re-inflates every
                        # request unless it is counted (and dropped) too.
                        for v in (b.get("input") or {}).values():
                            if isinstance(v, str):
                                n += len(v)
            return n

        if total_chars() <= self.truncation_char_budget:
            return

        # Drop oldest tool_result content first, then oversized tool_use inputs;
        # keep the block and its id so the tool_use/tool_result pairing survives.
        for m in messages:
            if not isinstance(m.get("content"), list):
                continue
            for b in m["content"]:
                if b.get("type") == "tool_result" and not str(
                        b.get("content", "")).startswith("[truncated"):
                    original_len = len(b.get("content", ""))
                    b["content"] = f"[truncated - {original_len} chars dropped]"
                    transcript.append({"type": "truncation",
                                       "tool_use_id": b.get("tool_use_id"),
                                       "dropped_chars": original_len})
                    if total_chars() <= self.truncation_char_budget:
                        return
        # Still over budget: shrink the largest string args of past tool_use
        # blocks (e.g. a giant edit_file `new`). The tool already ran; its
        # result is elsewhere in history, so the full input is no longer needed.
        for m in messages:
            if not isinstance(m.get("content"), list):
                continue
            for b in m["content"]:
                if b.get("type") != "tool_use":
                    continue
                inp = b.get("input") or {}
                for k, v in list(inp.items()):
                    if isinstance(v, str) and len(v) > 200 and not v.startswith(
                            "[truncated"):
                        inp[k] = f"[truncated - {len(v)} chars dropped]"
                        transcript.append({"type": "truncation",
                                           "tool_use_id": b.get("id"),
                                           "dropped_chars": len(v)})
                        if total_chars() <= self.truncation_char_budget:
                            return

    def _assistant_message(self, sr: StepResult) -> dict[str, Any]:
        # Canonical shape: content blocks are the single source of truth for
        # both text and tool calls. Each provider re-serializes from these -
        # no duplicated tool_calls copy to drift out of sync under truncation.
        content_blocks: list[dict[str, Any]] = []
        if sr.text:
            content_blocks.append({"type": "text", "text": sr.text})
        for tc in sr.tool_calls:
            content_blocks.append({"type": "tool_use", "id": tc.id,
                                   "name": tc.name, "input": tc.arguments})
        if not content_blocks:
            # The model produced nothing at all. An empty-content assistant
            # message is invalid to Anthropic (400, and 400 is not retryable,
            # so it ends the run) and reads as a blank turn to OpenAI, which
            # tends to beget more blank turns until the verify budget is gone.
            # A visible placeholder keeps the history well-formed and tells
            # the model what just happened.
            content_blocks.append({"type": "text", "text": "(no output)"})
        return {"role": "assistant", "content": content_blocks}

    def _execute_tool_calls(self, calls: list[ToolCall],
                            transcript: list[dict[str, Any]]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for call in calls:
            try:
                output = dispatch(call.name, call.arguments, bash=self.bash)
                is_error = False
            except ToolError as e:
                output = f"ERROR: {e}"
                is_error = True
            except Exception as e:  # noqa: BLE001 - a bad tool arg (e.g. wrong
                # type from a weak model) must come back as a fixable error,
                # not crash the whole run.
                output = f"ERROR: {type(e).__name__}: {e}"
                is_error = True
            transcript.append({"type": "tool_result", "id": call.id,
                               "name": call.name, "output": output,
                               "is_error": is_error})
            self._emit({"type": "tool_result", "id": call.id,
                        "output": output, "is_error": is_error})
            results.append({"type": "tool_result", "tool_use_id": call.id,
                            "content": output, "is_error": is_error})
        return results
