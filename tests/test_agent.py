import pytest
from unittest.mock import MagicMock

from nano.agent import Agent, AgentResult
from nano.providers import StepResult, ToolCall, Usage
from nano.tools import BashTool


class FakeProvider:
    """Returns a scripted sequence of StepResults; raises if we run past."""
    model = "fake-model"

    def __init__(self, scripted: list[StepResult]) -> None:
        self.scripted = list(scripted)
        self.calls: list[dict] = []

    def step(self, messages, tools, system) -> StepResult:
        self.calls.append({"messages": list(messages),
                           "tools": tools, "system": system})
        if not self.scripted:
            raise AssertionError("FakeProvider exhausted")
        return self.scripted.pop(0)


def _u(i, o):
    return Usage(input_tokens=i, output_tokens=o)


def test_agent_one_shot_end_turn():
    fp = FakeProvider([
        StepResult(text="task done", tool_calls=[], stop_reason="end_turn",
                   usage=_u(10, 5)),
    ])
    agent = Agent(provider=fp, system="sys", max_iterations=10)
    result = agent.run("solve x")

    assert isinstance(result, AgentResult)
    assert result.final_text == "task done"
    assert result.stop_reason == "end_turn"
    assert result.iterations == 1
    assert result.total_input_tokens == 10
    assert result.total_output_tokens == 5


def test_agent_executes_tool_then_completes(tmp_workdir):
    p = tmp_workdir / "a.txt"
    p.write_text("hello\n")

    fp = FakeProvider([
        StepResult(
            text="reading", tool_calls=[ToolCall(
                id="t1", name="read_file", arguments={"path": str(p)})],
            stop_reason="tool_use", usage=_u(50, 10),
        ),
        StepResult(text="done", tool_calls=[], stop_reason="end_turn",
                   usage=_u(70, 4)),
    ])
    agent = Agent(provider=fp, system="sys", max_iterations=10, verify=False)
    result = agent.run("read it")

    assert result.iterations == 2
    assert result.final_text == "done"
    second_call_msgs = fp.calls[1]["messages"]
    user_with_tool_result = [m for m in second_call_msgs if m["role"] == "user"][-1]
    assert "hello" in str(user_with_tool_result["content"])


def test_agent_iteration_cap_stops_loop():
    looper = [
        StepResult(text="loop",
                   tool_calls=[ToolCall(id=f"t{i}", name="bash",
                                        arguments={"command": "echo i"})],
                   stop_reason="tool_use", usage=_u(1, 1))
        for i in range(20)
    ]
    fp = FakeProvider(looper)
    agent = Agent(provider=fp, system="sys", max_iterations=3)
    result = agent.run("loop forever")
    assert result.iterations == 3
    assert result.stop_reason == "max_iterations"


def test_agent_token_cap_stops_when_context_outgrows_budget():
    # The cap is on per-step context size (what one request sends), not on
    # cumulative spend. First step fits; second step's context exceeds cap.
    huge = [
        StepResult(text=None,
                   tool_calls=[ToolCall(id="t1", name="bash",
                                        arguments={"command": "echo x"})],
                   stop_reason="tool_use", usage=_u(50_000, 100)),
        StepResult(text=None,
                   tool_calls=[ToolCall(id="t2", name="bash",
                                        arguments={"command": "echo x"})],
                   stop_reason="tool_use", usage=_u(90_000, 100)),
    ]
    fp = FakeProvider(huge)
    agent = Agent(provider=fp, system="sys", max_iterations=10,
                  max_input_tokens=80_000)
    result = agent.run("burn budget")
    assert result.stop_reason == "max_tokens"
    assert result.iterations == 2


def test_agent_cumulative_spend_does_not_kill_long_tasks():
    # Regression: cumulative input across steps (150k) exceeds the cap, but
    # each individual step's context (50k) fits — the task must complete.
    steps = [
        StepResult(text=None,
                   tool_calls=[ToolCall(id=f"t{i}", name="bash",
                                        arguments={"command": "echo x"})],
                   stop_reason="tool_use", usage=_u(50_000, 100))
        for i in range(2)
    ]
    steps.append(StepResult(text="ok", tool_calls=[], stop_reason="end_turn",
                            usage=_u(50_000, 50)))
    fp = FakeProvider(steps)
    agent = Agent(provider=fp, system="sys", max_iterations=10,
                  max_input_tokens=80_000, verify=False)
    result = agent.run("long task")
    assert result.stop_reason == "end_turn"
    assert result.total_input_tokens == 150_000


def test_agent_reports_tool_error_back_to_model():
    """When dispatch raises ToolError, the loop continues with is_error=True
    in the tool_result, and the model gets to retry."""
    from nano.tools import ToolError

    fp = FakeProvider([
        StepResult(
            text="trying", tool_calls=[ToolCall(
                id="t1", name="read_file", arguments={"path": "no/such/file"})],
            stop_reason="tool_use", usage=_u(10, 5),
        ),
        StepResult(text="gave up", tool_calls=[], stop_reason="end_turn",
                   usage=_u(10, 5)),
    ])
    agent = Agent(provider=fp, system="sys", max_iterations=10, verify=False)
    result = agent.run("read missing file")

    assert result.stop_reason == "end_turn"
    assert result.iterations == 2
    second_call_msgs = fp.calls[1]["messages"]
    last_user = [m for m in second_call_msgs if m["role"] == "user"][-1]
    content = last_user["content"]
    assert isinstance(content, list)
    tr = content[0]
    assert tr["type"] == "tool_result"
    assert tr["is_error"] is True
    assert "ERROR" in tr["content"]


def test_agent_truncates_oldest_tool_result_when_history_grows():
    # Five tool_use rounds then a final end_turn. Truncation budget set so
    # the oldest tool_result must be replaced with a placeholder before the
    # last step is sent to the provider.
    big = "x" * 5000
    rounds = []
    for i in range(5):
        rounds.append(StepResult(
            text=f"step{i}",
            tool_calls=[ToolCall(id=f"t{i}", name="bash",
                                 arguments={"command": "echo " + big})],
            stop_reason="tool_use", usage=_u(100, 10)))
    rounds.append(StepResult(text="done", tool_calls=[], stop_reason="end_turn",
                             usage=_u(100, 10)))
    fp = FakeProvider(rounds)

    class _RecordingBash:
        def run(self, command, timeout=30):
            return command.removeprefix("echo ")

    agent = Agent(provider=fp, system="sys",
                  max_iterations=20, max_input_tokens=10**9,
                  bash=_RecordingBash(), verify=False)
    agent.truncation_char_budget = 8000  # forces truncation by step 4+
    result = agent.run("loop")

    assert result.stop_reason == "end_turn"
    truncations = [t for t in result.transcript if t.get("type") == "truncation"]
    assert truncations, "expected at least one truncation event"
    last_call_messages = fp.calls[-1]["messages"]
    seen_placeholder = any(
        isinstance(m.get("content"), list)
        and any(b.get("content", "").startswith("[truncated")
                for b in m["content"] if b.get("type") == "tool_result")
        for m in last_call_messages
    )
    assert seen_placeholder


def test_agent_nudges_continuation_when_output_truncated():
    # A response cut off by the output limit (stop_reason=max_tokens, no tool
    # calls) must not be reported as success — the loop nudges a continuation.
    fp = FakeProvider([
        StepResult(text="half a thou", tool_calls=[], stop_reason="max_tokens",
                   usage=_u(10, 4096)),
        StepResult(text="done", tool_calls=[], stop_reason="end_turn",
                   usage=_u(20, 5)),
    ])
    agent = Agent(provider=fp, system="sys", max_iterations=10)
    result = agent.run("long answer")

    assert result.stop_reason == "end_turn"
    assert result.iterations == 2
    second_call_msgs = fp.calls[1]["messages"]
    last_user = [m for m in second_call_msgs if m["role"] == "user"][-1]
    assert "cut off" in str(last_user["content"])


def test_agent_verify_pass_accepts_done_backed_by_tool_evidence():
    # First done is challenged; the model then RUNS something (tool evidence)
    # and its next done is accepted.
    fp = FakeProvider([
        StepResult(text="fixing", tool_calls=[ToolCall(
            id="t1", name="bash", arguments={"command": "echo hi"})],
            stop_reason="tool_use", usage=_u(10, 5)),
        StepResult(text="done", tool_calls=[], stop_reason="end_turn",
                   usage=_u(20, 5)),
        StepResult(text="verifying", tool_calls=[ToolCall(
            id="t2", name="bash", arguments={"command": "run tests"})],
            stop_reason="tool_use", usage=_u(30, 5)),
        StepResult(text="verified done", tool_calls=[], stop_reason="end_turn",
                   usage=_u(40, 5)),
    ])

    class _OkBash:
        def run(self, command, timeout=30):
            return "ok\n"
    agent = Agent(provider=fp, system="sys", max_iterations=10, bash=_OkBash())
    result = agent.run("fix the bug")

    assert result.stop_reason == "end_turn"
    assert result.iterations == 4
    assert result.final_text == "verified done"
    third_call_msgs = fp.calls[2]["messages"]
    last_user = [m for m in third_call_msgs if m["role"] == "user"][-1]
    assert "re-read the original task" in str(last_user["content"])


def test_agent_pushback_skipped_on_final_iteration():
    # A "done" landing exactly on the last iteration must be accepted as-is:
    # a pushback here can never be answered, so it would turn a finished run
    # into max_iterations and throw away the summary the model just wrote.
    # With successful tool evidence behind it, it still counts as end_turn.
    fp = FakeProvider([
        StepResult(text="working", tool_calls=[ToolCall(
            id="t1", name="bash", arguments={"command": "echo hi"})],
            stop_reason="tool_use", usage=_u(10, 5)),
        StepResult(text="summary", tool_calls=[], stop_reason="end_turn",
                   usage=_u(20, 5)),
    ])

    class _OkBash:
        def run(self, command, timeout=30):
            return "ok\n"
    agent = Agent(provider=fp, system="sys", max_iterations=2, bash=_OkBash())
    result = agent.run("task")

    assert result.stop_reason == "end_turn"
    assert result.final_text == "summary"
    assert result.iterations == 2


def test_agent_unverified_when_no_evidence_and_no_pushback_room():
    # Failed-only tool round, then "done" on the final iteration: the loop
    # can't push back, but it must not report success either. The text is
    # kept, the stop reason says what actually happened.
    fp = FakeProvider([
        StepResult(text="working", tool_calls=[ToolCall(
            id="t1", name="read_file",
            arguments={"path": "definitely_missing_file_xyz"})],
            stop_reason="tool_use", usage=_u(10, 5)),
        StepResult(text="done", tool_calls=[], stop_reason="end_turn",
                   usage=_u(20, 5)),
    ])
    agent = Agent(provider=fp, system="sys", max_iterations=2)
    result = agent.run("task")

    assert result.stop_reason == "unverified"
    assert result.final_text == "done"
    assert result.iterations == 2


def test_agent_exits_cleanly_when_max_runtime_exceeded(monkeypatch):
    # Official scoring forces an errored (externally killed) trial to reward
    # zero even when the workspace would pass, while a clean exit is graded.
    # With a known budget the loop must stop itself before the outer kill.
    import nano.agent as agent_mod
    clock = {"t": 1000.0}
    monkeypatch.setattr(agent_mod.time, "monotonic", lambda: clock["t"])

    def step_then_advance(messages, tools, system):
        clock["t"] += 400.0  # each step costs 400 "seconds"
        return StepResult(text="working", tool_calls=[ToolCall(
            id=f"t{clock['t']}", name="bash", arguments={"command": "echo hi"})],
            stop_reason="tool_use", usage=_u(10, 5))

    fp = MagicMock()
    fp.step.side_effect = step_then_advance

    class _OkBash:
        def run(self, command, timeout=300):
            return "ok\n"

    agent = Agent(provider=fp, system="sys", max_iterations=50,
                  max_runtime_sec=900.0, bash=_OkBash())
    result = agent.run("task")

    assert result.stop_reason == "max_runtime"
    assert result.iterations <= 3  # 400s/step against a 900s budget


def test_agent_wrap_up_nudge_fires_late_in_the_budget(monkeypatch):
    # Past 80% of the budget the agent gets one explicit instruction to get
    # the workspace gradable and finish, and verify pushbacks stop (a
    # challenge this late risks converting a clean exit into an outer kill).
    import nano.agent as agent_mod
    clock = {"t": 0.0}
    monkeypatch.setattr(agent_mod.time, "monotonic", lambda: clock["t"])

    steps = iter([
        (300.0, StepResult(text="working", tool_calls=[ToolCall(
            id="t1", name="bash", arguments={"command": "echo hi"})],
            stop_reason="tool_use", usage=_u(10, 5))),
        (550.0, StepResult(text="more", tool_calls=[ToolCall(
            id="t2", name="bash", arguments={"command": "echo hi"})],
            stop_reason="tool_use", usage=_u(10, 5))),
        (860.0, StepResult(text="last bit", tool_calls=[ToolCall(
            id="t3", name="bash", arguments={"command": "echo hi"})],
            stop_reason="tool_use", usage=_u(10, 5))),
        (880.0, StepResult(text="done", tool_calls=[], stop_reason="end_turn",
                           usage=_u(10, 5))),
    ])
    calls = []

    def step(messages, tools, system):
        calls.append([dict(m) for m in messages])
        t, sr = next(steps)
        clock["t"] = t
        return sr

    fp = MagicMock()
    fp.step.side_effect = step

    class _OkBash:
        def run(self, command, timeout=300):
            return "ok\n"

    agent = Agent(provider=fp, system="sys", max_iterations=50,
                  max_runtime_sec=1000.0, bash=_OkBash())
    result = agent.run("task")

    # done at 86% of budget: verify pushback must NOT fire
    assert result.stop_reason == "end_turn"
    assert result.final_text == "done"
    # the wrap-up nudge appeared exactly once, as its own user message
    flat = [m for msgs in calls for m in msgs]
    nudges = [m for m in flat if m.get("role") == "user"
              and "Time is nearly up" in str(m.get("content"))]
    assert nudges, "wrap-up nudge never delivered"


def test_consecutive_duplicate_calls_get_a_signal():
    # 10.9% of calls in the measured run were exact repeats (worst streak
    # 251 consecutive). The 2nd+ identical call in a row gets a note appended
    # to its result; a different call resets, so pytest-after-edit is silent.
    class _OkBash:
        def run(self, command, timeout=300):
            return "ok\n"

    agent = Agent(provider=FakeProvider([]), system="sys", bash=_OkBash())
    agent._t0, agent._last_sig, agent._repeat_n = 0.0, None, 0
    tr = []
    r1 = agent._execute_tool_calls([ToolCall(
        id="a", name="bash", arguments={"command": "pytest"})], tr)
    r2 = agent._execute_tool_calls([ToolCall(
        id="b", name="bash", arguments={"command": "pytest"})], tr)
    r3 = agent._execute_tool_calls([ToolCall(
        id="c", name="bash", arguments={"command": "ls"})], tr)
    r4 = agent._execute_tool_calls([ToolCall(
        id="d", name="bash", arguments={"command": "pytest"})], tr)

    assert "cannot produce a new result" not in r1[0]["content"]
    assert "2x in a row" in r2[0]["content"]
    assert "cannot produce a new result" not in r3[0]["content"]
    assert "cannot produce a new result" not in r4[0]["content"], "reset failed"


def test_bash_timeout_clamped_to_remaining_budget():
    # A model told to 'set timeout generously' can pass timeout=3600 on a
    # 900s-budget task; one hung command then guarantees the external kill
    # (forced zero). With a known deadline the executed timeout is clamped to
    # the remaining budget minus teardown margin - without mutating the
    # arguments recorded in history.
    import time as _t
    captured = {}

    class _CapBash:
        def run(self, command, timeout=300):
            captured["timeout"] = timeout
            return "ok\n"

    agent = Agent(provider=FakeProvider([]), system="sys",
                  max_runtime_sec=900.0, bash=_CapBash())
    agent._t0 = _t.monotonic() - 700.0  # 700s elapsed, ~200s remaining
    agent._last_sig, agent._repeat_n = None, 0
    call = ToolCall(id="a", name="bash",
                    arguments={"command": "sleep 5", "timeout": 3600})
    agent._execute_tool_calls([call], [])

    assert captured["timeout"] <= 160
    assert call.arguments["timeout"] == 3600, "history args were mutated"


def test_tool_calls_execute_even_when_stop_reason_says_end_turn():
    # OpenAI-compatible proxies sometimes return finish_reason "stop" WITH
    # tool_calls populated; the provider maps "stop" to "end_turn". The old
    # flow checked stop_reason before tool_calls, so the pending calls were
    # silently dropped - and a verify pushback then produced an assistant
    # message with tool_use blocks followed by a plain user message, which
    # both APIs reject with a non-retryable 400. Calls present = execute.
    fp = FakeProvider([
        StepResult(text="running", tool_calls=[ToolCall(
            id="t1", name="bash", arguments={"command": "echo hi"})],
            stop_reason="end_turn", usage=_u(10, 5)),
        StepResult(text="done", tool_calls=[], stop_reason="end_turn",
                   usage=_u(20, 5)),
    ])

    class _OkBash:
        def run(self, command, timeout=300):
            return "ok\n"

    agent = Agent(provider=fp, system="sys", max_iterations=10,
                  verify=False, bash=_OkBash())
    result = agent.run("task")

    assert result.stop_reason == "end_turn"
    assert result.final_text == "done"
    assert result.iterations == 2
    # the tool actually ran: the second request carries its tool_result
    second = fp.calls[1]["messages"]
    assert any(isinstance(m.get("content"), list) and any(
        b.get("type") == "tool_result" for b in m["content"]) for m in second)


def test_truncation_cuts_to_low_water_mark_not_just_under_budget():
    # Truncating to exactly the budget means the very next tool result pushes
    # history back over it, so truncation re-fires every single step - and
    # each firing mutates a message near the head of the conversation, which
    # invalidates the prompt-cache prefix for every remaining request. Cut
    # down to a low-water mark instead so it fires once per ~N steps.
    fp = FakeProvider([])
    agent = Agent(provider=fp, system="sys")
    agent.truncation_char_budget = 1000

    messages = [{"role": "user", "content": "task"}]
    for i in range(5):
        messages.append({"role": "assistant", "content": [
            {"type": "tool_use", "id": f"t{i}", "name": "bash", "input": {}}]})
        messages.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": f"t{i}", "content": "x" * 400}]})

    transcript = []
    agent._truncate_if_needed(messages, transcript)

    def total():
        n = 0
        for m in messages:
            c = m.get("content")
            if isinstance(c, str):
                n += len(c)
            elif isinstance(c, list):
                for b in c:
                    n += len(b.get("text", "")) + len(str(b.get("content", "")))
        return n

    assert total() <= 600, (
        f"history still at {total()} chars; truncation stopped at the budget "
        f"line instead of the low-water mark, so it will re-fire every step")


def test_agent_nudges_on_empty_end_turn_instead_of_accepting():
    # A model that returns literally nothing (0 output tokens, no tool calls,
    # stop_reason end_turn) has not completed anything - it failed to
    # generate. Observed live: polyglot-rust-c ended at iteration 1 of 100
    # with `in=147 out=0`, was accepted as done, and scored 0 with 87% of its
    # clock unused. The loop must nudge a retry, not declare victory.
    fp = FakeProvider([
        StepResult(text=None, tool_calls=[], stop_reason="end_turn",
                   usage=_u(10, 0)),
        StepResult(text="real answer", tool_calls=[], stop_reason="end_turn",
                   usage=_u(20, 5)),
    ])
    agent = Agent(provider=fp, system="sys", max_iterations=10)
    result = agent.run("task")

    assert result.stop_reason == "end_turn"
    assert result.final_text == "real answer"
    assert result.iterations == 2
    # the model was told its turn was empty
    second_call = fp.calls[1]["messages"]
    last_user = [m for m in second_call if m["role"] == "user"][-1]
    assert "empty" in str(last_user["content"]).lower()


def test_agent_gives_up_after_three_consecutive_empty_turns():
    # A model that stays empty is dead; report an error, don't spin the
    # remaining budget on nudges or dress the nothing up as success.
    fp = FakeProvider([
        StepResult(text=None, tool_calls=[], stop_reason="end_turn",
                   usage=_u(10, 0))
        for _ in range(3)
    ])
    agent = Agent(provider=fp, system="sys", max_iterations=10)
    result = agent.run("task")

    assert result.stop_reason == "error"
    assert result.iterations == 3


def test_empty_model_turn_never_yields_an_empty_content_message():
    # A step with no text and no tool calls used to serialize as
    # {"role": "assistant", "content": []}. Anthropic rejects a non-final
    # message with empty content (400, and 400 is not retryable), so on the
    # Anthropic path that ends the run; on the OpenAI path it becomes
    # content:null and the model tends to emit another empty turn, burning
    # the verify budget. Observed on 5 trials, all scoring 0.
    fp = FakeProvider([])
    agent = Agent(provider=fp, system="sys")
    msg = agent._assistant_message(
        StepResult(text=None, tool_calls=[], stop_reason="end_turn",
                   usage=_u(10, 0)))

    assert msg["content"], "assistant message must never have empty content"


def test_empty_model_turn_survives_openai_serialization():
    from nano.providers import _normalize_for_openai

    fp = FakeProvider([])
    agent = Agent(provider=fp, system="sys")
    msg = agent._assistant_message(
        StepResult(text=None, tool_calls=[], stop_reason="end_turn",
                   usage=_u(10, 0)))
    out = _normalize_for_openai(msg)

    # Neither content nor tool_calls may be empty, or the turn carries nothing.
    assert out[0].get("content") or out[0].get("tool_calls")


def test_agent_survives_a_raising_on_event_observer():
    # _emit runs inside the loop's try, so a crashing observer (e.g. the CLI
    # printer choking on markup-shaped build output) used to be converted into
    # stop_reason="error" and end an otherwise healthy run. Observers are
    # telemetry: they must never be able to fail the task.
    fp = FakeProvider([
        StepResult(text="answer", tool_calls=[], stop_reason="end_turn",
                   usage=_u(10, 5)),
    ])

    def boom(event):
        raise RuntimeError("printer exploded")

    agent = Agent(provider=fp, system="sys", max_iterations=10, on_event=boom)
    result = agent.run("do it")

    assert result.stop_reason == "end_turn"
    assert result.final_text == "answer"


def test_agent_pushes_back_on_toolless_done_up_to_cap():
    # A model that keeps declaring done WITHOUT running anything gets pushed
    # back max_pushbacks times. Giving in must not masquerade as success:
    # the result is kept but flagged "unverified".
    fp = FakeProvider([
        StepResult(text="working", tool_calls=[ToolCall(
            id="t1", name="bash", arguments={"command": "echo hi"})],
            stop_reason="tool_use", usage=_u(10, 5)),
    ] + [
        StepResult(text=f"done {i}", tool_calls=[], stop_reason="end_turn",
                   usage=_u(20 + i, 5))
        for i in range(4)  # 3 pushbacks consumed, 4th done accepted
    ])
    agent = Agent(provider=fp, system="sys", max_iterations=20, max_pushbacks=3)
    result = agent.run("hard task")

    assert result.stop_reason == "unverified"
    assert result.iterations == 5
    assert result.final_text == "done 3"
    # each pushback mentions the remaining iteration budget
    last_user = [m for m in fp.calls[-1]["messages"] if m["role"] == "user"][-1]
    assert "iterations" in str(last_user["content"])


def test_agent_verify_pass_skipped_without_tool_use():
    # Pure text answer, no tools touched: nothing to verify, no extra step.
    fp = FakeProvider([
        StepResult(text="answer", tool_calls=[], stop_reason="end_turn",
                   usage=_u(10, 5)),
    ])
    agent = Agent(provider=fp, system="sys", max_iterations=10)
    result = agent.run("what is 2+2")
    assert result.iterations == 1
    assert result.final_text == "answer"


def test_agent_verify_pass_can_be_disabled():
    fp = FakeProvider([
        StepResult(text="working", tool_calls=[ToolCall(
            id="t1", name="bash", arguments={"command": "echo hi"})],
            stop_reason="tool_use", usage=_u(10, 5)),
        StepResult(text="done", tool_calls=[], stop_reason="end_turn",
                   usage=_u(20, 5)),
    ])
    agent = Agent(provider=fp, system="sys", max_iterations=10, verify=False)
    result = agent.run("fix it")
    assert result.iterations == 2
    assert result.final_text == "done"


def test_agent_emits_running_stats_each_step():
    # Token totals must survive an external kill - emitted every step,
    # not only in the final summary.
    events = []
    fp = FakeProvider([
        StepResult(text="working", tool_calls=[ToolCall(
            id="t1", name="bash", arguments={"command": "echo hi"})],
            stop_reason="tool_use", usage=_u(100, 20)),
        StepResult(text="done", tool_calls=[], stop_reason="end_turn",
                   usage=_u(150, 10)),
    ])
    agent = Agent(provider=fp, system="sys", max_iterations=10, verify=False,
                  on_event=events.append)
    agent.run("task")
    stats = [e for e in events if e["type"] == "stats"]
    assert len(stats) == 2
    assert stats[0]["iteration"] == 1
    assert stats[0]["input_tokens"] == 100
    assert stats[0]["output_tokens"] == 20
    assert "elapsed" in stats[0]  # wall-clock visible per step in the logs
    assert stats[1]["input_tokens"] == 250


def test_agent_refusal_not_reported_as_success():
    # A provider stop_reason other than end_turn, with no tool calls (refusal,
    # content_filter, unknown), must NOT be reported as end_turn success.
    fp = FakeProvider([
        StepResult(text="I can't help with that.", tool_calls=[],
                   stop_reason="refusal", usage=_u(10, 5)),
    ])
    agent = Agent(provider=fp, system="sys", max_iterations=10)
    result = agent.run("do a thing")
    assert result.stop_reason == "refusal"
    assert result.stop_reason != "end_turn"


def test_agent_failed_tool_does_not_satisfy_verify_gate():
    # After a real 'done', the verify nudge fires. If the model then only runs
    # a FAILING tool and says done again, that must not count as evidence -
    # push back again (up to the cap).
    fp = FakeProvider([
        StepResult(text="working", tool_calls=[ToolCall(
            id="t1", name="bash", arguments={"command": "echo hi"})],
            stop_reason="tool_use", usage=_u(10, 5)),
        StepResult(text="done", tool_calls=[], stop_reason="end_turn",
                   usage=_u(20, 5)),  # first done -> nudged
        StepResult(text="trying", tool_calls=[ToolCall(
            id="t2", name="read_file", arguments={"path": "no/such/file"})],
            stop_reason="tool_use", usage=_u(30, 5)),  # FAILS
        StepResult(text="done again", tool_calls=[], stop_reason="end_turn",
                   usage=_u(40, 5)),  # should be pushed back, not accepted
        StepResult(text="really done", tool_calls=[], stop_reason="end_turn",
                   usage=_u(50, 5)),
    ])
    agent = Agent(provider=fp, system="sys", max_iterations=20)
    result = agent.run("fix it")
    # The failing-tool 'done' was challenged, so more than 2 pushbacks happened.
    challenges = sum(1 for m in fp.calls[-1]["messages"]
                     if m["role"] == "user"
                     and "without a completed" in str(m["content"]))
    assert challenges >= 2


def test_agent_tool_exception_becomes_recoverable_error(tmp_workdir):
    # A tool that raises a NON-ToolError (edit_file old=None on an EXISTING
    # file -> text.count(None) TypeError) must come back as a recoverable
    # tool_result error, never crash the run.
    p = tmp_workdir / "x.py"
    p.write_text("a = 1\n")
    fp = FakeProvider([
        StepResult(text="editing", tool_calls=[ToolCall(
            id="t1", name="edit_file",
            arguments={"path": str(p), "old": None, "new": "y"})],
            stop_reason="tool_use", usage=_u(10, 5)),
        StepResult(text="ok", tool_calls=[], stop_reason="end_turn",
                   usage=_u(20, 5)),
    ])
    agent = Agent(provider=fp, system="sys", max_iterations=10, verify=False)
    result = agent.run("edit")  # must not raise
    assert result.stop_reason == "end_turn"
    tr = [m for m in fp.calls[1]["messages"] if m["role"] == "user"][-1]
    assert tr["content"][0]["is_error"] is True


def test_agent_run_never_raises_on_provider_error():
    # A provider that raises must yield stop_reason='error', not a traceback.
    class _BoomProvider:
        model = "boom"
        def step(self, messages, tools, system):
            raise RuntimeError("api exploded")
    agent = Agent(provider=_BoomProvider(), system="sys", max_iterations=10)
    result = agent.run("task")
    assert result.stop_reason == "error"


def test_agent_truncates_huge_tool_use_input():
    # A giant edit_file `new` arg lives under tool_use.input and must be counted
    # AND truncated, or it re-inflates every request forever.
    big = "x" * 5000
    fp = FakeProvider([
        StepResult(text="writing", tool_calls=[ToolCall(
            id="t1", name="edit_file",
            arguments={"path": "big.py", "old": "", "new": big})],
            stop_reason="tool_use", usage=_u(100, 10)),
        StepResult(text="more", tool_calls=[ToolCall(
            id="t2", name="bash", arguments={"command": "echo done"})],
            stop_reason="tool_use", usage=_u(100, 10)),
        StepResult(text="done", tool_calls=[], stop_reason="end_turn",
                   usage=_u(100, 10)),
    ])

    class _OkBash:
        def run(self, command, timeout=30):
            return "ok\n"

    agent = Agent(provider=fp, system="sys", max_iterations=10,
                  max_input_tokens=10**9, bash=_OkBash(), verify=False)
    agent.truncation_char_budget = 2000  # below the 5000-char input
    # edit_file writes a real file; point it at a tmp dir and let it run.
    import os as _os
    import tempfile
    fp.scripted[0].tool_calls[0].arguments["path"] = _os.path.join(
        tempfile.mkdtemp(), "big.py")
    result = agent.run("write big")
    assert result.stop_reason == "end_turn"
    truncs = [t for t in result.transcript if t.get("type") == "truncation"]
    assert any(t["dropped_chars"] >= 5000 for t in truncs), \
        "huge tool_use input was not truncated"
    # the giant input must be gone from the final request
    last = fp.calls[-1]["messages"]
    seen_big = any(
        isinstance(m.get("content"), list)
        and any(len(str(v)) >= 5000
                for b in m["content"] if b.get("type") == "tool_use"
                for v in (b.get("input") or {}).values())
        for m in last
    )
    assert not seen_big


# --- E8: adversarial-review fix set -------------------------------------


def test_repeated_max_tokens_turns_give_up_instead_of_spinning():
    # A gateway that keeps returning length-cut output with no parseable tool
    # calls used to be nudged forever: the only bound was max_iterations, so
    # the trial burned its whole budget and died on the EXTERNAL kill, which
    # scores zero even when the workspace would pass. Cap it like empty turns.
    fp = FakeProvider([
        StepResult(text="cut off mid-", tool_calls=[],
                   stop_reason="max_tokens", usage=_u(10, 8192))
        for _ in range(10)
    ])
    agent = Agent(provider=fp, system="sys", max_iterations=50, verify=False)
    result = agent.run("write a big file")

    assert result.stop_reason == "max_tokens"
    # 3 strikes, not 50 iterations of nudging.
    assert result.iterations <= 4, f"spun {result.iterations} times"


def test_empty_turn_flagged_max_tokens_still_counts_as_an_empty_turn():
    # finish_reason "length" with EMPTY content is a failed generation. The
    # continuation branch used to swallow it before the empty-turn guard ran,
    # so the dead-model detector never fired.
    fp = FakeProvider([
        StepResult(text=None, tool_calls=[], stop_reason="max_tokens",
                   usage=_u(10, 0))
        for _ in range(10)
    ])
    agent = Agent(provider=fp, system="sys", max_iterations=50, verify=False)
    result = agent.run("do it")

    assert result.iterations <= 4
    assert result.stop_reason in ("error", "max_tokens")


def test_a_productive_turn_resets_the_max_tokens_strike_count():
    # Two cutoffs, real progress, then two more must NOT trip the 3-strike cap:
    # long file writes legitimately hit the ceiling more than once per task.
    big = StepResult(text="cut", tool_calls=[], stop_reason="max_tokens",
                     usage=_u(10, 8192))
    fp = FakeProvider([
        big, big,
        StepResult(text="ok", tool_calls=[ToolCall(
            id="t1", name="bash", arguments={"command": "echo hi"})],
            stop_reason="tool_use", usage=_u(20, 10)),
        big, big,
        StepResult(text="done", tool_calls=[], stop_reason="end_turn",
                   usage=_u(30, 5)),
    ])
    agent = Agent(provider=fp, system="sys", max_iterations=50, verify=False)
    result = agent.run("write two big files")

    assert result.stop_reason == "end_turn"
    assert result.final_text == "done"


def test_truncation_spares_the_tool_result_the_model_has_not_seen(tmp_workdir):
    # Pass 1 blanked oldest-to-newest including the result appended at the end
    # of the PREVIOUS iteration - output the model never got to read. It then
    # re-runs the command, and the duplicate-call note scolds it for doing so.
    big = tmp_workdir / "big.txt"
    big.write_text("x" * 40_000)
    small = tmp_workdir / "small.txt"
    small.write_text("the answer is 42" + chr(10))

    fp = FakeProvider([
        StepResult(text="read big", tool_calls=[ToolCall(
            id="t1", name="read_file", arguments={"path": str(big)})],
            stop_reason="tool_use", usage=_u(10, 5)),
        StepResult(text="read small", tool_calls=[ToolCall(
            id="t2", name="read_file", arguments={"path": str(small)})],
            stop_reason="tool_use", usage=_u(10, 5)),
        StepResult(text="done", tool_calls=[], stop_reason="end_turn",
                   usage=_u(10, 5)),
    ])
    # read_file caps its own output at 16k, so the budget must sit below that
    # for the pair of results to breach it at all.
    agent = Agent(provider=fp, system="sys", max_iterations=10, verify=False,
                  truncation_char_budget=10_000)
    agent.run("read both")

    # On the 3rd request the older result is gone but the freshest survives.
    sent = fp.calls[-1]["messages"]
    results = {}
    for m in sent:
        if isinstance(m.get("content"), list):
            for b in m["content"]:
                if b.get("type") == "tool_result":
                    results[b["tool_use_id"]] = str(b["content"])
    assert results["t1"].startswith("[truncated")
    assert "42" in results["t2"],         "the freshest tool_result was blanked before the model ever saw it"


def test_a_single_oversized_fresh_result_is_still_truncated(tmp_workdir):
    # The exemption is a preference, not a guarantee: one tool_result bigger
    # than the whole budget (a giant log dumped to stdout) must still be cut,
    # or the next request is unsendable.
    huge = tmp_workdir / "huge.txt"
    huge.write_text("y" * 60_000)

    fp = FakeProvider([
        StepResult(text="read", tool_calls=[ToolCall(
            id="t1", name="read_file", arguments={"path": str(huge)})],
            stop_reason="tool_use", usage=_u(10, 5)),
        StepResult(text="done", tool_calls=[], stop_reason="end_turn",
                   usage=_u(10, 5)),
    ])
    agent = Agent(provider=fp, system="sys", max_iterations=10, verify=False,
                  truncation_char_budget=10_000)
    agent.run("read the huge one")

    sent = fp.calls[-1]["messages"]
    total = sum(len(str(b.get("content", "")))
                for m in sent if isinstance(m.get("content"), list)
                for b in m["content"])
    assert total < 10_000
