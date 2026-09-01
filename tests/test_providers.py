from nano.providers import StepResult, ToolCall, Usage


def test_step_result_minimum_fields():
    sr = StepResult(
        text="hello",
        tool_calls=[ToolCall(id="c1", name="bash", arguments={"command": "ls"})],
        stop_reason="tool_use",
        usage=Usage(input_tokens=10, output_tokens=5, cache_read_tokens=0),
    )
    assert sr.text == "hello"
    assert sr.tool_calls[0].name == "bash"
    assert sr.tool_calls[0].arguments == {"command": "ls"}
    assert sr.stop_reason == "tool_use"
    assert sr.usage.input_tokens == 10


def test_step_result_text_only():
    sr = StepResult(text="done", tool_calls=[], stop_reason="end_turn",
                    usage=Usage(input_tokens=1, output_tokens=1, cache_read_tokens=0))
    assert sr.tool_calls == []
    assert sr.stop_reason == "end_turn"


from unittest.mock import MagicMock

from nano.providers import AnthropicProvider


class _AnthroMsg:
    def __init__(self, content, stop_reason, usage):
        self.content = content
        self.stop_reason = stop_reason
        self.usage = usage


class _Block:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def _fake_anthropic_response_with_tool():
    return _AnthroMsg(
        content=[
            _Block(type="text", text="I'll list files."),
            _Block(type="tool_use", id="tu_1", name="bash",
                   input={"command": "ls"}),
        ],
        stop_reason="tool_use",
        usage=MagicMock(input_tokens=120, output_tokens=30,
                        cache_read_input_tokens=80, cache_creation_input_tokens=0),
    )


def test_anthropic_provider_translates_tool_use():
    fake_client = MagicMock()
    fake_client.messages.create.return_value = _fake_anthropic_response_with_tool()
    p = AnthropicProvider(model="claude-opus-4-7", client=fake_client)

    result = p.step(
        messages=[{"role": "user", "content": "list files"}],
        tools=[{"name": "bash", "description": "shell",
                "input_schema": {"type": "object", "properties": {
                    "command": {"type": "string"}}, "required": ["command"]}}],
        system="you help",
    )

    assert result.text == "I'll list files."
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].id == "tu_1"
    assert result.tool_calls[0].name == "bash"
    assert result.tool_calls[0].arguments == {"command": "ls"}
    assert result.stop_reason == "tool_use"
    assert result.usage.input_tokens == 120
    assert result.usage.cache_read_tokens == 80


def test_anthropic_provider_end_turn_text_only():
    msg = _AnthroMsg(
        content=[_Block(type="text", text="done")],
        stop_reason="end_turn",
        usage=MagicMock(input_tokens=5, output_tokens=2,
                        cache_read_input_tokens=0, cache_creation_input_tokens=0),
    )
    fake_client = MagicMock()
    fake_client.messages.create.return_value = msg
    p = AnthropicProvider(model="claude-opus-4-7", client=fake_client)

    result = p.step(messages=[{"role": "user", "content": "hi"}], tools=[], system="s")

    assert result.text == "done"
    assert result.tool_calls == []
    assert result.stop_reason == "end_turn"


def test_anthropic_provider_applies_cache_control_to_system_and_last_user():
    fake_client = MagicMock()
    fake_client.messages.create.return_value = _fake_anthropic_response_with_tool()
    p = AnthropicProvider(model="claude-opus-4-7", client=fake_client)

    p.step(
        messages=[
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "ack"},
            {"role": "user", "content": "second"},
        ],
        tools=[],
        system="SYS",
    )

    kwargs = fake_client.messages.create.call_args.kwargs
    sys_param = kwargs["system"]
    assert isinstance(sys_param, list)
    assert sys_param[0]["text"] == "SYS"
    assert sys_param[0]["cache_control"] == {"type": "ephemeral"}

    last_user = kwargs["messages"][-1]
    assert last_user["role"] == "user"
    last_block = last_user["content"][-1]
    assert last_block["cache_control"] == {"type": "ephemeral"}


def test_anthropic_provider_caches_last_user_not_last_message():
    # Mid-conversation: last message is assistant. Cache should still go on the
    # most recent user turn, not the assistant turn.
    fake_client = MagicMock()
    fake_client.messages.create.return_value = _fake_anthropic_response_with_tool()
    p = AnthropicProvider(model="claude-opus-4-7", client=fake_client)
    p.step(
        messages=[
            {"role": "user", "content": "do thing"},
            {"role": "assistant", "content": "thinking"},
        ],
        tools=[], system="s",
    )
    sent = fake_client.messages.create.call_args.kwargs["messages"]
    assert sent[0]["role"] == "user"
    assert sent[0]["content"][-1]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in sent[1]["content"][-1]


import json as _json

from nano.providers import OpenAIProvider


def _fake_openai_response_with_tool():
    return MagicMock(
        choices=[MagicMock(
            message=MagicMock(
                content="I'll list.",
                tool_calls=[MagicMock(
                    id="call_1",
                    function=MagicMock(name="bash",
                                       arguments=_json.dumps({"command": "ls"})),
                )],
            ),
            finish_reason="tool_calls",
        )],
        usage=MagicMock(prompt_tokens=50, completion_tokens=20),
    )


def test_openai_provider_translates_tool_calls():
    # MagicMock auto-sets `.name`; force it to the literal string we want.
    resp = _fake_openai_response_with_tool()
    resp.choices[0].message.tool_calls[0].function.name = "bash"

    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = resp
    p = OpenAIProvider(model="gpt-5", client=fake_client)

    result = p.step(
        messages=[{"role": "user", "content": "list"}],
        tools=[{"name": "bash", "description": "shell",
                "input_schema": {"type": "object",
                                 "properties": {"command": {"type": "string"}},
                                 "required": ["command"]}}],
        system="sys",
    )

    assert result.text == "I'll list."
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].id == "call_1"
    assert result.tool_calls[0].name == "bash"
    assert result.tool_calls[0].arguments == {"command": "ls"}
    assert result.stop_reason == "tool_use"  # normalized from "tool_calls"
    assert result.usage.input_tokens == 50


def test_openai_provider_end_turn():
    resp = MagicMock(
        choices=[MagicMock(
            message=MagicMock(content="done", tool_calls=None),
            finish_reason="stop",
        )],
        usage=MagicMock(prompt_tokens=3, completion_tokens=1),
    )
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = resp
    p = OpenAIProvider(model="gpt-5", client=fake_client)

    result = p.step(messages=[{"role": "user", "content": "hi"}], tools=[], system="s")

    assert result.text == "done"
    assert result.tool_calls == []
    assert result.stop_reason == "end_turn"  # normalized from "stop"


def test_openai_null_finish_reason_becomes_a_string_stop_reason():
    # Flaky gateways can return finish_reason null. StepResult.stop_reason is
    # typed str, so passing None through would die in pydantic validation and
    # end the run as an error - before the agent's empty-turn guard (built for
    # exactly this case) ever sees the turn.
    resp = MagicMock(
        choices=[MagicMock(
            message=MagicMock(content=None, tool_calls=None),
            finish_reason=None,
        )],
        usage=MagicMock(prompt_tokens=3, completion_tokens=0),
    )
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = resp
    p = OpenAIProvider(model="gpt-5", client=fake_client)

    result = p.step(messages=[{"role": "user", "content": "hi"}], tools=[], system="s")

    assert result.stop_reason == "unknown"
    assert result.text is None
    assert result.tool_calls == []


def test_openai_provider_translates_tool_schema_to_openai_format():
    resp = MagicMock(
        choices=[MagicMock(
            message=MagicMock(content="hi", tool_calls=None),
            finish_reason="stop",
        )],
        usage=MagicMock(prompt_tokens=1, completion_tokens=1),
    )
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = resp
    p = OpenAIProvider(model="gpt-5", client=fake_client)

    p.step(
        messages=[{"role": "user", "content": "x"}],
        tools=[{"name": "bash", "description": "shell",
                "input_schema": {"type": "object",
                                 "properties": {"command": {"type": "string"}},
                                 "required": ["command"]}}],
        system="SYS",
    )

    kwargs = fake_client.chat.completions.create.call_args.kwargs
    assert kwargs["messages"][0] == {"role": "system", "content": "SYS"}
    sent_tool = kwargs["tools"][0]
    assert sent_tool["type"] == "function"
    assert sent_tool["function"]["name"] == "bash"
    assert sent_tool["function"]["parameters"]["required"] == ["command"]


def test_normalize_for_openai_round_trips_assistant_tool_calls():
    from nano.providers import _normalize_for_openai
    out = _normalize_for_openai({
        "role": "assistant",
        "content": [{"type": "text", "text": "I'll list."},
                    {"type": "tool_use", "id": "tu_1", "name": "bash",
                     "input": {"command": "ls"}}],
        "tool_calls": [{"id": "tu_1", "name": "bash",
                        "arguments": {"command": "ls"}}],
    })
    assert len(out) == 1
    assert out[0]["content"] == "I'll list."
    assert out[0]["tool_calls"][0]["function"]["name"] == "bash"
    assert out[0]["tool_calls"][0]["function"]["arguments"] == '{"command": "ls"}'


def test_normalize_for_openai_splits_tool_results_into_role_tool_messages():
    from nano.providers import _normalize_for_openai
    out = _normalize_for_openai({
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "tu_1",
             "content": "hello\n", "is_error": False},
            {"type": "tool_result", "tool_use_id": "tu_2",
             "content": "ERROR: bad", "is_error": True},
        ],
    })
    assert len(out) == 2
    assert out[0] == {"role": "tool", "tool_call_id": "tu_1", "content": "hello\n"}
    assert out[1] == {"role": "tool", "tool_call_id": "tu_2", "content": "ERROR: bad"}


def test_normalize_for_openai_passes_plain_user_through():
    from nano.providers import _normalize_for_openai
    out = _normalize_for_openai({"role": "user", "content": "what"})
    assert out == [{"role": "user", "content": "what"}]


# --- retry behavior ---

from nano.providers import _call_with_retry


class _FlakyError(Exception):
    def __init__(self, status_code):
        self.status_code = status_code


def test_retry_recovers_from_transient_errors(monkeypatch):
    import nano.providers as providers
    monkeypatch.setattr(providers.time, "sleep", lambda s: None)
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise _FlakyError(429)
        return "ok"

    assert _call_with_retry(flaky) == "ok"
    assert calls["n"] == 3


def test_retry_gives_up_after_max_attempts(monkeypatch):
    import nano.providers as providers
    monkeypatch.setattr(providers.time, "sleep", lambda s: None)

    def always_529():
        raise _FlakyError(529)

    import pytest
    with pytest.raises(_FlakyError):
        _call_with_retry(always_529)


def test_retry_does_not_retry_client_errors():
    calls = {"n": 0}

    def bad_request():
        calls["n"] += 1
        raise _FlakyError(400)

    import pytest
    with pytest.raises(_FlakyError):
        _call_with_retry(bad_request)
    assert calls["n"] == 1


def test_retry_covers_client_side_timeouts(monkeypatch):
    # SDK timeout errors (anthropic/openai APITimeoutError) carry no
    # status_code and their concrete class name lacks "Connection" - they
    # must still be classified transient and retried.
    import nano.providers as providers
    monkeypatch.setattr(providers.time, "sleep", lambda s: None)
    calls = {"n": 0}

    class APITimeoutError(Exception):
        pass

    def slow_then_ok():
        calls["n"] += 1
        if calls["n"] < 2:
            raise APITimeoutError("request timed out")
        return "ok"

    assert _call_with_retry(slow_then_ok) == "ok"
    assert calls["n"] == 2


def test_anthropic_provider_retries_through_client(monkeypatch):
    import nano.providers as providers
    monkeypatch.setattr(providers.time, "sleep", lambda s: None)
    fake_client = MagicMock()
    fake_client.messages.create.side_effect = [
        _FlakyError(529),
        _fake_anthropic_response_with_tool(),
    ]
    p = AnthropicProvider(model="claude-opus-4-7", client=fake_client)
    sr = p.step([{"role": "user", "content": "hi"}], [], "sys")
    assert sr.stop_reason == "tool_use"
    assert fake_client.messages.create.call_count == 2


class _OaiMsg:
    def __init__(self, tool_calls):
        self.content = None
        self.tool_calls = tool_calls


class _OaiTC:
    def __init__(self, args, id="c1"):
        self.id = id
        self.type = "function"
        self.function = type("F", (), {"name": "bash", "arguments": args})()


def test_openai_null_tool_call_id_does_not_crash(monkeypatch):
    # Some OpenAI-compatible gateways omit the tool-call id. ToolCall.id is a
    # required str, so Pydantic raised ValidationError, which the agent loop
    # turned into stop_reason="error". Observed: db-wal-recovery died at
    # iteration 13 of a real benchmark run. Synthesize an id instead.
    import nano.providers as providers
    monkeypatch.setattr(providers.time, "sleep", lambda s: None)
    fake_client = MagicMock()
    resp = MagicMock()
    resp.choices = [MagicMock(
        message=_OaiMsg([_OaiTC('{"command": "ls"}', id=None)]),
        finish_reason="tool_calls")]
    resp.usage = MagicMock(prompt_tokens=1, completion_tokens=1)
    fake_client.chat.completions.create.return_value = resp
    p = OpenAIProvider(model="x", client=fake_client)
    sr = p.step([{"role": "user", "content": "hi"}], [], "sys")

    assert len(sr.tool_calls) == 1
    assert sr.tool_calls[0].id, "a usable id must be synthesized"
    assert sr.tool_calls[0].arguments == {"command": "ls"}


def test_openai_empty_choices_is_retried(monkeypatch):
    # A flaky gateway can return 200 with an empty choices list. Indexing it
    # raised IndexError outside the retry classifier (name matches neither
    # Timeout nor Connection), so one blip ended the run. It must retry.
    import nano.providers as providers
    monkeypatch.setattr(providers.time, "sleep", lambda s: None)
    fake_client = MagicMock()
    bad = MagicMock(); bad.choices = []
    good = MagicMock()
    good.choices = [MagicMock(message=_OaiMsg([]), finish_reason="stop")]
    good.choices[0].message.content = "ok"
    good.usage = MagicMock(prompt_tokens=1, completion_tokens=1)
    fake_client.chat.completions.create.side_effect = [bad, good]
    p = OpenAIProvider(model="x", client=fake_client)
    sr = p.step([{"role": "user", "content": "hi"}], [], "sys")

    assert sr.text == "ok"
    assert fake_client.chat.completions.create.call_count == 2


def test_openai_mixed_text_and_tool_results_keeps_tool_messages_adjacent():
    # OpenAI requires role:'tool' replies to IMMEDIATELY follow the assistant
    # message carrying the tool_calls. Inserting user text before them splits
    # the pair and 400s. Text must come after the tool messages.
    from nano.providers import _normalize_for_openai
    out = _normalize_for_openai({"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "t1", "content": "result"},
        {"type": "text", "text": "wrap up now"},
    ]})
    assert out[0]["role"] == "tool", f"tool reply must come first, got {out}"
    assert out[-1]["role"] == "user"


def test_openai_synthesized_ids_unique_across_turns(monkeypatch):
    # call_{index} restarts at call_0 in every response, so a gateway that
    # omits ids on two different turns puts two distinct tool calls with the
    # SAME id into the conversation - and tool_results pair by id. Synthesized
    # ids must be unique across the whole conversation, not just one response.
    import nano.providers as providers
    monkeypatch.setattr(providers.time, "sleep", lambda s: None)
    fake_client = MagicMock()
    resp = MagicMock()
    resp.choices = [MagicMock(
        message=_OaiMsg([_OaiTC('{"command": "ls"}', id=None)]),
        finish_reason="tool_calls")]
    resp.usage = MagicMock(prompt_tokens=1, completion_tokens=1)
    fake_client.chat.completions.create.return_value = resp
    p = OpenAIProvider(model="x", client=fake_client)

    id1 = p.step([{"role": "user", "content": "hi"}], [], "sys").tool_calls[0].id
    id2 = p.step([{"role": "user", "content": "hi"}], [], "sys").tool_calls[0].id

    assert id1 != id2, "synthesized ids collide across turns"


def test_openai_wrong_type_tool_args_wrapped(monkeypatch):
    import nano.providers as providers
    monkeypatch.setattr(providers.time, "sleep", lambda s: None)
    fake_client = MagicMock()
    resp = MagicMock()
    resp.choices = [MagicMock(message=_OaiMsg([_OaiTC("null")]), finish_reason="tool_calls")]
    resp.usage = MagicMock(prompt_tokens=1, completion_tokens=1)
    fake_client.chat.completions.create.return_value = resp
    p = OpenAIProvider(model="x", client=fake_client)
    sr = p.step([{"role": "user", "content": "hi"}], [], "sys")
    # 'null' parses to None (not a dict) -> must be wrapped, not crash Pydantic
    assert sr.tool_calls[0].arguments == {"_raw": "null"}


def test_openai_truncated_tool_input_not_reinflated():
    # A tool_use block whose input was truncated in history must serialize to
    # OpenAI with the truncated value - not a stale full copy. Regression for
    # the dual-storage divergence.
    from nano.providers import _normalize_for_openai
    assistant = {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "writing"},
            {"type": "tool_use", "id": "t1", "name": "edit_file",
             "input": {"path": "big.py", "old": "",
                       "new": "[truncated - 5000 chars dropped]"}},
        ],
    }
    out = _normalize_for_openai(assistant)
    args = out[0]["tool_calls"][0]["function"]["arguments"]
    assert "truncated" in args
    assert "5000 chars dropped" in args
    assert len(args) < 200  # the giant value is gone


# --- E8: adversarial-review fix set -------------------------------------


def test_gateway_504_is_retried_like_other_5xx(monkeypatch):
    # The openai SDK maps every unlisted status >= 500 to InternalServerError,
    # whose class name matches neither "Connection" nor "Timeout". A 504 from
    # the proxy therefore ended the trial on the first attempt, while the very
    # same blip returning 502 survived three attempts with backoff.
    class InternalServerError(Exception):
        def __init__(self, status_code):
            super().__init__(f"server error {status_code}")
            self.status_code = status_code

    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise InternalServerError(504)
        return "ok"

    import nano.providers as providers
    monkeypatch.setattr(providers.time, "sleep", lambda s: None)
    assert _call_with_retry(flaky) == "ok"
    assert calls["n"] == 3


def test_client_error_is_still_not_retried():
    class BadRequestError(Exception):
        def __init__(self):
            super().__init__("bad request")
            self.status_code = 400

    calls = {"n": 0}

    def bad():
        calls["n"] += 1
        raise BadRequestError()

    try:
        _call_with_retry(bad)
    except BadRequestError:
        pass
    else:
        raise AssertionError("a 400 must not be retried or swallowed")
    assert calls["n"] == 1


def test_request_timeout_scales_with_the_output_ceiling():
    # A non-streaming request returns nothing until the whole body is ready, so
    # a 16k-token generation needs more than 120s of headroom. The old fixed
    # 120s made large file writes time out deterministically on every retry.
    from nano.providers import _request_timeout
    assert _request_timeout(8192) > 120
    assert _request_timeout(16384) > _request_timeout(8192)
    assert _request_timeout(64000) <= 600  # still bounded
