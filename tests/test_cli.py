from unittest.mock import MagicMock, patch

from nano.cli import build_provider, main


def test_build_provider_anthropic_by_default(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-x")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    p = build_provider(model="claude-opus-4-7", base_url=None)
    assert p.__class__.__name__ == "AnthropicProvider"


def test_build_provider_openai_for_gpt_model(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    p = build_provider(model="gpt-5", base_url=None)
    assert p.__class__.__name__ == "OpenAIProvider"


def test_build_provider_openai_with_base_url():
    p = build_provider(model="local/llama3", base_url="http://localhost:8000/v1")
    assert p.__class__.__name__ == "OpenAIProvider"
    assert p.base_url == "http://localhost:8000/v1"


def test_main_runs_agent_and_prints(monkeypatch, capsys):
    fake_agent = MagicMock()
    fake_agent.run.return_value = MagicMock(
        final_text="DONE",
        stop_reason="end_turn",
        iterations=2,
        total_input_tokens=100,
        total_output_tokens=20,
        total_cache_read_tokens=80,
    )
    with patch("nano.cli.Agent", return_value=fake_agent), \
         patch("nano.cli.build_provider", return_value=MagicMock(model="m")):
        rc = main(["run", "do the thing", "--model", "m"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "DONE" in out


def test_main_setup_failure_is_clean_exit_not_traceback(capsys):
    # Provider/agent construction failures (missing SDK, no bash, bad key
    # config) happen before Agent.run()'s error boundary. The CLI must turn
    # them into a printed error + exit 1, never an uncaught traceback.
    with patch("nano.cli.build_provider",
               side_effect=RuntimeError("no api key configured")):
        rc = main(["run", "task", "--model", "m"])
    assert rc == 1
    out = capsys.readouterr().out
    assert "no api key configured" in out


def test_print_event_shows_tool_call_inputs(capsys):
    from nano.cli import _print_event
    from nano.providers import ToolCall

    _print_event({"type": "assistant", "text": None,
                  "tool_calls": [ToolCall(id="t1", name="bash",
                                          arguments={"command": "ls -la"})]})
    out = capsys.readouterr().out
    assert "bash(" in out
    assert "ls -la" in out


# Build tools print things like "make: *** [/usr/share/make/shlib.mk:10: x.so]
# Error 1". Rich parses "[...]" as console markup, and a "[/...]" fragment reads
# as a closing tag with no opener -> MarkupError. The printer runs inside the
# agent loop's try, so this turned ordinary build output into a dead run.
MARKUP_BOMB = "make: *** [/usr/share/R/share/make/shlib.mk:10: RcppEigen.so] Error 1"


def test_print_event_survives_markup_in_tool_output(capsys):
    from nano.cli import _print_event

    _print_event({"type": "tool_result", "output": MARKUP_BOMB, "is_error": True})
    assert "shlib.mk" in capsys.readouterr().out


def test_print_event_survives_markup_in_assistant_text_and_args(capsys):
    from nano.cli import _print_event
    from nano.providers import ToolCall

    _print_event({"type": "assistant", "text": MARKUP_BOMB, "tool_calls": []})
    _print_event({"type": "assistant", "text": None,
                  "tool_calls": [ToolCall(id="t1", name="bash",
                                          arguments={"command": MARKUP_BOMB})]})
    assert "shlib.mk" in capsys.readouterr().out
