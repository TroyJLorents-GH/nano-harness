from __future__ import annotations

import argparse
import os
import sys

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from .agent import Agent
from .prompts import SYSTEM_PROMPT
from .providers import (AnthropicProvider, OpenAIProvider, Provider,
                        _max_tokens_default, _request_timeout)

_console = Console()


def build_provider(*, model: str, base_url: str | None) -> Provider:
    if base_url:
        # Local OpenAI-compatible servers (vLLM, ollama, llama.cpp) accept any
        # api_key. The openai SDK requires one to instantiate, so supply a
        # placeholder when none is set in the env.
        import openai
        key = os.environ.get("OPENAI_API_KEY") or "sk-local"
        # Timeout scaled to the output ceiling and SDK retries off - the
        # provider's own retry loop owns backoff. Must match
        # OpenAIProvider.__post_init__: this is the path the benchmark takes.
        max_out = _max_tokens_default()
        client = openai.OpenAI(base_url=base_url, api_key=key,
                               timeout=_request_timeout(max_out),
                               max_retries=0)
        return OpenAIProvider(model=model, base_url=base_url, client=client,
                              max_completion_tokens=max_out)
    if model.startswith(("claude", "anthropic")):
        return AnthropicProvider(model=model)
    return OpenAIProvider(model=model)


def _print_event(event: dict) -> None:
    # Everything printed here is untrusted model/shell output. It goes through
    # Text() so rich never parses it as console markup: a build line like
    # "make: *** [/usr/.../shlib.mk:10: x.so] Error 1" reads as an unmatched
    # closing tag and raises MarkupError, which used to kill the whole run.
    et = event["type"]
    if et == "assistant":
        if event.get("text"):
            _console.print(Panel(Text(str(event["text"])), title="assistant",
                                 border_style="cyan"))
        # Log the tool calls themselves, not just their output. Without the
        # inputs a transcript is unreadable: you see what came back but never
        # what the model actually ran.
        for tc in event.get("tool_calls") or []:
            args = ", ".join(f"{k}={v!r}" for k, v in tc.arguments.items())
            _console.print(Panel(Text(f"{tc.name}({args})"), title="tool_call",
                                 border_style="yellow"))
    elif et == "tool_result":
        title = "tool_result" + (" (error)" if event.get("is_error") else "")
        _console.print(Panel(Text(str(event["output"])[:2000]), title=title,
                             border_style="red" if event.get("is_error") else "green"))
    elif et == "stats":
        _console.print(f"[dim]iter={event['iteration']} "
                       f"in={event['input_tokens']} out={event['output_tokens']}[/dim]")


def main(argv: list[str] | None = None) -> int:
    # Windows consoles and pipes default to cp1252; model output is full of
    # unicode. Never let the printer kill a finished run.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(prog="nano")
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="Run the agent on a task description.")
    run.add_argument("task", help="Plain-English task description.")
    run.add_argument("--model", default="claude-opus-4-8")
    run.add_argument("--base-url", default=None,
                     help="OpenAI-compatible base URL (Together, vLLM, etc.).")
    run.add_argument("--max-iterations", type=int, default=30)
    run.add_argument("--max-runtime", type=float, default=None,
                     help="Self-imposed wall clock (seconds). Past 80%% the "
                     "agent is told to wrap up; past 100%% it exits cleanly "
                     "with stop reason max_runtime.")
    args = parser.parse_args(argv)

    # Construction failures (missing SDK/key, no usable shell) happen before
    # Agent.run()'s error boundary - turn them into a clean nonzero exit.
    try:
        provider = build_provider(model=args.model, base_url=args.base_url)
        agent = Agent(provider=provider, system=SYSTEM_PROMPT,
                      max_iterations=args.max_iterations,
                      max_runtime_sec=args.max_runtime, on_event=_print_event)
    except Exception as e:
        _console.print(f"[bold red]setup error:[/] {type(e).__name__}: {e}")
        return 1
    result = agent.run(args.task)
    _console.print(f"\n[bold]stop:[/] {result.stop_reason}  "
                   f"iterations={result.iterations}  "
                   f"in={result.total_input_tokens}  "
                   f"out={result.total_output_tokens}  "
                   f"cache_read={result.total_cache_read_tokens}")
    if result.final_text:
        _console.print(Panel(result.final_text, title="final", border_style="bold"))
    return 0 if result.stop_reason == "end_turn" else 1


if __name__ == "__main__":
    sys.exit(main())
