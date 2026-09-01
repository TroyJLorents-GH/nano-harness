"""Terminal-Bench 2.0 adapter (Harbor installed agent).

Uploads the local nano-harness checkout into each task container, installs it
with uv, and runs `nano run "<instruction>"` inside. Harbor owns the sandbox,
timeouts, and grading; we own nothing but the agent.

Usage (host needs Docker running):

    pip install harbor
    export ANTHROPIC_API_KEY=...
    harbor run -d terminal-bench/terminal-bench-2 \
        --agent-import-path eval.tb_agent:NanoAgent \
        -m anthropic/claude-opus-4-7 \
        -n 4 --jobs-dir results/terminal-bench

Smoke test on a few tasks first: add `-t hello-world` (repeatable flag).
Results land in results/terminal-bench/<timestamp>/result.json; per-task agent
stdout in <task>/agent/nano.txt.
"""
from __future__ import annotations

import os
import re
import shlex
import tomllib
from pathlib import Path

from harbor.agents.installed.base import BaseInstalledAgent, with_prompt_template
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

_REPO_ROOT = Path(__file__).resolve().parent.parent
_REMOTE_DIR = "/installed-agent/nano-harness"

# Task images vary (debian, alpine, ...); make sure curl exists, then let uv
# bring its own Python so we never depend on the image's python3.
_ENSURE_CURL = (
    "command -v curl >/dev/null 2>&1 || { "
    "command -v apt-get >/dev/null && apt-get update && apt-get install -y curl; } || { "
    "command -v apk >/dev/null && apk add --no-cache curl bash; } || { "
    "command -v dnf >/dev/null && dnf install -y curl; } || { "
    "command -v yum >/dev/null && yum install -y curl; }"
)

_INSTALL_NANO = (
    "set -eu; "
    "curl -LsSf https://astral.sh/uv/install.sh | sh && "
    f'"$HOME/.local/bin/uv" tool install --python 3.12 {_REMOTE_DIR} && '
    '"$HOME/.local/bin/nano" --help >/dev/null'
)


def _task_agent_timeout_sec(environment: BaseEnvironment) -> float | None:
    """Read [agent] timeout_sec from the task's task.toml on the HOST.

    Harbor enforces the agent deadline host-side and deliberately passes it
    only to its Oracle agent, so the in-container agent cannot learn it from
    Harbor's plumbing. The adapter, however, runs host-side and
    environment.environment_dir points into the task directory, whose parent
    holds task.toml. This reads benchmark METADATA (a timeout), never the
    solution. It is gated behind NANO_USE_DEADLINE=1 so any run intended for
    leaderboard submission can omit it if the maintainers rule it out.
    """
    try:
        toml_path = Path(environment.environment_dir).parent / "task.toml"
        with open(toml_path, "rb") as f:
            data = tomllib.load(f)
        val = (data.get("agent") or {}).get("timeout_sec")
        return float(val) if val else None
    except Exception:
        # Regex fallback for a malformed-but-readable file; None otherwise.
        try:
            text = toml_path.read_text(encoding="utf-8", errors="replace")
            m = re.search(r"^\s*timeout_sec\s*=\s*([0-9.]+)", text, re.M)
            return float(m.group(1)) if m else None
        except Exception:
            return None


class NanoAgent(BaseInstalledAgent):
    """nano-harness as a Terminal-Bench 2.0 agent. The loop, tools, and system
    prompt are exactly what `nano run` ships locally — no benchmark forks."""

    @staticmethod
    def name() -> str:
        return "nano"

    def get_version_command(self) -> str | None:
        return (
            '"$HOME/.local/bin/uv" tool run --from /installed-agent/nano-harness '
            "python -c \"import nano; print(nano.__version__)\""
        )

    async def install(self, environment: BaseEnvironment) -> None:
        await environment.upload_dir(_REPO_ROOT / "nano", f"{_REMOTE_DIR}/nano")
        await environment.upload_dir(_REPO_ROOT / "eval", f"{_REMOTE_DIR}/eval")
        await environment.upload_file(
            _REPO_ROOT / "pyproject.toml", f"{_REMOTE_DIR}/pyproject.toml"
        )
        await self.exec_as_root(
            environment, _ENSURE_CURL, env={"DEBIAN_FRONTEND": "noninteractive"}
        )
        await self.exec_as_agent(environment, _INSTALL_NANO)

    @with_prompt_template
    async def run(
        self, instruction: str, environment: BaseEnvironment, context: AgentContext
    ) -> None:
        # Harbor model names look like "anthropic/claude-opus-4-7"; nano's
        # provider routing wants the bare model name. Exception: when routing
        # through an OpenAI-compatible gateway (OPENAI_BASE_URL set), the full
        # "provider/name" string IS the gateway's model id - pass it through.
        model = self.model_name or "anthropic/claude-opus-4-7"
        if not os.environ.get("OPENAI_BASE_URL"):
            model = model.split("/", 1)[-1]
        env = {
            k: v
            # NANO_MAX_TOKENS matters here: Harbor does not propagate host
            # env into the container, so a runner that sets the output ceiling
            # on the host had no effect at all until it was forwarded.
            for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENAI_BASE_URL",
                      "NANO_MAX_TOKENS")
            if (v := os.environ.get(k))
        }
        # Deadline-aware clean exit, opt-in via NANO_USE_DEADLINE=1: pass the
        # task's own budget (minus a 60s teardown margin) so nano exits
        # cleanly before Harbor's kill. An external kill is a forced zero on
        # the official metric even when the workspace would pass; a clean
        # exit is graded. See _task_agent_timeout_sec for the legality note.
        runtime_flag = ""
        if os.environ.get("NANO_USE_DEADLINE") == "1":
            budget = _task_agent_timeout_sec(environment)
            if budget and budget > 120:
                runtime_flag = f"--max-runtime {int(budget) - 60} "
        # `|| true`: a partial run (max_iterations) may still pass the tests —
        # never let the agent's exit code abort the trial before grading.
        await self.exec_as_agent(
            environment,
            # 130 iterations: the fallback guard when no deadline is passed.
            # Derived from measured iteration timing (fastest ~4.7s/iter ->
            # ~610s fits the smallest 900s budget; heaviest ~24-31s/iter fits
            # 3600s+). A clean max_iterations exit still gets graded (35 such
            # passes in the 2.0 run).
            f'"$HOME/.local/bin/nano" run {shlex.quote(instruction)} '
            f"--model {shlex.quote(model)} --max-iterations 130 "
            f"{runtime_flag}"
            "</dev/null 2>&1 | tee /logs/agent/nano.txt || true",
            env=env,
        )
