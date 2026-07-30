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
import shlex
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
            for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENAI_BASE_URL")
            if (v := os.environ.get(k))
        }
        # `|| true`: a partial run (max_iterations) may still pass the tests —
        # never let the agent's exit code abort the trial before grading.
        await self.exec_as_agent(
            environment,
            # 250 iterations: the wall clock is the real budget. 100 was
            # binding on 46 of 89 trials, and several failed runs hit it with
            # half their time unused (caffe-cifar-10: iter=100 at 2380s of a
            # 3600s budget). The clock kills slow tasks first either way, so
            # a higher cap can't cost anything there.
            f'"$HOME/.local/bin/nano" run {shlex.quote(instruction)} '
            f"--model {shlex.quote(model)} --max-iterations 250 "
            "</dev/null 2>&1 | tee /logs/agent/nano.txt || true",
            env=env,
        )
