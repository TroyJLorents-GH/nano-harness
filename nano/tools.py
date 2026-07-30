from __future__ import annotations

import os
import queue
import re
import stat
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any


def _resolve_shell() -> tuple[list[str], bool]:
    """Pick the persistent shell. Models emit POSIX/bash commands (pipes, &&,
    heredocs, `;`), so we use bash everywhere it exists — including Windows,
    where Git ships bash.exe. cmd.exe is a last resort only; it cannot run the
    commands models actually write. Returns (argv, is_cmd)."""
    if sys.platform != "win32":
        return ["bash", "--norc", "--noprofile"], False
    # Prefer Git Bash. `shutil.which("bash")` on Windows usually resolves to
    # C:\Windows\System32\bash.exe — the WSL launcher, which runs in a separate
    # /mnt/c filesystem namespace and breaks the Windows paths our file tools
    # use. Only accept a `which` result that is not that WSL shim.
    candidates = [
        r"C:\Program Files\Git\bin\bash.exe",
        r"C:\Program Files (x86)\Git\bin\bash.exe",
    ]
    found = shutil.which("bash")
    if found and "system32" not in found.lower():
        candidates.insert(0, found)
    bash = next((p for p in candidates if os.path.exists(p)), None)
    if bash:
        return [bash, "--norc", "--noprofile"], False
    return ["cmd.exe", "/Q", "/K", "prompt $G"], True


class ToolError(Exception):
    """Raised when a tool call fails. The message is shown to the model."""


_OUTPUT_LIMIT = 16_000  # chars; spec §3.3 leaves "large output truncation" to impl


def _truncate(text: str, limit: int = _OUTPUT_LIMIT) -> str:
    if len(text) <= limit:
        return text
    head = text[: limit // 2]
    tail = text[-limit // 2 :]
    dropped = len(text) - limit
    return f"{head}\n... [truncated {dropped} chars] ...\n{tail}"


def _strip_cmd_prompt(text: str) -> str:
    """Remove cmd.exe '>' prompt artifacts from captured shell output (Windows)."""
    out_lines: list[str] = []
    for line in text.splitlines(keepends=True):
        stripped = line.lstrip()
        if stripped.startswith(">"):
            stripped = stripped[1:]
            # Drop the line entirely if it's now blank.
            if not stripped.strip():
                continue
            out_lines.append(stripped)
        else:
            out_lines.append(line)
    return "".join(out_lines)


class BashTool:
    """Persistent shell. Each `run()` writes the command followed by a sentinel
    echo, then reads stdout until the sentinel appears. Cwd, env, and shell
    state survive between calls. On timeout we kill the shell and start a new
    one — bash supports nothing reliable here.
    """

    def __init__(self) -> None:
        self._proc: subprocess.Popen | None = None
        self._spawn()

    def _spawn(self) -> None:
        cmd, self._is_cmd = _resolve_shell()
        # Put the shell in its own process group / session so a timeout can kill
        # the whole tree (the shell AND its children), not just the shell.
        kw: dict[str, Any] = {}
        if sys.platform == "win32":
            kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kw["start_new_session"] = True
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",  # binary output must not kill the reader thread
            bufsize=1,
            env={**os.environ, "PS1": "", "PROMPT_COMMAND": ""},
            **kw,
        )
        # Each generation gets its OWN queue, bound to its OWN reader thread.
        # A thread from a killed shell keeps writing to its now-orphaned queue,
        # so leftover output can never contaminate the respawned shell's stream.
        self._lines: queue.Queue[str] = queue.Queue()
        threading.Thread(target=self._pump, args=(self._proc, self._lines),
                         daemon=True).start()

    @staticmethod
    def _pump(proc: subprocess.Popen, q: queue.Queue[str]) -> None:
        for line in proc.stdout:  # type: ignore[union-attr]
            q.put(line)

    def run(self, command: str, timeout: int = 300) -> str:
        if self._proc is None or self._proc.poll() is not None:
            self._spawn()
        sentinel = f"__NANO_DONE_{uuid.uuid4().hex}__"
        # The sentinel carries the command's exit status ("<sentinel>:<code>").
        # Without it a failed command (`false`, a failing test, `grep` with no
        # match) reads as success. The empty echo first guarantees the sentinel
        # starts a fresh line even when the command's output lacks a trailing
        # newline; the anchored match below then only accepts a *complete*
        # sentinel line - `set -x` traces ('+ echo <sentinel>:...') must never
        # be mistaken for the real thing.
        nl = "\r\n" if self._is_cmd else "\n"
        if self._is_cmd:
            # Single parse line: %errorlevel% expands before echo. runs.
            tail = f"echo.&echo {sentinel}:%errorlevel%"
        else:
            tail = f"__nano_rc=$?; echo; echo {sentinel}:$__nano_rc"
        assert self._proc and self._proc.stdin
        self._proc.stdin.write(f"{command}{nl}{tail}{nl}")
        self._proc.stdin.flush()

        sentinel_re = re.compile(re.escape(sentinel) + r":(-?\d+)\s*$")
        deadline = time.monotonic() + timeout
        out_lines: list[str] = []
        exit_code = 0
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._kill()
                raise ToolError(
                    f"Command exceeded timeout of {timeout}s and was killed: "
                    f"{command!r}. The shell was restarted: cwd, env vars, and "
                    f"background processes are reset. Re-establish state if "
                    f"needed; pass a larger timeout for long commands."
                )
            try:
                line = self._lines.get(timeout=min(remaining, 0.5))
            except queue.Empty:
                if self._proc.poll() is not None:
                    raise ToolError("Shell process exited unexpectedly.")
                continue
            m = sentinel_re.match(line)
            if m:
                exit_code = int(m.group(1))  # anchored: never unparseable
                break
            out_lines.append(line)

        joined = "".join(out_lines).rstrip("\r\n") + "\n"
        if self._is_cmd:
            joined = _strip_cmd_prompt(joined)
        if exit_code != 0:
            # Raise, don't return: the loop marks ToolError results
            # is_error=True, so a failing test run can never count as
            # verification evidence. The output rides along for diagnosis.
            raise ToolError(_truncate(joined) + f"[exit code {exit_code}]")
        return _truncate(joined)

    def _kill(self) -> None:
        proc = self._proc
        self._proc = None
        if not proc or proc.poll() is not None:
            return
        # Kill the whole tree, not just the shell - a timed-out build, server,
        # or `nohup ... &` child must not survive the shell's death. Tree-kill
        # failure (access denied, race with exit) falls back to killing the
        # shell itself; the wait reaps it so repeated timeouts can't pile up
        # zombie processes.
        try:
            if sys.platform == "win32":
                r = subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                                   capture_output=True, check=False)
                if r.returncode != 0:
                    proc.kill()
            else:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                proc.kill()
            except OSError:
                pass
        try:
            proc.wait(timeout=5)
        except (subprocess.TimeoutExpired, OSError):
            pass

    def __del__(self) -> None:
        try:
            self._kill()
        except Exception:
            pass

    def close(self) -> None:
        self._kill()


def read_file(path: str, line_start: int | None = None,
              line_end: int | None = None) -> str:
    p = Path(path)
    if not p.exists():
        raise ToolError(f"File not found: {path}")
    if not p.is_file():
        raise ToolError(f"Not a regular file: {path}")
    try:
        text = p.read_text(encoding="utf-8")
    except UnicodeDecodeError as e:
        raise ToolError(f"Cannot decode {path} as UTF-8 (binary file?): {e}")

    lines = text.splitlines()
    start = (line_start or 1) - 1
    end = line_end if line_end is not None else len(lines)
    if start < 0 or start > len(lines):
        raise ToolError(f"line_start {line_start} out of range (1..{len(lines)})")
    selected = lines[start:end]
    return _truncate("\n".join(f"{i + start + 1}\t{ln}"
                               for i, ln in enumerate(selected)) + "\n")


def _raw_span(raw: str, n_start: int, n_end: int) -> tuple[int, int]:
    """Map a [start, end) span in the LF-normalized view of `raw` back to raw
    indices. Each \\r\\n in raw collapses to one \\n in the normalized view."""
    spans = []
    r = n = 0
    for target in (n_start, n_end):
        while n < target:
            r += 2 if raw.startswith("\r\n", r) else 1
            n += 1
        spans.append(r)
    return spans[0], spans[1]


def _write_exact(p: Path, text: str) -> None:
    """Write text verbatim: no newline translation (a one-char edit in an
    LF repo must not rewrite the whole file to CRLF), and atomically via a
    temp file + replace so a crash or disk-full can't leave a truncated file.
    The replacement inherits the original's permission bits - editing
    deploy.sh must not strip its executable bit."""
    if not isinstance(text, str):
        raise ToolError(f"'new' must be a string, got {type(text).__name__}.")
    tmp = p.with_name(f"{p.name}.nano-{uuid.uuid4().hex}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8", newline="")
        try:
            os.chmod(tmp, stat.S_IMODE(os.stat(p).st_mode))
        except OSError:
            pass  # new file: keep the default mode
        os.replace(tmp, p)
    finally:
        if tmp.exists():
            tmp.unlink()


def edit_file(path: str, old: str, new: str) -> str:
    if not isinstance(old, str):
        raise ToolError(f"'old' must be a string, got {type(old).__name__}.")
    p = Path(path)
    if old == "":
        if p.exists():
            raise ToolError(
                f"edit_file with old='' creates a new file but {path} already exists. "
                f"Read it first, then call with the exact text to replace."
            )
        p.parent.mkdir(parents=True, exist_ok=True)
        _write_exact(p, new)
        return f"Created {path} ({len(new)} chars)."

    if not p.exists():
        raise ToolError(f"File not found: {path}")
    # Edit through symlinks: replacing the link itself would silently turn it
    # into a regular file and leave the real target untouched.
    p = p.resolve()
    # Preserve the file's line-ending style: read raw (newline="") and prefer
    # an exact byte match - replacing in place leaves every untouched line's
    # ending alone, even in a mixed CRLF/LF file. Only when that misses (the
    # model sent LF for a CRLF file) fall back to LF-normalized matching,
    # splice into the matched RAW span only, and give the replacement the
    # span's own ending style. Editing one line must never flip other lines'
    # newlines - in either direction.
    with p.open(encoding="utf-8", newline="") as f:  # newline="" preserves \r\n
        raw = f.read()
    if raw.count(old) == 1:
        result = raw.replace(old, new, 1)
    else:
        work = raw.replace("\r\n", "\n")
        old_n, new_n = old.replace("\r\n", "\n"), new.replace("\r\n", "\n")
        count = work.count(old_n)
        if count == 0:
            raise ToolError(
                f"old string not found in {path}. Re-read the file and try again."
            )
        if count > 1:
            raise ToolError(
                f"old string matches {count} places in {path} - must be unique. "
                f"Add surrounding context to disambiguate."
            )
        i = work.index(old_n)
        start, end = _raw_span(raw, i, i + len(old_n))
        span = raw[start:end]
        if "\r\n" in span:
            use_crlf = True
        elif "\n" in span:
            use_crlf = False
        else:  # single-line span: fall back to the file's overall style
            use_crlf = "\r\n" in raw and "\n" not in raw.replace("\r\n", "")
        insert = new_n.replace("\n", "\r\n") if use_crlf else new_n
        result = raw[:start] + insert + raw[end:]
    _write_exact(p, result)
    return f"Edited {path} (1 replacement, {len(old)}->{len(new)} chars)."


TOOLS: list[dict[str, Any]] = [
    {
        "name": "bash",
        "description": (
            "Run a shell command in a persistent session. cwd, env, and shell "
            "state are preserved across calls. Use for running tests, listing "
            "files, building, anything stateful."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "timeout": {"type": "integer", "default": 300,
                            "description": "Seconds before kill. Default 300. "
                            "Set generously for builds, installs, and tests."},
            },
            "required": ["command"],
        },
    },
    {
        "name": "read_file",
        "description": (
            "Read a UTF-8 text file. Returns 1-indexed lines prefixed with "
            "'<n>\\t'. Use line_start/line_end to slice large files."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "line_start": {"type": "integer", "minimum": 1},
                "line_end": {"type": "integer", "minimum": 1},
            },
            "required": ["path"],
        },
    },
    {
        "name": "edit_file",
        "description": (
            "Replace exactly one occurrence of `old` with `new` in the file. "
            "Fails if `old` is missing or matches more than one location. "
            "Pass old='' to create a new file with `new` as content."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old": {"type": "string"},
                "new": {"type": "string"},
            },
            "required": ["path", "old", "new"],
        },
    },
]


def _require(arguments: dict[str, Any], name: str, *keys: str) -> None:
    """A malformed tool call (missing required arg, often from a weaker model
    or a truncated JSON blob) must come back as a ToolError the model can fix,
    never an exception that kills the run."""
    missing = [k for k in keys if k not in arguments]
    if missing:
        raise ToolError(
            f"Tool {name!r} called without required argument(s) "
            f"{', '.join(missing)}. Provided: {sorted(arguments)}. "
            f"Re-issue the call with all required arguments."
        )


def _int_arg(arguments: dict[str, Any], key: str, default: int | None = None) -> int | None:
    """Weak models pass numbers as strings ('5' not 5). Coerce; a value that
    won't parse comes back as a ToolError, never a TypeError that kills the run."""
    value = arguments.get(key)
    if value is None:  # missing OR an explicit JSON null: use the default
        return default
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ToolError(
            f"Argument {key!r} must be an integer, got {value!r}. "
            f"Re-issue the call with an integer value."
        )


def dispatch(name: str, arguments: dict[str, Any], *, bash: BashTool) -> str:
    if name == "bash":
        _require(arguments, name, "command")
        return bash.run(arguments["command"], timeout=_int_arg(arguments, "timeout", 300))
    if name == "read_file":
        _require(arguments, name, "path")
        return read_file(arguments["path"],
                         _int_arg(arguments, "line_start"),
                         _int_arg(arguments, "line_end"))
    if name == "edit_file":
        _require(arguments, name, "path", "old", "new")
        return edit_file(arguments["path"], arguments["old"], arguments["new"])
    raise ToolError(f"Unknown tool: {name}")
