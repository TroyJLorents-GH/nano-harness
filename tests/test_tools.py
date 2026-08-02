import os
import sys

import pytest

from nano.tools import TOOLS, BashTool, ToolError, dispatch, edit_file, read_file


@pytest.fixture
def bash():
    t = BashTool()
    yield t
    t.close()


def test_bash_runs_simple_command(bash):
    out = bash.run("echo hello", timeout=5)
    assert "hello" in out


def test_bash_default_timeout_fits_real_builds():
    # 61 of 64 timeout kills in the measured 89-task run were at the old 60s
    # default; compile-compcert alone lost ~44% of its runtime to them, and
    # every kill destroys the shell (cwd/env/background state) on top of the
    # wasted wait. The default must accommodate a compile or test suite.
    import inspect
    sig_default = inspect.signature(BashTool.run).parameters["timeout"].default
    assert sig_default == 300
    schema = next(t for t in TOOLS if t["name"] == "bash")
    assert schema["input_schema"]["properties"]["timeout"]["default"] == 300


def test_bash_kill_reaps_shell_and_respawns(bash):
    # After a kill, the old shell must be reaped (not left as a zombie) and
    # the next run() must transparently respawn a working shell.
    old = bash._proc
    bash.close()
    assert old.poll() is not None, "shell not reaped after kill"
    out = bash.run("echo back", timeout=5)
    assert "back" in out


def test_bash_cwd_persists_across_calls(bash, tmp_workdir):
    sub = tmp_workdir / "sub"
    sub.mkdir()
    bash.run(f'cd "{sub}"', timeout=5)
    out = bash.run("cd" if bash._is_cmd else "pwd", timeout=5)
    assert "sub" in out


def test_bash_env_persists_across_calls(bash):
    bash.run("set NANO_TEST=42" if bash._is_cmd
             else "export NANO_TEST=42", timeout=5)
    out = bash.run("echo %NANO_TEST%" if bash._is_cmd
                   else "echo $NANO_TEST", timeout=5)
    assert "42" in out


def test_bash_handles_posix_compound_commands(bash):
    # Models write bash: pipes, &&, ; . These must work on every platform
    # (the t2 head-to-head failure was cmd.exe choking on exactly this).
    if bash._is_cmd:
        pytest.skip("no bash available; cmd.exe fallback cannot run POSIX")
    out = bash.run("echo one && echo two ; echo three | cat", timeout=5)
    assert "one" in out and "two" in out and "three" in out


def test_bash_timeout_raises_or_reports(bash):
    with pytest.raises(ToolError) as exc:
        bash.run("ping -n 6 127.0.0.1 > nul" if bash._is_cmd
                 else "sleep 5", timeout=1)
    assert "timeout" in str(exc.value).lower()


def test_bash_truncates_huge_output(bash):
    cmd = ("python -c \"print('x'*200000)\"")
    out = bash.run(cmd, timeout=10)
    assert len(out) < 200000
    assert "truncated" in out.lower()


def test_bash_output_has_no_prompt_or_command_echo(bash):
    # Output must be just the command's stdout - no shell prompt, no echoed
    # stdin. A leaked prompt or echo would confuse the model every turn.
    out = bash.run("echo clean", timeout=5)
    assert out.strip() == "clean", f"prompt or echo leaked: {out!r}"


def test_read_file_full(tmp_workdir):
    p = tmp_workdir / "a.txt"
    p.write_text("line1\nline2\nline3\n")
    out = read_file(str(p))
    assert "1\tline1" in out
    assert "2\tline2" in out
    assert "3\tline3" in out


def test_read_file_range(tmp_workdir):
    p = tmp_workdir / "a.txt"
    p.write_text("\n".join(f"row{i}" for i in range(1, 11)) + "\n")
    out = read_file(str(p), line_start=3, line_end=5)
    assert "3\trow3" in out
    assert "5\trow5" in out
    assert "row1" not in out
    assert "row6" not in out


def test_read_file_missing_raises(tmp_workdir):
    with pytest.raises(ToolError) as exc:
        read_file(str(tmp_workdir / "no.txt"))
    assert "not found" in str(exc.value).lower()


def test_read_file_binary_rejected(tmp_workdir):
    p = tmp_workdir / "blob.bin"
    p.write_bytes(b"\x00\x01\x02\x03BINARY\xff")
    with pytest.raises(ToolError) as exc:
        read_file(str(p))
    assert "binary" in str(exc.value).lower() or "decode" in str(exc.value).lower()


def test_edit_file_replaces_unique(tmp_workdir):
    p = tmp_workdir / "x.py"
    p.write_text("a = 1\nb = 2\n")
    out = edit_file(str(p), old="b = 2", new="b = 99")
    assert "edited" in out.lower()
    assert p.read_text() == "a = 1\nb = 99\n"


def test_edit_file_creates_when_old_empty(tmp_workdir):
    p = tmp_workdir / "new.py"
    out = edit_file(str(p), old="", new="print('hi')\n")
    assert p.read_text() == "print('hi')\n"
    assert "created" in out.lower()


def test_edit_file_non_unique_raises(tmp_workdir):
    p = tmp_workdir / "y.py"
    p.write_text("x = 1\nx = 1\n")
    with pytest.raises(ToolError) as exc:
        edit_file(str(p), old="x = 1", new="x = 2")
    assert "non-unique" in str(exc.value).lower() or "matches 2" in str(exc.value).lower()
    assert p.read_text() == "x = 1\nx = 1\n"  # unchanged


def test_edit_file_no_match_raises(tmp_workdir):
    p = tmp_workdir / "z.py"
    p.write_text("hello\n")
    with pytest.raises(ToolError) as exc:
        edit_file(str(p), old="nope", new="something")
    assert "not found" in str(exc.value).lower() or "no match" in str(exc.value).lower()


def test_registry_has_three_tools():
    assert {t["name"] for t in TOOLS} == {"bash", "read_file", "edit_file"}
    for t in TOOLS:
        assert "description" in t and t["description"]
        assert t["input_schema"]["type"] == "object"


def test_dispatch_runs_read_file(tmp_workdir, bash):
    p = tmp_workdir / "f.txt"
    p.write_text("hello\n")
    out = dispatch("read_file", {"path": str(p)}, bash=bash)
    assert "hello" in out


def test_dispatch_unknown_tool(bash):
    with pytest.raises(ToolError) as exc:
        dispatch("nope", {}, bash=bash)
    assert "unknown tool" in str(exc.value).lower()


def test_dispatch_missing_required_arg_raises_toolerror(bash):
    # A malformed tool call (weak model, truncated JSON) must surface as a
    # ToolError the loop feeds back, never an uncaught KeyError.
    with pytest.raises(ToolError) as exc:
        dispatch("bash", {}, bash=bash)
    assert "command" in str(exc.value)

    with pytest.raises(ToolError):
        dispatch("edit_file", {"path": "x"}, bash=bash)


def test_dispatch_coerces_string_numbers(tmp_workdir, bash):
    # Weak models pass numbers as strings ("5" not 5). A TypeError here
    # killed a whole benchmark task - coerce, don't crash.
    p = tmp_workdir / "n.txt"
    p.write_text("\n".join(f"row{i}" for i in range(1, 11)) + "\n")
    out = dispatch("read_file", {"path": str(p), "line_start": "3",
                                 "line_end": "5"}, bash=bash)
    assert "3\trow3" in out and "row6" not in out
    out = dispatch("bash", {"command": "echo hi", "timeout": "5"}, bash=bash)
    assert "hi" in out


def test_dispatch_unparseable_number_is_toolerror(bash):
    with pytest.raises(ToolError) as exc:
        dispatch("read_file", {"path": "x", "line_start": "abc"}, bash=bash)
    assert "line_start" in str(exc.value)


def test_dispatch_explicit_null_optional_arg_uses_default(bash):
    # Weak models send '"timeout": null' instead of omitting the key.
    # dict.get only defaults on a MISSING key, so an explicit None must be
    # mapped to the default - not passed through to `time.monotonic() + None`.
    out = dispatch("bash", {"command": "echo nulled", "timeout": None},
                   bash=bash)
    assert "nulled" in out


def test_bash_surfaces_nonzero_exit(bash):
    # A failed command must not read as success. It must raise ToolError so
    # the loop records is_error=True and the verify gate never counts a
    # failing test run as evidence of completed work.
    with pytest.raises(ToolError) as exc:
        bash.run("false", timeout=5)
    assert "exit code 1" in str(exc.value).lower()


def test_bash_reports_specific_exit_code(bash):
    if bash._is_cmd:
        pytest.skip("cmd.exe subshell exit differs")
    with pytest.raises(ToolError) as exc:
        bash.run("(exit 7)", timeout=5)
    assert "exit code 7" in str(exc.value).lower()


def test_bash_failed_command_output_still_visible(bash):
    if bash._is_cmd:
        pytest.skip("posix echo")
    # The model needs the failing command's output to diagnose it - the error
    # must carry what was printed before the nonzero exit.
    with pytest.raises(ToolError) as exc:
        bash.run("echo the reason; false", timeout=5)
    assert "the reason" in str(exc.value)


def test_bash_clean_output_on_success(bash):
    # Exit 0 stays clean - no exit-code noise appended to good output.
    out = bash.run("echo hi", timeout=5)
    assert out.strip() == "hi"


def test_bash_grep_no_match_is_visible(bash):
    if bash._is_cmd:
        pytest.skip("grep semantics")
    # grep with no match exits 1 - the agent must see that, not think it passed.
    with pytest.raises(ToolError) as exc:
        bash.run("echo apple | grep zebra", timeout=5)
    assert "exit code 1" in str(exc.value).lower()


def test_bash_set_x_does_not_break_sentinel_framing(bash):
    if bash._is_cmd:
        pytest.skip("bash tracing")
    # `set -x` traces every command to stderr (merged into stdout), including
    # the sentinel echo itself. A substring sentinel match latches onto the
    # trace line and mis-frames every subsequent call.
    bash.run("set -x", timeout=5)
    out = bash.run("printf 'FRESH\\n'", timeout=5)
    assert "FRESH" in out
    # No bare sentinel line may leak into the output - trace lines are
    # prefixed ('+ echo ...') and must not be mistaken for the real sentinel.
    import re
    assert not re.search(r"^__NANO_DONE_[0-9a-f]+__:", out, re.M), out
    bash.run("set +x", timeout=5)


def test_bash_stdin_reader_returns_immediately_not_timeout(bash):
    if bash._is_cmd:
        pytest.skip("posix stdin semantics")
    # A command that reads stdin (bare cat, git commit opening an editor, a
    # REPL) used to consume the sentinel line and burn the ENTIRE timeout -
    # 300s, a third of a 900s task budget - before the shell was killed.
    # The command group runs with stdin from /dev/null: instant EOF instead.
    import time as _t
    t0 = _t.monotonic()
    out = bash.run("cat", timeout=30)
    assert _t.monotonic() - t0 < 10, "stdin reader hung instead of EOF"
    assert "NANO_DONE" not in out


def test_bash_stdin_guard_preserves_shell_state(bash):
    if bash._is_cmd:
        pytest.skip("posix")
    # The guard must be a brace group, not a subshell: cwd/env changes made
    # inside the command must still persist to the next call.
    bash.run("export NANO_GUARD_TEST=alive", timeout=5)
    out = bash.run("echo $NANO_GUARD_TEST", timeout=5)
    assert "alive" in out


def test_bash_heredoc_still_works_under_stdin_guard(bash):
    if bash._is_cmd:
        pytest.skip("posix")
    # Heredocs carry their own stdin; the group-level /dev/null redirection
    # must not starve them.
    out = bash.run("cat <<'EOF'\nheredoc-payload\nEOF", timeout=5)
    assert "heredoc-payload" in out


def test_bash_output_without_trailing_newline(bash):
    if bash._is_cmd:
        pytest.skip("posix printf")
    # A command whose output lacks a trailing newline must still frame
    # correctly and return that output intact.
    out = bash.run("printf 'x'", timeout=5)
    assert out.strip() == "x"


def test_bash_timeout_does_not_contaminate_next_command(bash):
    # A timed-out command's leftover output must NOT leak into the next run.
    # (Old reader thread writing into the respawned shell's queue.)
    if bash._is_cmd:
        pytest.skip("posix subshell timing")
    with pytest.raises(ToolError):
        bash.run("(sleep 2; echo LATE_LEAK) & wait", timeout=1)
    out = bash.run("echo FRESH", timeout=5)
    assert "FRESH" in out
    assert "LATE_LEAK" not in out
    assert "NANO_DONE" not in out  # no stale sentinel from the killed shell


def test_edit_file_preserves_lf_newlines(tmp_workdir):
    # A one-char edit in an LF file must NOT rewrite the whole file to CRLF.
    p = tmp_workdir / "lf.py"
    p.write_bytes(b"a = 1\nb = 2\nc = 3\n")
    edit_file(str(p), old="b = 2", new="b = 99")
    raw = p.read_bytes()
    assert b"\r\n" not in raw, "edit introduced CRLF"
    assert raw == b"a = 1\nb = 99\nc = 3\n"


def test_edit_file_rejects_non_string_args(tmp_workdir):
    p = tmp_workdir / "x.py"
    p.write_text("a = 1\n")
    with pytest.raises(ToolError):
        edit_file(str(p), old=None, new="y")
    assert p.read_text() == "a = 1\n"  # untouched


def test_edit_file_preserves_crlf_newlines(tmp_workdir):
    # Editing one line in a CRLF file must NOT convert the whole file to LF,
    # and `old` sent with LF must still match a CRLF file.
    p = tmp_workdir / "crlf.py"
    p.write_bytes(b"a = 1\r\nb = 2\r\nc = 3\r\n")
    edit_file(str(p), old="b = 2", new="b = 99")  # old uses LF
    raw = p.read_bytes()
    assert raw == b"a = 1\r\nb = 99\r\nc = 3\r\n", raw
    assert b"\r\n" in raw and raw.count(b"\r\n") == 3


def test_edit_file_mixed_newlines_leaves_untouched_lines_alone(tmp_workdir):
    # A file with BOTH endings (one stray CRLF in an LF file, or vice versa)
    # must come back byte-identical outside the edited region - editing line 1
    # must not homogenize every other line's ending.
    p = tmp_workdir / "mixed.txt"
    p.write_bytes(b"a = 1\nb = 2\r\nc = 3\nd = 4\n")
    edit_file(str(p), old="a = 1", new="a = 99")
    assert p.read_bytes() == b"a = 99\nb = 2\r\nc = 3\nd = 4\n", p.read_bytes()


def test_edit_file_mixed_newlines_fallback_touches_only_matched_span(tmp_workdir):
    # LF-sent `old` spanning CRLF lines forces the normalized fallback path.
    # Even there, lines outside the matched span must keep their endings,
    # and the replacement adopts the endings of the span it replaces.
    p = tmp_workdir / "mix2.txt"
    p.write_bytes(b"a\r\nb\r\nKEEP_LF\nc\r\n")
    edit_file(str(p), old="a\nb", new="A\nB")
    assert p.read_bytes() == b"A\r\nB\r\nKEEP_LF\nc\r\n", p.read_bytes()


def test_edit_file_preserves_unix_mode(tmp_workdir):
    if sys.platform == "win32":
        pytest.skip("unix file modes")
    p = tmp_workdir / "deploy.sh"
    p.write_text("#!/bin/sh\necho hi\n")
    p.chmod(0o755)
    edit_file(str(p), old="echo hi", new="echo bye")
    assert (p.stat().st_mode & 0o777) == 0o755, oct(p.stat().st_mode)


def test_edit_file_edits_through_symlink(tmp_workdir):
    target = tmp_workdir / "shared.txt"
    target.write_text("v = 1\n")
    link = tmp_workdir / "link.txt"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlinks unavailable (needs privilege on Windows)")
    edit_file(str(link), old="v = 1", new="v = 2")
    assert target.read_text() == "v = 2\n"
    assert link.is_symlink(), "edit replaced the symlink with a regular file"
