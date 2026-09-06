# Experiment log

Method: one hypothesis at a time, smallest safe change, measured against the
same workload, kept only with evidence, reverted and recorded otherwise.

Rules (project norms):
- Never invent benchmark results. Never claim an improvement without a measurement.
- Change one major variable at a time.
- Do not remove or weaken tests to make them pass.
- Benchmark runs are the expensive resource here (hours of wall clock per run);
  scoring semantics are checked BEFORE spending a run, because a plausible
  optimization can worsen official scores (see E3).

Primary metrics for this harness (an API-calling agent loop; there is no local
model runtime, GPU, or database to tune):
- clean-termination rate at 1.0x timeout (submission scoring forces errored
  trials to reward zero, so this dominates everything)
- submission-equivalent score (errors zeroed) vs raw verifier score
- tool calls per iteration (round trips are the wall clock)
- seconds per iteration
- prompt-cache hit rate (Anthropic path)

Standard workload: the 10-task TB 2.1 smoke slice (scripts/tb21_smoke.ps1),
1.0x timeout, Opus 4.8. Curated (prior failures + high-delta tasks + controls),
so slice scores are NOT projectable to the full 89; deltas between identical
slice runs are the signal.

---

## E1 — Correctness fix set (run-killer bugs)
- Hypothesis: three classes of observed run-death (rich markup crash, empty-turn
  cascade, null tool-call id) are harness bugs; fixing them recovers trials.
- Changed: nano/cli.py, nano/agent.py, nano/providers.py (commits 3f468ac,
  3a06929, 3cd10d3, df9d1b8), all test-first.
- Baseline: smoke run 0729-1043 - polyglot-rust-c dead at iteration 1 (empty
  turn accepted as done); 3 prior full-run deaths from markup; db-wal-recovery
  dead at iteration 13 (null id).
- Result: smoke run 0730-1755 - polyglot-rust-c worked 125 iterations and its
  workspace passes; write-compressor (prior empty-cascade victim) converted to
  a clean PASS. No regressions in 97 tests.
- Decision: KEPT.

## E2 — Throughput thresholds
- Hypothesis: measured clock sinks (60s bash kills, per-step truncation
  re-fire, 600s SDK request timeout) waste budget without buying anything.
- Changed: bash default timeout 60->300s, truncation low-water mark 0.6,
  client request timeout 120s (commit 2b888cf).
- Baseline: 61/64 bash kills at exactly 60s across the full 2.0 run;
  truncation mutated history every step once over budget.
- Result: measured jointly with E1/E3 in run 0730-1755 (see E4 note on
  attribution); no test regressions.
- Decision: KEPT (thresholds only; no behavior change).

## E3 — Iteration cap 250 (REVERTED) -> 130
- Hypothesis (WRONG): "the wall clock kills first either way, so raising the
  iteration cap 100->250 is free."
- Correction (external review, handoff 010): official leaderboard scoring
  forces errored trials to reward zero even when the verifier would pass the
  workspace, while a clean max_iterations exit still gets graded (35 such
  passes in the 2.0 run). A cap that lets the run live until Harbor's kill
  converts countable results into zeros. Same argument reverted
  max_pushbacks 8->3.
- Changed: cap retuned to 130, derived from measured iteration timing
  (fastest ~4.7s/iter fits the smallest 900s budget; heaviest ~24-31s/iter
  fits 3600s+ budgets). Commit e68b488.
- Result: run 0730-1755 - caffe-cifar-10 exited cleanly at iter=130, 3510s of
  a 3600s budget, graded, PASSED. First clean conversion of that task ever.
- Decision: KEPT at 130. Lesson recorded: check scoring semantics before
  reasoning about "free" changes.

## E4 — Fix-set measurement (runs 0729-1043 vs 0730-1755, identical slice)
- Raw verifier: 5/10 -> 6/10. Submission-equivalent: 1/10 -> 2/10.
  Errored trials: 7 -> 6. Clean terminations: 3 -> 4.
- Attribution caveat: E1+E2+E3 landed together, so per-experiment credit is
  not separable in this pair of runs. Directionally consistent with all three.
- Variance note: mteb-retrieve regressed from a 254s clean pass to a 128-iter
  timeout with no relevant code change - single stochastic trials swing.
- Verdict vs go/no-go (0-2 errors go, 3 inspect, >3 no-go): 6 errors, NO-GO
  for the full 89. Dominant remaining failure: 900s-budget tasks need more
  iterations than the budget holds at ~1 tool call per round trip.

## E6 — Best-possible ensemble (bundled by owner decision; attribution sacrificed)
- Scope: everything below lands before ONE measured slice run. Per-change
  attribution is knowingly given up; the slice measures the ensemble.
- Landed:
  - B1: tool calls now execute regardless of stop_reason (gateways return
    finish_reason "stop" WITH tool_calls; the old order silently dropped
    them and the follow-up request 400s non-retryably). Latent run-killer.
  - B3: empty-turn guard covers unmapped finish reasons.
  - B4: empty-choices gateway responses classified transient and retried.
  - B6: user text now serializes AFTER role:"tool" replies (OpenAI requires
    adjacency; prerequisite for the wrap-up nudge).
  - P5: stdin-reading commands (bare cat, git commit editor, REPLs) get
    instant EOF via a brace-group </dev/null instead of burning the full
    300s bash timeout and losing shell state.
  - P1/P3: max_runtime_sec - self-imposed wall clock. Past 80%: one wrap-up
    nudge + verify pushbacks stop. Past 100%: clean stop_reason="max_runtime"
    exit. Executed bash timeouts clamped to remaining-45s. Adapter passes
    each task's [agent] timeout_sec minus 60s, GATED behind
    NANO_USE_DEADLINE=1 (host-side task.toml read; legality is a judgment
    call - the flag exists so submission runs can omit it).
  - P4: consecutive-duplicate call signal appended to the repeated result
    (measured 10.9% repeats, worst streak 251; consecutive-only so
    pytest-after-edit is silent).
  - P6: prompt - verify against stated criteria instead of authoring tests
    (hidden verifier makes test-writing pure cost), keep the workspace
    gradable at all times, no narration, redirect verbose output to files.
  - P9/P10: cached_tokens telemetry on the gateway path; elapsed seconds in
    every stats line.
  - P8: NANO_MAX_TOKENS env knob (default 8192); runner sets 16384.
- Gateway probe findings (scripts run 2026-08-02, single cheap requests):
  - image_url: ACCEPTED but the image is silently DROPPED (model reports no
    image attached). Vision (P7) is therefore REJECTED for any gateway run -
    it would silently no-op. Revisit only on a direct-Anthropic path.
  - max_completion_tokens: accepted through 64000. 16384 chosen.
  - cached_tokens: absent, and prompt_tokens=5 for a ~1.4k-token prompt -
    the gateway's token reporting is confirmed fiction; caching status
    unknowable from here.
- RESULT (run nano-tb21-smoke-0802-1127, 1.0x, Opus 4.8):
  - Errored trials 6 -> 2. Submission-equivalent 2/10 -> 5/10.
    Raw verifier 6/10 (flat). Clean terminations 4 -> 8.
  - Meets the pre-registered go/no-go threshold (0-2 errors = GO) for the
    full 89-task run.
  - Throughput moved hard: torch-tensor-parallelism 53 iters/900s -> 21
    iters/227s; write-compressor cap-bound -> 66 iters/391s; overfull-hbox
    and qemu-startup converted from errors to clean end_turn exits;
    polyglot-c-py passed while exiting cleanly at the cap.
  - Duplicate-call signal fired on 2 trials (mechanism live in the wild).
  - Remaining 2 errors are the genuinely tight ones: caffe-cifar-10 used
    its full 3600s, regex-log 899s of a 900s budget.
- IMPORTANT SCOPE CORRECTION: the run was launched WITHOUT -UseDeadline, so
  P1/P3 (max_runtime exit, wrap-up nudge, timeout clamp) never executed -
  confirmed by zero "Time is nearly up" occurrences and no --max-runtime in
  the invocation. Every gain above therefore comes from the bug fixes, the
  stdin guard, the batching prompt, prompt tuning, and the 16k ceiling.
  None of it carries the host-side-task.toml legality caveat. The accidental
  omission bought back the per-feature attribution E6 had given up.
- Decision: KEPT.

## E7 — Deadline feature in isolation
- Hypothesis: -UseDeadline converts the 2 remaining errored trials into
  clean graded exits by stopping before Harbor's kill.
- Change: same code, run with -UseDeadline. Single variable vs E6's run.
  (One latent-bug fix also landed first: a null finish_reason from the
  gateway now maps to stop_reason "unknown" instead of dying in pydantic
  validation. Inert on healthy turns; no trial in either run hit it.)
- Caveat to carry into any writeup: this reads the task's timeout host-side,
  which Harbor does not expose to agents by design. A number produced with
  the flag is a research number and must be labeled as such; the E6 number
  (5/10 submission-equivalent) is the one with no caveat.
- RESULT (run nano-tb21-smoke-0808-1346, 1.0x, Opus 4.8): CONFIRMED.
  - Errored trials 2 -> 0. Raw verifier 6/10 -> 7/10.
    Submission-equivalent 5/10 -> 7/10. Clean terminations 8 -> 10/10.
  - Both E6 errors converted exactly as hypothesized: caffe-cifar-10 went
    from AgentTimeoutError at 3754s to a clean end_turn PASS at 2911s
    (first full pass of that task ever); regex-log went from a
    forced-zero AgentTimeoutError (verifier would have passed) to a clean
    PASS at 770s.
  - make-doom-for-mips exercised the new exit path directly: clean
    stop_reason=max_runtime at iter 106 instead of an external kill
    (reward 0 either way, but graded, not errored).
  - Single-trial variance both directions: polyglot-rust-c 0 -> 1,
    write-compressor 1 -> 0 (max_iterations at 130). qemu-startup still 0.
  - Duplicate-call signal fired on 7/10 trials.
- Decision: KEPT as an opt-in flag. Full-89 runs intended for a
  publishable/leaderboard number stay -UseDeadline OFF (E6 config);
  deadline-on full runs are research numbers, labeled as such.

## E5 — Batching instruction (bundled into E6's measurement, not yet run)
- Hypothesis: the measured 1.01 tool calls per iteration means every round
  trip buys one shell command; instructing the model to issue multiple
  independent tool calls per turn and chain independent shell work raises
  calls/iteration and lowers wall clock enough to convert 900s-budget
  timeouts into clean terminations.
- Change: +2 lines in nano/prompts.py. Nothing else in the same experiment.
- Measure: identical smoke slice; compare calls/iteration, s/iteration,
  errored count, both scores.
- Separately queued (NOT in this experiment): duplicate-call signal (+5 lines
  in agent.py) - 10.9% of calls in the full run were exact repeats, 251
  consecutive.

## E8 — Adversarial-review fix set (bundled; no slice gate)
- Source: 5-dimension multi-agent review of the whole harness, each finding
  then put through an adversarial verifier whose default stance was "this
  claim is wrong". 16 raw findings -> 10 after dedup -> 10 survived
  refutation. Three were reproduced live against the running code.
- Landed (8 of the 10; all test-first, 108 -> 120 tests):
  - NANO_MAX_TOKENS is now forwarded into the task container. Harbor does not
    propagate host env, and the adapter forwarded only the three API vars, so
    the probe-verified 16384 ceiling set by BOTH runners had never applied to
    a single trial - every measurement to date, E6 and E7 included, actually
    ran at the 8192 default.
  - The max_tokens continuation nudge is capped at 3 consecutive cutoffs and
    resets on any productive turn. Uncapped, a gateway stuck at the output
    ceiling burned the whole task budget and died on the EXTERNAL kill
    (forced zero); a clean max_tokens exit is graded instead.
  - A turn that is empty AND flagged "length" now reaches the empty-turn
    guard. The continuation branch used to swallow it, so the dead-model
    detector never fired - the polyglot-rust-c failure mode, resurrected
    through a different stop_reason.
  - Truncation no longer blanks the freshest tool_result (the output of the
    calls the model just issued and has not been shown). Oversized tool_use
    inputs are reclaimed first; a single result larger than the whole budget
    still gets cut as a last resort.
  - `set -e` can no longer kill the shell. The brace group is now the left
    arm of an OR list, so a model that once ran `set -euo pipefail` does not
    arm every later failing command to terminate the shell before the
    sentinel - which had been surfacing as "Shell process exited
    unexpectedly" with the diagnostics discarded and cwd/env silently reset.
  - Timeout kills carry the output captured before the hang, matching the
    exit-code path. A suite that passed 40 tests then deadlocked used to come
    back with nothing at all.
  - edit_file judges uniqueness on the NORMALIZED view - the only view the
    model ever sees. A byte-exact single match no longer bypasses the
    ambiguity error when a second copy differs only in line endings.
  - Every 5xx is retryable (not a hand-listed set): the SDK maps 504 and
    Cloudflare 520-524 to one InternalServerError whose class name matches no
    substring check, so the likeliest gateway failure was classified
    non-transient and ended the trial on the first attempt. Request timeouts
    now scale with the output ceiling (a non-streaming 16k generation cannot
    fit 120s, and retrying an identical too-slow request fails every time),
    and SDK-internal retries are off so they no longer multiply with
    _call_with_retry's three attempts.
  - Found while wiring the above: cli.py builds its OWN OpenAI client on the
    gateway path - the path the benchmark actually takes - with the old fixed
    timeout and default SDK retries. Provider-side fixes alone would have
    changed nothing in a benchmark run. Both paths now share
    _request_timeout().
- NOT landed, deliberately:
  - Lone-CR files defeat multi-line edit_file (verified). Rare in benchmark
    tasks; the model escapes via sed. Low.
  - No absolute bash-timeout cap when the deadline flag is off (verified).
    Own run data argues against acting: neither E6 error was a hung command,
    because the stdin guard already removed the dominant hang class, and any
    static cap risks killing legitimate long builds on 3600s-budget tasks.
    Recorded as a known risk instead.
- NO SLICE GATE for this set, by decision. The 10-task slice is curated
  (prior failures + high-delta tasks), so chasing a higher slice score fits
  noise on the same ten tasks; single-trial flips with no code change are
  already documented in E4 and E7. The slice's job was clean-termination
  regression detection and it is at 0 errors. The next measurement is the
  full 89, whose failures are unbiased signal.
- Measure: full 89-task TB 2.1, 1.0x, no -UseDeadline, Opus 4.8. Compare
  errored-trial count and both scores against the 2.0 baseline (53/89).
- PRE-RUN FINDING (2026-09-01): the two smoke attempts on 08-26 and 08-27
  both died at iteration 1 on every task with HTTP 403 permission_denied
  from the gateway, zero tokens in or out. Not a harness defect. A real
  OpenAI key had been set in the Windows User environment (for another
  project) after E7; every runner preferred an inherited OPENAI_API_KEY
  over the ASU token, forwarded it into the container, and the ASU gateway
  rejected it. Runners now pair the ASU token with the ASU endpoint
  whenever .env carries both, overriding anything inherited, and print the
  key source at launch. E8 code was separately proven end to end with a
  live `nano run` through the real gateway path (clean end_turn, 4 iters).

## E9 — Opus 5 through the gateway (model swap, same harness)
- Run: nano-tb21-smoke-0901-1530, 1.0x, no deadline, aws/claude5_opus, E8 code,
  first run with the 16384 ceiling actually applied.
- Result: raw 7/10, errored 3, submission-equivalent 6/10 (vs 4.8 E6 same
  config: 6/10, 2, 5/10). Far fewer iterations per task (13-37 vs 43-130)
  at 2-4x the seconds per iteration.
- The three errors: caffe-cifar-10 and make-doom-for-mips were still
  iterating at the kill (capability/time, same as 4.8); qemu-startup PASSED
  the verifier but the agent ran to 900.0s and was force-zeroed - the
  deadline flag would have saved it, and it is off for submission runs.
- NEW FAILURE CLASS, not the harness: write-compressor died on "3
  consecutive empty turns" at iteration 5. Reproduced against the gateway:
  Opus 5 returns finish=stop, no content, no tool calls, 0 thinking tokens,
  in ~1.2s, once there is any tool history. Trigger is LEXICAL - a paraphrase
  of the same task answers. Sweep of all 89 task instructions with one tool
  round, three passes: Opus 5 empty on 6-8 (stable core of 6: polyglot-c-py,
  polyglot-rust-c, crack-7z-hash, distribution-search, vulnerable-secret,
  break-filter-js-from-html), Sonnet 5 empty on 6, Opus 4.8 empty on 0.
  Retry of the identical request: the core 6 stay empty 5/5; two others
  recover (cancel-async-tasks ...E., write-compressor ..EE.).
- Harness change (model-agnostic, test-first): an empty completion is
  re-asked twice at the provider before it reaches the agent's empty-turn
  guard, so stochastic blips no longer pollute history with a placeholder
  and a nudge. The persistent case still surfaces as an honest error.
  CLI stats lines now carry elapsed seconds (the qemu kill-vs-finish race
  was undiagnosable without it).
- Decision: full 89 and any leaderboard passes run on Opus 4.8. Opus 5 via
  this gateway carries a defect that force-zeros a handful of tasks no
  harness can fix; committing to it for 445 trials bakes those in.
