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
