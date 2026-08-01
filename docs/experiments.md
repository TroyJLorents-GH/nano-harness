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

## E5 — Batching instruction (NEXT, not yet run)
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
