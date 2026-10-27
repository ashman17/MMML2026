# Implementation review — 2026-10-24

## Follow-up after the operational fixes

The four original findings below have been addressed in the current local-client
workflow. An independent rerun of `tests/test_protocol_gaps.py` passed all **10
tests in 27.05 seconds**. All seven PSC shell/batch scripts passed `bash -n`;
`git diff --check` also passed. The coding-agent handoff reports 235 tests passing
across both suites; this follow-up reran the ten focused regression tests rather
than repeating the full suites.

**Ready for the development pilot using `serve.sbatch` plus the local runner.**
Actual PSC compatibility, token usage, extraction quality and latency still need
real inference. Start with the one-request synthetic10 check before dev/timing.

One residual issue is confined to optional `batch_all.sbatch` recovery:
`idea4_recover_saved_run` in `psc/serve_common.sh` treats the presence of
`run_config.json` inside `.partial` as proof that copying finished. If the first
publish is interrupted after copying that file but before copying records and
status, the next recovery promotes the incomplete directory. A temporary-directory
reproduction returned success and promoted such a directory with neither records
nor status present. Before relying on this optional route, add a completion marker
written only after a successful copy and require it during partial recovery;
test interruption during the first copy. This does not affect the planned
PSC-server/local-client pilot.

The original review below is retained as historical context; its four findings
should no longer be read as outstanding blockers for the local-client workflow.

---

The implementation covers the planned midterm experiment. The remaining work is
operational validation and the corrections below, not another experimental
condition or a larger implementation. Fix the two high-priority findings before
the full evaluation. The next PSC milestone is a development pilot.

This review did not modify experiment code, start a PSC job, or call a real model.
No real world-state run directory was present at `runs/idea4_world_state`.

## What is implemented

- Committed, reproducible selection: Raw on 1,000 items; graph/evidence on 200
  two-image items; a nested 100-item fallback; 22 development items; 6 timing items.
- Raw, flat evidence, and graph conditions. Flat and graph share one extraction
  and use the same extracted facts. The answer stages retain the original images.
- A PSC server workflow, SSH tunnel instructions, model identity manifests,
  local inference, request logging, caching, and result exports.
- Analysis of paired outcomes, uncertainty, extraction coverage and cost, plus
  a 20-item audit workflow for inspecting when evidence helps or hurts.
- Tests for preparation, extraction validation, serialization, backend behavior,
  resume, freezing, analysis, and a mock end-to-end run through audit generation.

These are sufficient components for a reportable baseline comparison. Whether
Qwen produces useful inventories and whether graphs improve answers remain
empirical questions; passing software tests cannot establish either result.

## Findings

### 1. [P1] A new PSC allocation prevents resuming an existing run

Location: `runner.py:513`, `runner.py:526–533`;
`psc/write_manifest.py:145–147`.

The run configuration embeds the complete server manifest. Resume compares the
entire configuration for equality. Every new server launch changes manifest
fields such as creation time, host and Slurm job ID, even when model revision,
precision, and inference settings are unchanged. After a wall-time limit or
server restart, fetching the new manifest and resuming therefore fails with
`run_config.json does not match this invocation; use a new output directory`.

Reproduced locally by retaining every protocol field and changing only those
three server-session fields. The mismatch check rejected the configuration.
Starting another output directory can reuse cached calls, but it is not the
documented resume behavior and fragments the run's attempt history.

**Correction:** distinguish the stable protocol/model fingerprint from individual
server sessions. Validate the stable fingerprint on resume, and retain each
session's manifest as provenance. Continue rejecting changed model revisions or
inference settings. Add a regression test that resumes a paused run using a fresh
manifest for the same protocol.

The optional `psc/batch_all.sbatch` also creates a fresh staging directory for each
job and does not restore the saved run before adding `--resume`. Address that
separately before using the all-on-PSC route. The planned local-client workflow
does not need this optional script.

### 2. [P1] Freeze can certify prompts that were never tested

Location: `runner.py:1064–1080`, `runner.py:1415–1417`.

The freeze compatibility check compares model identity and numeric settings,
but not input runs' prompt hashes. The resulting freeze records the *current*
prompt, schema and serializer hashes. Consequently, editing the extractor after
development, or mixing development and timing runs from different prompt
versions, can pass freeze and authorize evaluation with untested behavior.

Reproduced through the actual freeze command using completed mock development,
timing and synthetic runs. Replacing the current extractor hash in memory with
64 zeroes produced exit code 0 and a freeze containing the new hash; the dev run
still recorded `dbe0097ec3d9d9d487c37c284eeeb1b3319fa45d6af4a4a298b08140c192578e`.
No source file or real result was altered. A separate check confirmed differing
prompt hashes across input runs are accepted by `_require_same_protocol`.

**Correction:** record the relevant prompt, inventory schema and serializer
fingerprints with every run. Require input runs and current artifacts to match
before freezing. A changed artifact should require the affected development
checks to be rerun. Test both an edit after development and mixed-version runs.

### 3. [P2] Job-count projection does not reserve startup inside each job

Location: `runner.py:1369–1371`.

The job count is calculated from padded inference time divided by wall time,
then startup is added afterward. Startup consumes the same allocation's wall
time, so the calculation can undercount jobs and service units near a boundary.

Reproduction: a padded workload of 3.9 hours, a 4-hour wall limit and the current
0.5-hour startup allowance reports one job and 4.4 total hours. One such job can
only provide 3.5 hours of inference. It needs two jobs, with a total modeled cost
of 4.9 hours under the same startup assumptions.

**Correction:** use usable inference time per job (`walltime - startup`) when
calculating the number of jobs; reject a wall time no larger than the startup
allowance. Update the corresponding formula in the implementation plan. Queue
delay remains an additional unknown, not measured GPU time.

### 4. [P2] The real ten-image check accepts missing token usage

Location: `runner.py:522`, `runner.py:1188–1194`.

`synthetic` is true for both mock inference and real inference on the generated
ten-image input. The token-limit check uses that flag to exempt missing token
usage. A real Qwen response without usage can therefore pass despite the
documented requirement to verify a prompt-token count of at most 30,000.

Reproduced with an otherwise successful synthetic10 record and null input-token
usage: the criterion returned `passed: true`.

**Correction:** distinguish the mock backend from generated test data. Only mock
backend results should receive an explicit testing exemption; real synthetic10
inference must provide usable token counts to pass this check.

## Validation

- The first world-state test invocation was blocked by sandbox restrictions on
  binding the mock HTTP server to localhost: 174 passed, 2 failed, 18 setup
  errors. These were environment restrictions, not established code defects.
- With localhost permission, the world-state suite passed: **194 passed in
  177.16 seconds**, including the mock end-to-end analysis/audit test.
- Existing `idea4_visual_reasoning/tests`: **31 passed**.
- All seven PSC shell/batch scripts passed `bash -n`.
- `git diff --check` passed for the tracked changes.
- Four targeted reproductions confirmed the findings above. Existing tests do
  not cover these scenarios.

## Recommended execution order

1. Correct the four findings and run their focused regression checks. Keep the
   planned local runner plus PSC server workflow for the first experiment.
2. Start one PSC server and tunnel, fetch its manifest, and verify the exact
   Qwen checkpoint/revision. Use the existing PSC README for account-specific
   allocation and launch commands. GPU compatibility and memory use still need
   to be checked on the actual node.
3. Run the ten-image synthetic check: **one model request** containing ten images.
4. Run the 22-item development set with `raw,flat,graph`: **88 requests** before
   cache reuse (Raw + extraction + flat answer + graph answer per item). Inspect
   real inventories and evidence manually as well as checking schema/parse
   statistics. Empty or inaccurate but schema-valid graphs are not proof that
   extraction is adequate.
5. Run the six larger-image timing items: **6 requests**. These three stages
   total about **95 requests**, excluding retries, prompt revisions, or the
   optional concurrency experiment. Measure real elapsed time, token usage,
   truncations and transport failures before estimating a full run.
6. Freeze the tested protocol and select eval200 or the predefined eval100
   fallback using actual GPU, remaining allocation, wall time, deadline and
   measured timings. Do not choose sample size after inspecting evaluation
   accuracy. The README's example deadline must be replaced with the real one.
7. Run Raw on all 1,000 items, then flat/graph on the selected sample. The nominal
   main-run budget is **1,600 requests** for eval200 or **1,300** for eval100:
   `1,000 + 3 × sample_size`, before cache reuse. Keep completed Raw results
   compatible with teammates by sharing the model and inference settings.
8. Produce the paired analysis and 20-item audit. Report gains, regressions,
   extraction failures and runtime, including a null or negative graph result
   if that is what the experiment shows. The 1,000-item Raw score and the paired
   subset comparison answer different questions and should be labeled clearly.

The scientific scope is sufficient. A full evaluation should follow the fixes
and a successful real pilot; another broad brainstorming round is unnecessary.
