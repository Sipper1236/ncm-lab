# Roadmap: Make This NCM More Agentic

**Status:** working plan for the public `work/ncm-prototype` branch
**Last updated:** 2026-11-24
**Scope:** continue the small Python coding agent in this repository. This is a
research roadmap, not a claim that the current checkpoint is a general coding
agent or a reproduction of Stanford's reported models.

## 1. The destination

Build an NCM agent that can take a coding task, use a bounded set of tools over
multiple turns, inspect tool results, correct mistakes, and stop only when the
task's tests pass. It should generalize across new input values and new ways of
describing a task. Its NCM-specific value should be measured against a fair
autoregressive baseline, rather than inferred from the architecture or from a
low teacher-forced loss.

For this repository, a useful first destination is:

- The agent solves a documented set of short Python tasks from held-out inputs.
- Rewording a task does not change it into a different task.
- After a script fails, the agent uses the error to fix the script and reruns it.
- Each tool call has a narrow contract and produces an observation the model can
  use on the next turn.
- A final answer is accepted only after independent checks pass.
- The complete NCM rollout is compared with a baseline on task success, runtime,
  and compute.

“Agentic” here means that the model repeatedly chooses actions in response to
observations. It does not mean the current program can browse the repository,
run arbitrary shell commands, install packages, or work indefinitely. Those
capabilities would need separate tools, training traces, and safety boundaries.

## 2. What exists now

These facts come from the current code, data, checkpoint, and observed runs.

### Model and data

- The project has a byte-level multiscale residual VQ tokenizer and a
  prompt-conditioned NCM generator.
- The tokenizer uses a 256-byte target window and ten levels:
  `[1, 2, 4, 8, 16, 32, 64, 128, 192, 256]`.
- The current tokenizer has width 64 and 64 entries per level. The generator
  has width 128, two transformer layers, and four heads.
- Each generator level predicts all of its codes in parallel; levels are
  generated sequentially from coarse to fine.
- `--corruption 0.5` currently means a linearly decreasing per-level
  replacement rate from 1.0 at the coarsest level to 0.0 at the finest. The
  target labels remain clean.
- The current generated corpus contains 14,976 prompt/action rows from 2,304
  synthetic episodes across nine short task families. Each next-action target
  has one compact invalid-action recovery example. A quarter of episodes start
  with a faulty script and demonstrate run, observe, repair, rerun, and finish.
  Repair traces cover runtime errors, syntax errors, and wrong-but-running
  scripts. Wrong-output traces include a hidden pass/fail observation; expected
  output values are not included in the model prompt.
- The tasks cover line and word counts, integer sums and sorting, line reversal,
  means, CSV totals, a JSON maximum, and sorted unique integers.
- The checkpoints and generated corpus are committed under `artifacts/agent/`
  and `data/agent_pairs.jsonl`.

### Agent and tools

- `write_script(code)` writes one fixed file, `main.py`.
- `run_script()` executes that file on the provided stdin in a Bubblewrap
  sandbox. The sandbox has no network, a temporary writable directory, and
  time, memory, and file limits.
- The agent retains action/observation history and updates `CURRENT_SCRIPT`
  after a failed run. Its prompt and output windows are short.
- The runtime retries invalid actions and accepts narrowly defined near-miss
  forms of `run_script` and `final`. It refuses to accept a final response
  unless the current script has passed its available checks.
- By default, the runner can only check exit status. `--expected-stdout` adds an
  exact-output check without adding the expected text to the prompt. The
  evaluator supplies this hidden check and returns only an
  `expected_output_mismatch` observation when output is wrong. This checks one
  input at a time; the agent does not yet run a hidden multi-input test suite.

### Evidence so far

Three focused inference runs succeeded on the current checkpoint:

| Task | Observed result |
|---|---|
| Sum `3 5 7` | Wrote a script, ran it, observed `15`, and finished. The runtime corrected a near-miss `run_script` action. |
| Repair a sum script that fails on spaces | Observed a `ValueError`, rewrote the script, reran it, observed `15`, and finished. The runtime corrected a malformed final action. |
| Count words in `red blue green`, using the training wording | Wrote and ran the script; observed `3`; finished. |

A materially different prompt, `Write a Python script that counts words in
stdin.`, failed on the pre-expansion checkpoint. It is now included in the
expanded corpus, and the follow-up checkpoint solves the fixed user-regression
case.

The rollout evaluator uses a fixed 28-episode manifest with familiar prompts,
paraphrases, the user regression, and one repair episode per family. The
pre-expansion checkpoint scored 7/28, with a 42.1% raw valid-action rate. The
expanded-data checkpoint before fine-tuning also scored 7/28 (46.7% raw
validity). Fine-tuning that checkpoint for 3,000 steps at corruption 0.5 kept
the score at 7/28 but raised raw validity to 64.4%.

Before adding the output checker, the 0.30-corruption seed-42 checkpoint scored
10/28 and 29/84, with one accepted false final on the 84-case manifest. A
controlled seed-43 run scored 10/28 and 27/84 with one and four false finals.
Those results showed seed sensitivity.

The current default is a 1,500-step continuation from the seed-42 checkpoint,
trained on the expanded repair traces with seed 42 and split seed 42. On fixed
manifests with the hidden output checker, it scores 10/28 and 35/84, compared
with 10/28 and 29/84 for the previous checkpoint. On the 184-episode repair
suite it scores 59/184, compared with 55/184. The new checkpoint produces 70
exact outputs on that suite versus 59 for the previous checkpoint. Both have
zero accepted false finals because the checker blocks them. The new checkpoint
attempted 43 rejected false finals versus 186 for the old checkpoint on the
184-episode suite; raw valid-action rates were 49.8% and 43.6%. On the fixed
84-case set, both made eight rejected false-final attempts; raw valid-action
rates were 47.1% and 46.7%. This is a measured improvement, but raw action
validity and task success remain far below the proposed release gates. Its
203-episode extension scores 64/203, with 75 exact outputs, zero accepted false
finals, 44 rejected false-final attempts, and 49.0% raw valid-action rate. Full
traces and training metadata are under `artifacts/eval/repair-v3-*`.

The user's exact count-words prompt previously succeeded through the normal
CLI, which logs the RX 580 and needs no `sudo`; rerun it after any checkpoint
promotion.

The failed traces contain near-miss script text such as `sys.stdinnread()` and
`neet(r)`. On the full 1,520-pair validation split, the existing tokenizer
reconstructs 776 targets exactly (51.1%, 98.4% aligned byte accuracy). A fresh
3,000-step tokenizer with the same 64-entry codebooks scores 712/1,520
(46.8%). Increasing each codebook to 256 entries raises exact reconstruction
to 1,160/1,520 (76.3%, 99.0% aligned byte accuracy). Its first 3,000-step
generator scored 0/28 and produced one false final. After 5,000 more steps it
scores 6/28, with 39.8% raw action validity and one false final. The
larger-codebook generator remains an experiment despite the tokenizer's
reconstruction improvement.

A 0.15-corruption fine-tune of the 64-entry generator scored 9/28 task
successes, with 10 exact outputs, 53.4% raw action validity, and two false
finals. The 0.30 corruption setting is selected for the current default based
on higher exact success and no false-final increase on the expanded manifest.
The second seed was trained with `--split-seed 42`, which holds the split fixed
independently of `--seed`.

The earlier generator checkpoint recorded validation loss 0.0041; the expanded
data fine-tune recorded 0.0525. These used different data splits, so the losses
are not directly comparable. Both are teacher-forced per-level code
cross-entropy, not end-to-end success rates.

## 3. Work in this order

The first priority is to measure agent behavior reliably. Do not spend another
large training run optimizing only the teacher-forced loss before building the
rollout evaluator.

### Milestone 0 — Make local setup and device selection obvious

**Status: done.** The root-owned venv was replaced with one owned by `rayyan`;
normal-user inference logs the selected RX 580.

**Goal:** anyone in the project checkout can run the model as their normal user
and confirm which accelerator it selected.

Current setup command:

```sh
uv sync --python 3.12 --extra vulkan
```

Run inference without `sudo`:

```sh
TORCHVULKAN_STRICT=1 uv run --extra vulkan ncm-agent \
  --device vulkan \
  --task "Write a Python script that prints the number of words in stdin." \
  --stdin "red blue green"
```

`sudo uv` is the wrong fix for a venv permission error. It can create a venv
that points into `/root/.local`, which the regular user cannot access. In this
checkout, the root-owned venv was replaced with one owned by `rayyan`.
If a project venv has become stale, recreate that generated environment as the
normal owner:

```sh
uv venv --clear --python 3.12
uv sync --extra vulkan
```

The Vulkan device diagnostic for this machine found one supported device at
index 0: **AMD Radeon RX 580 2048SP (RADV POLARIS10)**. The warning about
Intel HD Graphics 4600 says that device was skipped; it does not mean the RX
580 was skipped. The agent now prints `vulkan_device=...` at startup when
`--device vulkan` is selected.

**Done when:** a normal-user invocation starts without a Python permission
error, logs the selected device, and produces an agent transcript. If Vulkan
is unavailable, the command should fail clearly; it must not quietly claim to
have used the RX 580.

### Milestone 1 — Add an end-to-end rollout evaluator

**Status: implemented; fixed 28-, 84-, 184-, and 203-episode reports are recorded.**

The baseline manifest is committed at `data/agent_eval_manifest.jsonl`. The
expanded baseline is `data/agent_eval_manifest_extended.jsonl`; the
runtime-, syntax-, and wrong-output repair suite is
`data/agent_eval_repairs_v2.jsonl`, with a 203-episode extension at
`data/agent_eval_repairs_203.jsonl`. Corpus generation supports
`--exclude-manifest` and reserves its cases so later checkpoints use the same
genuinely unseen inputs.

**Goal:** measure whether an episode actually solves the task, including tool
use and repair.

`scripts/evaluate_agent.py` runs episodes through the same `run_agent` path as
the CLI. The evaluator, not the model, holds expected outputs and test cases.
Expected answers are not put in prompts or training context. The runtime gets
an exact-output check and returns only a pass/fail mismatch observation.

For each episode, record at least:

1. Task family, prompt template, input case, and split name.
2. The raw decoded action and whether the runtime had to repair its format.
3. Tool name, arguments, observation, script exit status, and elapsed time.
4. Whether generated code parsed and ran.
5. Whether stdout matched the independent oracle on the supplied case.
6. For stronger checks, whether the script passed additional hidden test inputs.
7. Whether an initial failure was followed by a correct repair and rerun.
8. Whether the agent stopped, hit its decision budget, repeated itself, or made
   a false final claim.

Report these metrics separately:

- Raw valid-action rate before parser recovery.
- Tool-selection accuracy by state: first action, after a run error, and after
  a successful run.
- Script syntax rate and sandbox execution rate.
- Exact task success against the oracle.
- Repair success conditional on a deliberately failing initial script.
- Accepted false-final rate and rejected false-final attempts, loop rate,
  decisions per successful episode, and latency.
- Per-level code accuracy and teacher-forced loss as diagnostic metrics only.

Start with a manageable held-out suite: for example, 10–20 cases per task
family across both familiar and paraphrased prompts. Save the exact manifest,
checkpoint hashes, seed, device, and command with every result. Keep a fixed
test set so training changes can be compared fairly.

**Done evidence:** one command produces machine-readable episode traces and a
summary, distinguishing script execution from exact oracle output. The first
run is small (28 episodes) and is a starting baseline, not a final benchmark.

### Milestone 2 — Measure prompt and task generalization

**Goal:** fix the observed failure where “counts words” was answered with a
sum script.

The current dataset now uses eight prompt templates and 64 sample cases per
family, but it still repeats the same solution pattern. The current group split
keeps an input case together; it does not hold out whole task families, prompt
templates, or solution patterns. A low validation loss on this split therefore
gives weak evidence about new wording.

**Status: expanded traces generated; retraining remains.** The exact `counts
words` regression is now a training template. The data generator also creates
recovery pairs where a malformed action and an `invalid_action` observation
are followed by the intended valid action.

Create explicit evaluation tiers:

| Tier | Hold out | What it measures |
|---|---|---|
| A: new values | Input values only | Basic execution on inputs not seen in training. |
| B: new wording | Prompt templates and word order | Whether the model follows intent rather than memorizing a sentence. |
| C: new implementation | Equivalent solution forms | Whether it can generate different valid code for the same task. |
| D: new composition | Combine two known operations | Whether it can compose learned behavior. |
| E: new task family | Entirely held-out operation | Broader task generalization; expect this to be hardest. |

For training data, add meaning-preserving prompt variants, different input
formats, edge cases, and multiple correct solutions. Keep all near-duplicates
and all cases from one generated episode in a single split. Track semantic
families and solution-template IDs in the data manifest. Avoid putting a
template in train and test and then describing the result as unseen-task
generalization.

Use an independent oracle to compute expected outputs. For each generated
solution, test more than one input so the model cannot pass by hard-coding the
visible sample. Include empty input, one item, negative numbers, whitespace,
blank lines, duplicate values, malformed JSON/CSV where relevant, and boundary
cases appropriate to the task.

**Done when:** evaluation reports Tiers A–C separately and the paraphrase
failure above is part of the fixed regression set.

### Milestone 3 — Improve tool-action reliability

**Status: optional exact-output checking and mismatch feedback are implemented.
The current 184-episode checkpoint reduces rejected false-final attempts from
186 to 43, while raw action validity remains 49.8%.**

**Goal:** make the model choose and serialize valid actions without depending
on runtime typo correction.

The current parser has two jobs: parse the action protocol and recover a small
set of near-miss `run_script` and `final` outputs. Keep counting those
recoveries. They are useful resilience, but if the metric hides them, apparent
agent quality can exceed what the generator actually learned.

Next actions:

- Add training episodes where the agent sees a tool observation and must
  choose the next valid action. Include invalid-action observations, failed
  writes, syntax errors, runtime exceptions, wrong outputs, and successful
  runs.
- Teach a consistent action schema from all prompts. Add examples for each
  action and state; do not make valid output depend on one memorized sentence.
- Consider constrained decoding for the outer JSON envelope while leaving the
  `code` string free-form. The constraint must handle JSON escaping and must not
  silently rewrite script contents.
- Track recent tool state explicitly: script written, script hash, last run
  status, and whether a write happened after that run.
- Detect repeated identical actions and stop or return a clear stuck state
  before wasting the entire decision budget.
- Keep the current rule that a final answer requires a successful run of the
  current script. The optional exact-output check now blocks a wrong result
  when one expected output is available; add multiple hidden tests and preserve
  their results as workspace state.

**Done when:** raw action validity is high, parser interventions are rare and
reported, and the evaluator records zero false finals across a fixed suite.

### Milestone 4 — Expand from one script to a useful coding workspace

**Goal:** support real multi-step coding tasks without giving the model an
unbounded shell.

The current environment only writes `main.py` and runs it on one stdin string.
It cannot inspect an existing project, edit multiple files, or run a test
suite. Add capabilities in small, explicit steps:

1. `list_files()` restricted to the task workspace.
2. `read_file(path)` with path traversal protection and a file-size limit.
3. `write_file(path, content)` limited to an allowlisted workspace and size.
4. `run_tests()` invoking a configured test runner with fixed arguments; do not
   accept an arbitrary shell command from the model.
5. A small `get_diff()` observation so the agent can inspect its own edits.

For every new tool, define a JSON schema, argument validation, limits, and
example episodes for success and failure. Keep network and package installation
disabled by default. If a task needs dependencies, prepare them outside the
agent sandbox or add a separate reviewed workflow; do not add an unrestricted
`shell(command)` tool as a shortcut.

Support a workspace state summary in the prompt so long interactions do not
depend on retaining every raw stdout byte. Preserve the task, relevant files,
latest failure, and test status when old history is dropped. Train on the same
state representation used at inference.

**Done when:** the agent can inspect a small project, make a bounded change,
run its tests, react to a failing test, and finish only after passing tests.

### Milestone 5 — Improve the NCM training signal

**Status: tokenizer diagnostics and codebook-size ablation complete; the
0.30-corruption seed-42 recipe remains selected, now fine-tuned on the varied
repair corpus. Repeatability is still under investigation.**
`scripts/diagnose_tokenizer.py` reports exact-byte reconstruction after every
level, codebook usage and entropy, and mismatch examples. On all 1,520
validation traces, the 256-entry tokenizer improves exact reconstruction from
51.1% to 76.3%, but its 8,000-step matching generator scores only 6/28.

**Goal:** reduce errors that accumulate from coarse to fine and optimize the
actual tool-loop objective.

The tokenizer diagnostics are implemented. Use them to compare:

- Exact byte reconstruction at full depth and after each level.
- Per-level codebook usage and code entropy.
- Decoded output trajectories after every level.
- Character errors around quotes, braces, whitespace, names, and numbers.

Gate generator work on tokenizer quality. A generator cannot reliably emit
valid JSON or Python if the frozen tokenizer loses punctuation or identifiers.

Then compare controlled generator experiments on the same train/validation/test
splits:

| Experiment | Variants | Primary comparison |
|---|---|---|
| Corruption | 0, 0.15, 0.30, 0.50 mean settings | Rollout success and error recovery, not only CE. |
| Corruption schedule | Constant, linearly decreasing, shifted linear | Which levels need the most robustness? |
| Error source | Uniform random codes vs. codes sampled from actual model mistakes | Does training resemble the model's rollout errors? |
| Loss weighting | Current equal-level loss vs. code-position weighting | Do coarse levels improve without hurting final output? |
| Sampling | Greedy vs. a small number of sampled candidates | Success per wall-clock budget. |
| Model size | Current width/depth vs. one larger configuration | Quality gain per added compute. |

Keep clean target codes as labels when corrupting conditioning. The NCM paper
reports that corruption degree matters and says corruption based on the model's
actual errors may better match inference. Implement that only after the fixed
rollout evaluator can measure its effect. Change one variable at a time and
record every seed and checkpoint hash.

Save enough training metadata to reproduce a run: dataset hash, tokenizer and
generator configurations, corruption schedule, seed, optimizer state, step,
and accelerator. The current `--resume-generator` path resumes weights but
restarts the optimizer; document this when comparing continued runs.

**Done when:** the selected recipe improves exact held-out episode success over
the current checkpoint in repeated runs, with no hidden split changes.

### Milestone 6 — Use inference budget for verified candidates

**Goal:** exploit NCM's ability to produce a full candidate in a small number
of level steps without blindly accepting a single greedy result.

Explore a small candidate set only after single-candidate metrics are stable.
For coding tasks, rank candidates by sandboxed tests over several inputs, not
surface similarity alone. Keep test inputs hidden from the model while scoring
candidate scripts. Record both candidate-generation cost and selection cost.

An alternative is to compare candidates with a minimum-Bayes-risk method where
no exact oracle exists, but text similarity is not a substitute for code
execution checks. Bound the number of candidates and total runtime. Report
success at matched wall-clock budgets.

**Done when:** candidate selection improves pass rate at an explicitly stated
inference budget and does not leak evaluator tests into prompts.

### Milestone 7 — Make an honest NCM comparison

**Goal:** establish whether this implementation's NCM structure is useful for
the chosen agent workload.

Build a small autoregressive baseline trained on the same prompt/action pairs.
Match tokenizer or output representation where possible, parameter budget,
training examples, data splits, and hardware. Compare:

- exact task success and repair success;
- raw action validity and parser-recovery rate;
- training steps and tokens seen;
- number of model evaluations per action;
- wall-clock time for a complete successful episode;
- memory use and throughput at the same batch size.

Do not cite the Stanford paper's speed or quality numbers as results for this
prototype. The paper evaluates different datasets and much larger models. The
comparison here should answer the local question: does coarse-to-fine NCM
produce a better or faster tool-using Python agent at this budget?

**Done when:** a report includes the baseline, exact configuration, repeated
seeds, task-success metrics, and full episode timing.

## 4. Proposed release gates

These are suggested targets for the next substantial checkpoint, not current
results. Adjust them only before looking at the final test scores.

| Gate | Proposed target |
|---|---:|
| Correct output on unseen inputs for supported families | At least 90% |
| Correct output on held-out prompt paraphrases | At least 80% |
| Repair success after one injected script failure | At least 80% |
| Raw valid actions before parser recovery | At least 98% |
| Accepted false finals when oracle checks are supplied | 0 in at least 200 episodes |
| Rejected false-final attempts | Report the count and reduce it across repeated suites |
| Sandbox boundary violations | 0 in the reviewed adversarial suite |
| NCM vs. matched autoregressive baseline | Report both success and end-to-end time; do not cherry-pick one metric |

Use a fixed test manifest and a second seed before declaring a gate passed.
If a gate is missed, report the miss and keep improving the relevant stage.

## 5. Immediate next actions

In order:

1. Add multiple hidden inputs per script so one visible sample cannot make a
   faulty or hard-coded solution look correct.
2. Improve raw action validity and reduce parser recovery; the current default
   is still below 50% raw valid actions on the 203-episode suite.
3. Repeat the repair-aware training recipe with another training seed and the
   same split, then compare task success and rejected false-final attempts.
4. Add a repeated-action guard before increasing the decision budget.
5. Retain the 256-entry tokenizer as an experiment until its matching generator
   improves end-to-end success.

The larger codebook is an isolated tokenizer experiment justified by byte-exact
reconstruction measurements. Keep generator size fixed until rollout evidence
shows whether this tokenizer change helps the task loop.

## 6. Working commands

### Run a familiar-template baseline

```sh
TORCHVULKAN_STRICT=1 uv run --extra vulkan ncm-agent \
  --device vulkan \
  --task "Write a Python script that prints the number of words in stdin." \
  --stdin "red blue green"
```

Expected stdout from the generated script is `3`. This wording is a known
training template; use it as an in-distribution check, not as evidence of
paraphrase generalization.
Append `--expected-stdout $'3\n'` if you want the runtime to check that sample
output. The expected string is not shown to the model.

### Run the known paraphrase regression

```sh
TORCHVULKAN_STRICT=1 uv run --extra vulkan ncm-agent \
  --device vulkan \
  --task "Write a Python script that counts words in stdin." \
  --stdin "red blue green"
```

The follow-up checkpoint passed this phrasing on the frozen case. Keep it in
the evaluation set as a regression check; this single success does not establish
generalization.

### Rebuild the current synthetic corpus

```sh
uv run python scripts/prepare_agent_pairs.py \
  --episodes-per-task 256 \
  --cases-per-task 64 \
  --prefix-bytes 510 \
  --seq-len 256 \
  --exclude-manifest data/agent_eval_manifest.jsonl
```

Do not replace the committed evaluation manifest when regenerating training
data. It contains the exact same episode inputs used by baseline and follow-up
checkpoints.

### Run the fixed rollout evaluation

```sh
TORCHVULKAN_STRICT=1 uv run --extra vulkan python scripts/evaluate_agent.py \
  --device vulkan \
  --manifest data/agent_eval_manifest.jsonl \
  --output artifacts/eval/rollouts.jsonl
```

Run the larger repair suite with the same evaluator:

```sh
TORCHVULKAN_STRICT=1 uv run --extra vulkan python scripts/evaluate_agent.py \
  --device vulkan \
  --manifest data/agent_eval_repairs_203.jsonl \
  --output artifacts/eval/repair-v3-203-rollouts.jsonl
```

## 7. Reference material

- [Stanford NCM publication page](https://scalingintelligence.stanford.edu/pubs/ncm/)
- [NCM paper](https://arxiv.org/html/2610.14162v1): Section 3 describes the
  two-stage method; Appendix F.3 discusses conditioning corruption and notes
  that corruption based on model errors is an open direction; Appendix I lists
  limitations and future work.
- [Official NCM implementation](https://github.com/HazyResearch/ncm): consult
  its model, sampling, and evaluation paths as a reference. This project is an
  independent small implementation and should not claim the reference model's
  results.
- [Project README](README.md): current setup and commands.
