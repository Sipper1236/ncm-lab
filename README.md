# NCM Lab

A small, independent PyTorch implementation of Next-Concept Modeling (NCM),
trained here as a tool-using Python coding agent. The model chooses one action
at a time: write `main.py`, run it on supplied stdin, inspect the result, then
finish or revise the script. NCM itself generates each action coarse to fine;
the surrounding loop runs the tools and returns observations to the next step.

This is a small research prototype, not a reproduction of Stanford's reported
model sizes, data, or results. The current byte-level model has a short context
and output window, so it is intended for compact scripts and tasks.

For an evidence-based plan to improve this prototype, see
[NEXT_STEPS.md](NEXT_STEPS.md). It separates demonstrated behavior from the
current gaps and lists the next evaluation and training milestones.

## Setup

Python 3.12 and `uv` are used here. The AMD RX 580 uses RADV Vulkan; the
experimental [torchvulkan backend](https://pypi.org/project/torchvulkan/0.2.2/)
is an optional dependency. Its source and operator/fallback notes are in the
[project repository](https://github.com/zainsharief/torchvulkan).

```sh
cd ~/Projects/ncm-lab
uv sync --python 3.12 --extra vulkan
```

Run `uv` and `ncm-agent` as your normal user; do not use `sudo`. The agent
prints the selected Vulkan device at startup when `--device vulkan` is used.
On the RX 580 setup here, Intel HD Graphics 4600 is skipped and Vulkan device
0 is the AMD Radeon RX 580 2048SP.

`--device auto` tries CUDA, MPS, Vulkan, then CPU. To require the RX 580 and
fail on unsupported Vulkan operations instead of silently falling back, set
`TORCHVULKAN_STRICT=1` and pass `--device vulkan`.

## Agent training data

Create synthetic, multi-step Python coding traces:

```sh
uv run python scripts/prepare_agent_pairs.py \
  --output data/agent_pairs_varied_lines_v2.jsonl \
  --episodes-per-task 256 \
  --cases-per-task 64 \
  --prefix-bytes 510 \
  --seq-len 256 \
  --exclude-manifest data/agent_eval_manifest.jsonl
```

The corpus covers nine short script tasks: count non-empty lines, sum or sort
numbers, count words, reverse lines, compute a mean, sum a CSV column, find a
maximum in JSON, and deduplicate numbers. Each family has eight prompt
templates and 64 sample inputs. Three quarters of the 2,304 episodes ask for a
new script. The rest provide a faulty existing script; the agent runs it, uses
its error or wrong output to choose a fix, and runs the corrected script. Faults
include runtime errors, syntax errors, and wrong-but-running output. Wrong
output traces include a hidden pass/fail observation like the one produced by
the optional output checker. Count-lines inputs include one to six nonblank
lines, with blank lines retained so the wrong-output repair remains observable.
The short canonical solution is
`import sys;print(sum(1 for x in sys.stdin if x.strip()))`. Each next-action
target also has a compact invalid-action recovery example, giving 14,976
prompt/action rows. The corpus builder executes examples in the same tool
implementation used at inference. All episodes are synthetic and generated
locally; no downloaded agent-trace dataset is used.

`data/agent_pairs.jsonl` is the original corpus used to train the checked-in
seed-43 generator. The generator now writes the varied-line corpus to
`data/agent_pairs_varied_lines_v2.jsonl`; that later candidate remains an
experiment. The checkpoint is independent of Stanford's checkpoints.

Each JSONL record trains one next action. The trainer keeps all episodes using
the same sample stdin in one split, so validation examples use inputs absent
from training. Actions use a small JSON protocol and only these tools:

- `write_script(code)` writes one fixed file, `main.py`.
- `run_script()` runs it on the task's `SAMPLE_STDIN` and returns exit status,
  stdout, and stderr.
- A final action ends the loop.

The runner needs Bubblewrap (`bwrap`) and Linux user namespaces. It runs Python
with network disabled, system files read-only, only a temporary script
directory writable, a short timeout, and memory/file limits. The script
interface has no shell, arbitrary path, or package installation tool.

## Evaluate end-to-end behavior

Run the fixed rollout evaluator, including in-training prompts, paraphrases,
the `counts words` regression, and one repair episode per task family:

```sh
TORCHVULKAN_STRICT=1 uv run --extra vulkan python scripts/evaluate_agent.py \
  --device vulkan \
  --manifest data/agent_eval_manifest.jsonl \
  --output artifacts/eval/rollouts.jsonl
```

It generates test inputs absent from the training corpus, checks stdout against
independent task oracles, and records action validity, parser recoveries,
repairs, checkpoint hashes, and full traces. It keeps the oracle output out of
the prompt but gives the agent a pass/fail observation when a script exits
successfully with the wrong output. The repair suite at
`data/agent_eval_repairs_v2.jsonl` has 184 episodes and tests runtime, syntax,
and wrong-output faults. Its 203-episode extension at
`data/agent_eval_repairs_203.jsonl` adds new held-out input values and prompt
variants. The 84-episode manifest at
`data/agent_eval_manifest_extended.jsonl` remains a fixed checkpoint
comparison. Increase `--cases-per-family` for more coverage by choosing a new
`--manifest` path. The rollout score, not teacher-forced loss by itself, is the
main quality signal. The current checkpoint scores 72/203 on the original
larger extension; its full traces are in
`artifacts/eval/repair-v3-203-rollouts.jsonl`.

The multi-input suite adds two fixed verification inputs to each of the 203
episodes. Generate or refresh it with:

```sh
uv run python scripts/prepare_hidden_cases.py \
  --manifest data/agent_eval_repairs_203.jsonl \
  --data data/agent_pairs_varied_lines_v2.jsonl \
  --output data/agent_eval_repairs_203_multitest.jsonl \
  --hidden-cases 2 \
  --seed 20261126
```

Evaluate the checked-in seed-43 checkpoint on that suite with:

```sh
TORCHVULKAN_STRICT=1 uv run --extra vulkan python scripts/evaluate_agent.py \
  --device vulkan \
  --data data/agent_pairs_varied_lines_v2.jsonl \
  --manifest data/agent_eval_repairs_203_multitest.jsonl \
  --output artifacts/eval/repair-v3-multitest-seed43-rollouts.jsonl
```

The visible sample stays in the prompt. Hidden inputs and expected outputs do
not; after the visible sample passes, the runner checks immutable copies of the
same script on each hidden input and returns only an aggregate pass/fail result.
Episodes without a successful visible run are reported as unchecked.

Measure the tokenizer separately before spending more generator training:

```sh
TORCHVULKAN_STRICT=1 uv run --extra vulkan python scripts/diagnose_tokenizer.py \
  --device vulkan \
  --max-pairs 1520 \
  --output artifacts/eval/tokenizer-diagnostics.json
```

This reports exact byte reconstruction after every residual level, codebook
usage and entropy, and full-depth mismatch examples across the full held-out
split. A smaller `--max-pairs` value gives a faster diagnostic sample.

## Train both NCM stages

Train the multiscale residual VQ tokenizer:

```sh
TORCHVULKAN_STRICT=1 uv run --extra vulkan python -m ncm_lab.train \
  --stage tokenizer \
  --data data/agent_pairs_varied_lines_v2.jsonl \
  --output artifacts/agent \
  --device vulkan \
  --steps 3000 \
  --batch-size 4 \
  --eval-pairs 64 \
  --prefix-len 512 \
  --seq-len 256 \
  --latent-dim 64 \
  --tokenizer-layers 2
```

Freeze that tokenizer and train the prompt-conditioned generator:

```sh
TORCHVULKAN_STRICT=1 uv run --extra vulkan python -m ncm_lab.train \
  --stage generator \
  --data data/agent_pairs_varied_lines_v2.jsonl \
  --output artifacts/agent \
  --tokenizer artifacts/agent/tokenizer.pt \
  --device vulkan \
  --steps 1500 \
  --batch-size 4 \
  --eval-pairs 64 \
  --prefix-len 512 \
  --seq-len 256 \
  --generator-width 128 \
  --generator-layers 2 \
  --corruption 0.3 \
  --seed 43 \
  --split-seed 42
```

To continue training an existing generator with the same configuration, add
`--resume-generator artifacts/agent/generator.pt`.
The corruption value applies a linearly decreasing replacement rate across
levels, helping generation tolerate mistakes in earlier codes.
The checked-in generator received a 1,500-step repair-aware continuation with
seed 43 and split seed 42. On the original single-input 203-case suite it
scores 72/203, compared with 64/203 for seed 42; on the shared 184 repair
episodes it scores 64/184 vs 59/184. Seed 43 is one task behind on the fixed
84-case set (34/84 vs 35/84), but has higher raw validity and fewer rejected
false-final attempts. Both have zero accepted false finals because the
checker blocks them.

On the new multi-input 203-case suite, seed 43 scores 73/203, with 77 exact
visible outputs and 58.6% raw valid actions. Each of those 77 visible successes
also passed both hidden inputs; 126 episodes were unchecked. A matched seed-44
run scored 83/203, 36/84, and 11/28, versus seed 43's 73/203, 34/84, and
10/28. Seed 44's raw action validity was much lower (48.0% vs 58.6% on the
multi-input suite), with 74 vs 26 rejected false-final attempts. Seed 43 stays
the default for cleaner action generation and fewer false-final attempts.

A seed-43 candidate trained on varied line-count targets and a shorter
canonical script scored 76/203 on the multi-input suite, but fell to 28/84 on
the fixed comparison versus 34/84 for the default. It was not promoted. Keep
`--split-seed` fixed when comparing model seeds; it controls the train/validation
group split independently of `--seed`.

## Run the coding agent

```sh
TORCHVULKAN_STRICT=1 uv run --extra vulkan ncm-agent \
  --device vulkan \
  --task "Write a Python script that prints the sum of integers from stdin." \
  --stdin "3 5 7"
```

The default checkpoints are `artifacts/agent/tokenizer.pt` and
`artifacts/agent/generator.pt`. Use `--max-steps` to change the number of
write/run/revise decisions allowed for one task.
The optional `--stdin` value is shown to the model as `SAMPLE_STDIN` and is
limited to 100 UTF-8 bytes. Pass `--initial-script` to give it an existing
script to run and debug; the source is limited to 160 UTF-8 bytes.
If you know the expected stdout for the sample input, pass
`--expected-stdout $'15\n'`. The runtime checks it without showing the value to
the model and will not accept a final response until the current script passes.
Without that option, the runtime can only check that the script exits
successfully.

## Model overview

- Text is encoded as UTF-8 bytes with BOS, EOS, and padding IDs.
- The tokenizer quantizes a sequence into residual code levels from coarse to
  fine, then reconstructs text from the cumulative latent representation.
- The generator predicts all codes within one level in parallel and moves
  through levels sequentially.
- Each agent turn generates one JSON action. The runtime executes only the
  named local tool, appends its observation, and asks the model for its next
  action. It retries invalid actions with runtime feedback, accepts obvious
  near-miss `run_script` and `final` actions, and requires a successful script
  run before accepting a final response. When an expected output is supplied,
  it also requires an exact output match and returns only a pass/fail mismatch
  signal to the model. Repeated identical actions are blocked after the first
  retry; a third consecutive identical output stops the turn. The evaluator
  reports repeated outputs separately from invalid actions and parser recovery.

The earlier instruction-to-code experiment used
[`flytech/python-codes-25k`](https://huggingface.co/datasets/flytech/python-codes-25k),
whose dataset card lists an MIT license. That corpus remains available through
`scripts/prepare_code_pairs.py`, but it is separate from the synthetic agent
traces used above. Its attribution is recorded in `data/SOURCES.md`.

## References

- [Stanford NCM page](https://scalingintelligence.stanford.edu/pubs/ncm/)
- [NCM paper](https://arxiv.org/html/2610.14162v1)
- [Official reference implementation](https://github.com/HazyResearch/ncm)
