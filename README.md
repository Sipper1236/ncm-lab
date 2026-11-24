# NCM Lab

A small, independent PyTorch implementation of Next-Concept Modeling (NCM),
trained here as a tool-using Python coding agent. The model chooses one action
at a time: write `main.py`, run it on supplied stdin, inspect the result, then
finish or revise the script. NCM itself generates each action coarse to fine;
the surrounding loop runs the tools and returns observations to the next step.

This is a small research prototype, not a reproduction of Stanford's reported
model sizes, data, or results. The current byte-level model has a short context
and output window, so it is intended for compact scripts and tasks.

## Setup

Python 3.12 and `uv` are used here. The AMD RX 580 uses RADV Vulkan; the
experimental [torchvulkan backend](https://pypi.org/project/torchvulkan/0.2.2/)
is an optional dependency. Its source and operator/fallback notes are in the
[project repository](https://github.com/zainsharief/torchvulkan).

```sh
cd ~/Projects/ncm-lab
uv sync --python 3.12 --extra vulkan
```

`--device auto` tries CUDA, MPS, Vulkan, then CPU. To require the RX 580 and
fail on unsupported Vulkan operations instead of silently falling back, set
`TORCHVULKAN_STRICT=1` and pass `--device vulkan`.

## Agent training data

Create synthetic, multi-step Python coding traces:

```sh
uv run python scripts/prepare_agent_pairs.py \
  --episodes-per-task 256 \
  --prefix-bytes 510 \
  --seq-len 256
```

The corpus covers nine short script tasks: count non-empty lines, sum or sort
numbers, count words, reverse lines, compute a mean, sum a CSV column, find a
maximum in JSON, and deduplicate numbers. Three quarters of the episodes ask
for a new script. The rest provide a faulty existing script; the agent runs it,
uses its error or wrong output to choose a fix, and runs the corrected script.
The corpus builder executes examples in the same tool implementation used at
inference. All episodes are synthetic and generated locally; no downloaded
agent-trace dataset is used.

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

## Train both NCM stages

Train the multiscale residual VQ tokenizer:

```sh
TORCHVULKAN_STRICT=1 uv run --extra vulkan python -m ncm_lab.train \
  --stage tokenizer \
  --data data/agent_pairs.jsonl \
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
  --data data/agent_pairs.jsonl \
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
  --corruption 0.5
```

To continue training an existing generator with the same configuration, add
`--resume-generator artifacts/agent/generator.pt`.
`--corruption 0.5` applies a linearly decreasing corruption rate across levels
to help generation tolerate mistakes in earlier codes.

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

## Model overview

- Text is encoded as UTF-8 bytes with BOS, EOS, and padding IDs.
- The tokenizer quantizes a sequence into residual code levels from coarse to
  fine, then reconstructs text from the cumulative latent representation.
- The generator predicts all codes within one level in parallel and moves
  through levels sequentially.
- Each agent turn generates one JSON action. The runtime executes only the
  named local tool, appends its observation, and asks the model for its next
  action.

The earlier instruction-to-code experiment used
[`flytech/python-codes-25k`](https://huggingface.co/datasets/flytech/python-codes-25k),
whose dataset card lists an MIT license. That corpus remains available through
`scripts/prepare_code_pairs.py`, but it is separate from the synthetic agent
traces used above. Its attribution is recorded in `data/SOURCES.md`.

## References

- [Stanford NCM page](https://scalingintelligence.stanford.edu/pubs/ncm/)
- [NCM paper](https://arxiv.org/html/2610.14162v1)
- [Official reference implementation](https://github.com/HazyResearch/ncm)
- [Local implementation notes](../../.agents/docs/next-concept-modeling-build-guide.md)
