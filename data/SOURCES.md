# Dataset sources

## Synthetic coding-agent traces

`scripts/prepare_agent_pairs.py` creates local synthetic episodes for nine
short Python script tasks. It varies prompt wording and input cases, includes
faulty-script repair trajectories with runtime, syntax, and wrong-output faults,
and adds compact invalid-action recovery examples. Wrong-output traces include
a pass/fail mismatch observation without the expected value. The expected
actions and observations are derived by running the scripts through the
project's isolated Python tool. No external agent-trace corpus or
model-generated teacher outputs are used.

`agent_eval_*.jsonl` files are separate fixed sets of prompt/input
cases with expected stdout for end-to-end evaluation. Corpus generation
reserves the small baseline manifest; larger manifests are created from input
cases verified absent from the corpus so the evaluator can compare checkpoints
without leaking test inputs into training. Expected outputs are held by the
evaluator and runtime checker, not placed in model prompts.

## Earlier instruction-to-code corpus

The starter prompt/code corpus is derived from
[flytech/python-codes-25k](https://huggingface.co/datasets/flytech/python-codes-25k).

- The Hugging Face dataset card lists the MIT license and fields for
  instruction, optional input, and Python output.
- `scripts/prepare_code_pairs.py` extracts fenced Python or plain code, keeps
  unique syntax-valid examples within the selected byte length, and adds a
  “Return Python code only” suffix.
- Preserve upstream attribution and license terms if distributing derived
  data or model artifacts.
