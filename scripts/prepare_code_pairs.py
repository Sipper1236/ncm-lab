from __future__ import annotations

import argparse
import ast
import json
import random
import re
import urllib.request
import warnings
from pathlib import Path

DATASET = "flytech/python-codes-25k"
DATA_URL = (
    "https://huggingface.co/datasets/flytech/python-codes-25k/"
    "resolve/main/python-codes-25k.jsonl?download=true"
)
FENCE = chr(96) * 3
CODE_BLOCK = re.compile(
    re.escape(FENCE) + r"(?:python|py)?\s*(.*?)" + re.escape(FENCE),
    flags=re.IGNORECASE | re.DOTALL,
)


def download_dataset(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return
    with urllib.request.urlopen(DATA_URL, timeout=120) as response:
        path.write_bytes(response.read())


def extract_python(output: str) -> str | None:
    blocks = CODE_BLOCK.findall(output)
    code = "\n\n".join(blocks).strip() if blocks else output.strip()
    if not code:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(code)
    except SyntaxError:
        return None
    if not tree.body:
        return None
    code_nodes = (
        ast.FunctionDef,
        ast.AsyncFunctionDef,
        ast.ClassDef,
        ast.Import,
        ast.ImportFrom,
        ast.Assign,
        ast.AnnAssign,
        ast.AugAssign,
        ast.For,
        ast.AsyncFor,
        ast.While,
        ast.If,
        ast.With,
        ast.AsyncWith,
        ast.Try,
        ast.Call,
    )
    if not any(isinstance(node, code_nodes) for node in ast.walk(tree)):
        return None
    return code


def fit_utf8(text: str, max_bytes: int) -> str:
    return text.encode("utf-8")[:max_bytes].decode("utf-8", errors="ignore").strip()


def make_prompt(instruction: str, extra_input: str, max_bytes: int) -> str:
    suffix = "\n\nReturn Python code only."
    budget = max_bytes - len(suffix.encode("utf-8"))
    if budget <= 0:
        raise ValueError("prefix length is too small for the prompt suffix")

    instruction = fit_utf8(instruction, budget)
    remaining = budget - len(instruction.encode("utf-8"))
    if extra_input and remaining > len("\nInput: ".encode("utf-8")):
        label = "\nInput: "
        extra_input = fit_utf8(
            extra_input,
            remaining - len(label.encode("utf-8")),
        )
        if extra_input:
            instruction += label + extra_input
    return instruction + suffix


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare short Python instruction/code pairs."
    )
    parser.add_argument("--output", default="data/pairs.jsonl")
    parser.add_argument("--raw-file", default="data/raw/python-codes-25k.jsonl")
    parser.add_argument("--max-examples", type=int, default=5000)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--prefix-len", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.max_examples < 1 or args.seq_len < 8 or args.prefix_len < 32:
        raise SystemExit(
            "--max-examples must be positive, --seq-len at least 8, "
            "and --prefix-len at least 32"
        )

    raw_path = Path(args.raw_file)
    download_dataset(raw_path)
    candidates: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    malformed = 0
    too_long = 0
    invalid_python = 0
    with raw_path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            instruction = str(row.get("instruction") or "").strip()
            extra_input = str(row.get("input") or "").strip()
            output = str(row.get("output") or "").strip()
            code = extract_python(output)
            if not instruction or code is None:
                invalid_python += 1
                continue
            if len(code.encode("utf-8")) + 1 > args.seq_len:
                too_long += 1
                continue
            key = (instruction, code)
            if key in seen:
                continue
            seen.add(key)
            prompt = make_prompt(instruction, extra_input, args.prefix_len - 2)
            candidates.append(
                {
                    "prompt": prompt,
                    "completion": code,
                }
            )

    if not candidates:
        raise RuntimeError("No unique valid Python examples fit the target length")
    rng = random.Random(args.seed)
    rng.shuffle(candidates)
    selected = candidates[: args.max_examples]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in selected),
        encoding="utf-8",
    )
    print(
        f"dataset={DATASET} unique_valid={len(candidates)} selected={len(selected)} "
        f"too_long={too_long} invalid_or_non_code={invalid_python} "
        f"malformed={malformed} output={output_path}"
    )


if __name__ == "__main__":
    main()
