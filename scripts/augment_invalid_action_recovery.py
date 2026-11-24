from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from ncm_lab.agent_tools import compact_json, format_agent_prompt, parse_action
from augment_repeat_recovery import parse_prompt_history


def malformed_code_retry_row(
    row: dict[str, Any], max_bytes: int
) -> dict[str, Any] | None:
    prompt = row.get("prompt")
    completion = row.get("completion")
    if not isinstance(prompt, str) or not isinstance(completion, str):
        raise ValueError("Each input row needs string prompt and completion fields")
    try:
        action = json.loads(completion)
    except json.JSONDecodeError:
        return None
    if not isinstance(action, dict) or action.get("tool") != "write_script":
        return None

    parsed = parse_prompt_history(prompt)
    if parsed is None:
        raise ValueError("Could not parse an input prompt")
    task, history = parsed
    if history and history[-1] == {
        "role": "tool:runtime",
        "content": compact_json({"error": "invalid_action"}),
    }:
        return None

    malformed = completion.replace('"code":', '"code"::', 1)
    if malformed == completion:
        return None
    try:
        parse_action(malformed)
    except ValueError:
        pass
    else:
        raise ValueError("Generated malformed write_script action was accepted")

    retry_history = [
        *history,
        {"role": "assistant", "content": malformed[:48]},
        {
            "role": "tool:runtime",
            "content": compact_json({"error": "invalid_action"}),
        },
    ]
    retry_prompt = format_agent_prompt(task, retry_history, max_bytes)
    if len(retry_prompt.encode("utf-8")) > max_bytes:
        raise ValueError("Could not fit malformed-action prompt in the byte limit")

    augmented = dict(row)
    augmented["prompt"] = retry_prompt
    augmented["recovery_type"] = "malformed_write_script"
    return augmented


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Add repair examples for malformed write_script JSON."
    )
    parser.add_argument("--input", default="data/agent_pairs.jsonl")
    parser.add_argument(
        "--output", default="data/agent_pairs_malformed_code_recovery_v1.jsonl"
    )
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=510,
        help="Prompt byte budget used by the agent before tokenization.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    if args.max_bytes < 1:
        raise SystemExit("--max-bytes must be positive")
    if input_path.resolve() == output_path.resolve():
        raise SystemExit("--output must differ from --input")

    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        input_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON on line {line_number}: {error}") from error
        if not isinstance(row, dict):
            raise ValueError(f"Line {line_number} must contain a JSON object")
        rows.append(row)

    seen = {
        (row.get("prompt"), row.get("completion"))
        for row in rows
        if isinstance(row.get("prompt"), str)
        and isinstance(row.get("completion"), str)
    }
    augmented_rows: list[dict[str, Any]] = []
    skipped = {"other_action": 0, "already_invalid_retry": 0, "duplicate": 0}
    for row in rows:
        if row.get("recovery_type") == "malformed_write_script":
            continue
        completion = row.get("completion")
        try:
            action = json.loads(completion) if isinstance(completion, str) else None
        except json.JSONDecodeError:
            action = None
        if not isinstance(action, dict) or action.get("tool") != "write_script":
            skipped["other_action"] += 1
            continue
        augmented = malformed_code_retry_row(row, args.max_bytes)
        if augmented is None:
            skipped["already_invalid_retry"] += 1
            continue
        key = (augmented["prompt"], augmented["completion"])
        if key in seen:
            skipped["duplicate"] += 1
            continue
        seen.add(key)
        augmented_rows.append(augmented)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        for row in [*rows, *augmented_rows]:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "input": str(input_path),
                "input_sha256": sha256(input_path),
                "output": str(output_path),
                "output_sha256": sha256(output_path),
                "base_rows": len(rows),
                "malformed_write_script_rows": len(augmented_rows),
                "output_rows": len(rows) + len(augmented_rows),
                "skipped": skipped,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
