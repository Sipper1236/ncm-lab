from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from ncm_lab.agent_tools import compact_json, format_agent_prompt

HISTORY_ROLES = (
    "tool:write_script",
    "tool:run_script",
    "tool:runtime",
    "assistant",
)


def parse_prompt_history(prompt: str) -> tuple[str, list[dict[str, str]]] | None:
    header, history_marker, history_and_end = prompt.partition("\nHISTORY:")
    if not history_marker:
        return None
    history_text, next_marker, _ = history_and_end.partition("\nNEXT:")
    if not next_marker:
        return None
    _, task_marker, task = header.partition("\nTASK:")
    if not task_marker:
        return None

    history: list[dict[str, str]] = []
    if history_text != "none":
        for line in history_text.splitlines():
            for role in HISTORY_ROLES:
                role_prefix = f"{role}:"
                if line.startswith(role_prefix):
                    history.append(
                        {"role": role, "content": line[len(role_prefix) :]}
                    )
                    break
            else:
                return None
    return task, history


def repeat_recovery_row(
    row: dict[str, Any], max_bytes: int
) -> dict[str, Any] | None:
    prompt = row.get("prompt")
    completion = row.get("completion")
    if not isinstance(prompt, str) or not isinstance(completion, str):
        raise ValueError("Each input row needs string prompt and completion fields")
    parsed = parse_prompt_history(prompt)
    if parsed is None:
        return None
    task, history = parsed
    if not history:
        return None

    if history[-1] == {
        "role": "tool:runtime",
        "content": compact_json({"error": "invalid_action"}),
    }:
        return None

    previous_action = next(
        (
            event["content"]
            for event in reversed(history)
            if event["role"] == "assistant"
        ),
        None,
    )
    if previous_action is None:
        return None

    repeated_history = [
        *history,
        {"role": "assistant", "content": previous_action[:48]},
        {
            "role": "tool:runtime",
            "content": compact_json({"error": "repeated_action"}),
        },
    ]
    repeated_prompt = format_agent_prompt(task, repeated_history, max_bytes)
    if len(repeated_prompt.encode("utf-8")) > max_bytes:
        raise ValueError("Could not fit repeat-recovery prompt in the byte limit")

    augmented = dict(row)
    augmented["prompt"] = repeated_prompt
    augmented["recovery_type"] = "repeated_action"
    return augmented


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Add supervised next actions after repeated-action feedback."
    )
    parser.add_argument("--input", default="data/agent_pairs.jsonl")
    parser.add_argument(
        "--output", default="data/agent_pairs_repeat_recovery_v1.jsonl"
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
    source_lines = input_path.read_text(encoding="utf-8").splitlines()
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(source_lines, start=1):
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
    skipped = {
        "no_history": 0,
        "invalid_retry": 0,
        "no_prior_action": 0,
        "duplicate": 0,
    }
    for row in rows:
        if row.get("recovery_type") == "repeated_action":
            continue
        augmented = repeat_recovery_row(row, args.max_bytes)
        if augmented is None:
            parsed = parse_prompt_history(row.get("prompt", ""))
            if parsed is None or not parsed[1]:
                skipped["no_history"] += 1
            elif parsed[1][-1] == {
                "role": "tool:runtime",
                "content": compact_json({"error": "invalid_action"}),
            }:
                skipped["invalid_retry"] += 1
            else:
                skipped["no_prior_action"] += 1
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
                "repeat_recovery_rows": len(augmented_rows),
                "output_rows": len(rows) + len(augmented_rows),
                "skipped": skipped,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
