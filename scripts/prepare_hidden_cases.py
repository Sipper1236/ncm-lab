from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

from evaluate_agent import expected_stdout, known_inputs
from prepare_agent_pairs import TASKS, make_case


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Attach deterministic hidden inputs to a fixed agent manifest."
    )
    parser.add_argument("--manifest", default="data/agent_eval_repairs_203.jsonl")
    parser.add_argument("--data", default="data/agent_pairs.jsonl")
    parser.add_argument(
        "--output", default="data/agent_eval_repairs_203_multitest.jsonl"
    )
    parser.add_argument("--hidden-cases", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261126)
    return parser


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on {path}:{line_number}: {error}"
                ) from error
    return rows


def build_hidden_cases(
    family: str,
    visible_stdin: str,
    visible_output: str,
    count: int,
    seed: int,
    already_seen: set[str],
) -> list[dict[str, str]]:
    hidden: list[dict[str, str]] = []
    seen_outputs = {visible_output}

    if family == "count_lines":
        line_counts = (1, 5, 2, 4, 6, 7, 8, 9, 3)
        case_index = 0
        for line_count in line_counts:
            output = f"{line_count}\n"
            if output in seen_outputs:
                continue
            token = f"hidden-{seed}-{case_index}"
            lines = [f"{token}-{index}" for index in range(line_count)]
            stdin = "\n  \n".join(lines)
            if stdin == visible_stdin or stdin in already_seen:
                raise ValueError(
                    "Could not create a distinct count_lines hidden input"
                )
            already_seen.add(stdin)
            hidden.append(
                {"stdin": stdin, "expected_stdout": expected_stdout(family, stdin)}
            )
            seen_outputs.add(output)
            case_index += 1
            if len(hidden) == count:
                break
        if len(hidden) != count:
            raise ValueError("Could not create enough distinct count_lines hidden cases")
        return hidden

    family_index = list(TASKS).index(family)
    for hidden_index in range(count):
        chosen: tuple[str, str] | None = None
        for attempt in range(10_000):
            case_seed = seed + family_index * 1009 + hidden_index * 97 + attempt
            rng = random.Random(case_seed)
            start_index = rng.randrange(100_000)
            for offset in range(64):
                stdin, _ = make_case(family, start_index + offset, rng)
                if stdin in already_seen:
                    continue
                actual = expected_stdout(family, stdin)
                if actual in seen_outputs:
                    continue
                already_seen.add(stdin)
                seen_outputs.add(actual)
                chosen = (stdin, actual)
                break
            if chosen is not None:
                break
        if chosen is None:
            raise ValueError(
                f"Could not generate a distinct hidden result for {family}"
            )
        hidden.append({"stdin": chosen[0], "expected_stdout": chosen[1]})
    return hidden


def main() -> None:
    args = build_parser().parse_args()
    if args.hidden_cases < 1 or args.hidden_cases > 8:
        raise SystemExit("--hidden-cases must be between 1 and 8")

    source_path = Path(args.manifest)
    rows = read_jsonl(source_path)
    metadata = next(
        (row for row in rows if row.get("type") == "manifest_metadata"), {}
    )
    episodes = [row for row in rows if row.get("type") != "manifest_metadata"]
    if not episodes:
        raise SystemExit(f"Manifest contains no episodes: {source_path}")

    seen = known_inputs(Path(args.data))
    visible_inputs: dict[str, list[str]] = {}
    for row in episodes:
        family, stdin = row["family"], row["stdin"]
        seen.setdefault(family, set()).add(stdin)
        if stdin not in visible_inputs.setdefault(family, []):
            visible_inputs[family].append(stdin)

    input_manifest_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    hidden_by_visible: dict[tuple[str, str], list[dict[str, str]]] = {}
    families = dict.fromkeys(row["family"] for row in episodes)
    for family_index, family in enumerate(families):
        for input_index, stdin in enumerate(visible_inputs[family]):
            episode = next(
                row
                for row in episodes
                if row["family"] == family and row["stdin"] == stdin
            )
            hidden_by_visible[(family, stdin)] = build_hidden_cases(
                family,
                stdin,
                episode["expected_stdout"],
                args.hidden_cases,
                args.seed + family_index * 1009 + input_index * 53,
                seen.setdefault(family, set()),
            )

    for row in episodes:
        row["verification_cases"] = hidden_by_visible[(row["family"], row["stdin"])]

    output_metadata = {
        **metadata,
        "type": "manifest_metadata",
        "schema_version": 2,
        "parent_manifest": str(source_path),
        "parent_manifest_sha256": input_manifest_sha256,
        "training_data_sha256": hashlib.sha256(
            Path(args.data).read_bytes()
        ).hexdigest(),
        "hidden_cases_per_script": args.hidden_cases,
        "hidden_case_seed": args.seed,
        "hidden_case_policy": (
            "unseen inputs with outputs distinct from the visible case"
        ),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(output_metadata, ensure_ascii=False) + "\n")
        for row in episodes:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "episodes": len(episodes),
                "hidden_cases_per_script": args.hidden_cases,
                "unique_visible_inputs": sum(
                    len(items) for items in visible_inputs.values()
                ),
                "output": str(output_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
