from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import json
import random
import re
import time
from pathlib import Path
from typing import Any

from ncm_lab.agent import run_agent
from prepare_agent_pairs import CODE, TASKS, make_case


PARAPHRASES = {
    "count_lines": "How many lines in this input contain non-whitespace text? Print the count.",
    "sum_numbers": "What do all the supplied integers add up to? Print the result.",
    "count_words": "Read this text with Python and report how many words it contains.",
    "sort_numbers": "Output these numbers from lowest to highest, each on a new line.",
    "reverse_lines": "Print the input lines in the same order, but spell each line backwards.",
    "mean_numbers": "Calculate the average value of the numbers I provide and print it.",
    "csv_total": "In the CSV input, total the values in the column headed amount.",
    "json_max": "From the supplied JSON, output the highest number in the scores array.",
    "unique_sorted": "Print the distinct input integers in increasing order, one per line.",
}

USER_REGRESSION = "Write a Python script that counts words in stdin."


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure end-to-end NCM Python-agent task success."
    )
    parser.add_argument("--tokenizer", default="artifacts/agent/tokenizer.pt")
    parser.add_argument("--generator", default="artifacts/agent/generator.pt")
    parser.add_argument("--data", default="data/agent_pairs.jsonl")
    parser.add_argument("--manifest", default="data/agent_eval_manifest.jsonl")
    parser.add_argument("--output", default="artifacts/eval/rollouts.jsonl")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--cases-per-family", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--families", nargs="+", choices=sorted(TASKS))
    parser.add_argument(
        "--skip-repairs",
        action="store_true",
        help="Do not run the faulty-script repair episode for each family.",
    )
    return parser


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def known_inputs(path: Path) -> dict[str, set[str]]:
    known: dict[str, set[str]] = {}
    if not path.is_file():
        return known
    pattern = re.compile(r"SAMPLE_STDIN:(.*?)(?:\nHISTORY:|$)", re.DOTALL)
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            match = pattern.search(row.get("prompt", ""))
            if match:
                try:
                    value = json.loads(match.group(1).strip())
                except json.JSONDecodeError:
                    continue
                group = row.get("group", "")
                family = group.split(":", 1)[0] if isinstance(group, str) else ""
                if isinstance(value, str):
                    known.setdefault(family, set()).add(value)
    return known


def build_manifest(
    families: list[str],
    cases_per_family: int,
    seed: int,
    known: dict[str, set[str]],
    include_repairs: bool,
) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    for family_index, family in enumerate(families):
        cases = unseen_cases(
            family,
            cases_per_family,
            seed + family_index * 1009,
            known.setdefault(family, set()),
        )
        for stdin in cases:
            expected = expected_stdout(family, stdin)
            for variant, task in prompt_variants(family):
                manifest.append(
                    {
                        "family": family,
                        "variant": variant,
                        "task": task,
                        "stdin": stdin,
                        "expected_stdout": expected,
                        "initial_script": "",
                    }
                )
            if include_repairs:
                _, faulty_script = CODE[family]
                manifest.append(
                    {
                        "family": family,
                        "variant": "repair",
                        "task": TASKS[family][0],
                        "stdin": stdin,
                        "expected_stdout": expected,
                        "initial_script": faulty_script,
                    }
                )
    return manifest


def load_or_create_manifest(
    path: Path,
    families: list[str],
    args: argparse.Namespace,
    known: dict[str, set[str]],
) -> list[dict[str, Any]]:
    training_inputs = {family: set(values) for family, values in known.items()}
    if path.is_file():
        rows: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid evaluation manifest row {line_number}: {error}"
                    ) from error
                if row.get("type") != "manifest_metadata":
                    rows.append(row)
        rows = [row for row in rows if row.get("family") in families]
        if args.skip_repairs:
            rows = [row for row in rows if row.get("variant") != "repair"]
    else:
        used_inputs = {family: set(values) for family, values in training_inputs.items()}
        rows = build_manifest(
            families,
            args.cases_per_family,
            args.seed,
            used_inputs,
            include_repairs=not args.skip_repairs,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "type": "manifest_metadata",
            "seed": args.seed,
            "cases_per_family": args.cases_per_family,
            "families": families,
            "includes_repairs": not args.skip_repairs,
        }
        with path.open("w", encoding="utf-8") as stream:
            stream.write(json.dumps(metadata, ensure_ascii=False) + "\n")
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    for row in rows:
        family = row.get("family")
        stdin = row.get("stdin")
        if family not in TASKS or not isinstance(stdin, str):
            raise ValueError(f"Invalid evaluation case in {path}: {row}")
        if stdin in training_inputs.get(family, set()):
            raise ValueError(
                f"Evaluation input for {family} appears in training data. "
                "Regenerate the corpus with --exclude-manifest pointing to "
                f"{path}, or choose a new manifest."
            )
        if row.get("expected_stdout") != expected_stdout(family, stdin):
            raise ValueError(f"Incorrect oracle output in evaluation manifest: {row}")
    return rows


def unseen_cases(
    family: str,
    count: int,
    seed: int,
    already_seen: set[str],
) -> list[str]:
    rng = random.Random(seed)
    result: list[str] = []
    for case_index in range(100_000):
        stdin, _ = make_case(family, case_index, rng)
        if stdin in already_seen:
            continue
        already_seen.add(stdin)
        result.append(stdin)
        if len(result) == count:
            return result
    raise RuntimeError(
        f"Could not generate {count} unseen inputs for {family}; "
        "increase the task's input variety."
    )


def expected_stdout(family: str, stdin: str) -> str:
    if family == "count_lines":
        value = sum(bool(line.strip()) for line in stdin.splitlines())
        return f"{value}\n"
    if family == "sum_numbers":
        return f"{sum(map(int, stdin.split()))}\n"
    if family == "count_words":
        return f"{len(stdin.split())}\n"
    if family == "sort_numbers":
        values = sorted(map(int, stdin.split()))
        return "".join(f"{value}\n" for value in values)
    if family == "reverse_lines":
        return "".join(f"{line[::-1]}\n" for line in stdin.splitlines())
    if family == "mean_numbers":
        values = list(map(float, stdin.split()))
        return f"{sum(values) / len(values)}\n"
    if family == "csv_total":
        rows = csv.reader(io.StringIO(stdin))
        next(rows, None)
        return f"{sum(int(row[1]) for row in rows)}\n"
    if family == "json_max":
        values = json.loads(stdin)["scores"]
        return f"{max(values)}\n"
    if family == "unique_sorted":
        values = sorted(set(map(int, stdin.split())))
        return "".join(f"{value}\n" for value in values)
    raise ValueError(f"Unknown task family: {family}")


def prompt_variants(family: str) -> list[tuple[str, str]]:
    variants = [("training_template", TASKS[family][0])]
    if family == "count_words":
        variants.append(("user_regression", USER_REGRESSION))
    variants.append(("paraphrase", PARAPHRASES[family]))
    return variants


def evaluate_episode(
    *,
    family: str,
    variant: str,
    task: str,
    stdin: str,
    expected: str,
    args: argparse.Namespace,
    initial_script: str = "",
) -> dict[str, Any]:
    from argparse import Namespace

    trace: list[dict[str, Any]] = []
    agent_args = Namespace(
        tokenizer=args.tokenizer,
        generator=args.generator,
        task=task,
        stdin=stdin,
        initial_script=initial_script,
        device=args.device,
        max_steps=args.max_steps,
        temperature=0.0,
        top_k=16,
    )
    output = io.StringIO()
    started = time.perf_counter()
    runtime_error: str | None = None
    try:
        with contextlib.redirect_stdout(output):
            final = run_agent(agent_args, trace=trace)
    except Exception as error:  # Keep other task-family results if one episode crashes.
        final = None
        runtime_error = f"{type(error).__name__}: {error}"
    elapsed = time.perf_counter() - started

    run_observations = [
        event
        for event in trace
        if event.get("type") == "observation" and event.get("tool") == "run_script"
    ]
    last_run = run_observations[-1] if run_observations else None
    result = last_run.get("result", {}) if last_run else {}
    stdout = result.get("stdout") if isinstance(result, dict) else None
    exit_code = result.get("exit_code") if isinstance(result, dict) else None
    exact = exit_code == 0 and stdout == expected
    actions = [event for event in trace if event.get("type") == "action"]
    invalid_actions = [event for event in trace if event.get("type") == "invalid_action"]
    parser_recoveries = [
        event for event in actions if event.get("parser_recovered") is True
    ]
    final_accepted = any(
        event.get("type") == "final" and event.get("accepted") is True
        for event in trace
    )
    return {
        "family": family,
        "variant": variant,
        "repair": bool(initial_script),
        "task": task,
        "stdin": stdin,
        "expected_stdout": expected,
        "actual_stdout": stdout,
        "last_exit_code": exit_code,
        "exact_output": exact,
        "final_accepted": final_accepted,
        "task_success": exact and final_accepted,
        "decision_count": len(actions) + len(invalid_actions),
        "tool_call_count": sum("tool" in event.get("action", {}) for event in actions),
        "invalid_action_count": len(invalid_actions),
        "parser_recovery_count": len(parser_recoveries),
        "elapsed_seconds": round(elapsed, 3),
        "final_text": final,
        "runtime_error": runtime_error,
        "trace": trace,
        "console": output.getvalue(),
    }


def main() -> None:
    args = build_parser().parse_args()
    if args.cases_per_family < 1:
        raise SystemExit("--cases-per-family must be positive")
    if args.max_steps < 1:
        raise SystemExit("--max-steps must be positive")

    families = args.families or list(TASKS)
    seen = known_inputs(Path(args.data))
    manifest_path = Path(args.manifest)
    manifest = load_or_create_manifest(manifest_path, families, args, seen)
    episodes: list[dict[str, Any]] = []
    for index, case in enumerate(manifest, start=1):
        episodes.append(
            evaluate_episode(
                family=case["family"],
                variant=case["variant"],
                task=case["task"],
                stdin=case["stdin"],
                expected=case["expected_stdout"],
                args=args,
                initial_script=case.get("initial_script", ""),
            )
        )
        print(f"evaluated {case['family']} {case['variant']} ({index}/{len(manifest)})")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "type": "run_metadata",
        "seed": args.seed,
        "device": args.device,
        "selected_device": next(
            (
                line.split("=", 1)[1]
                for episode in episodes
                for line in episode["console"].splitlines()
                if line.startswith("vulkan_device=")
            ),
            None,
        ),
        "tokenizer_sha256": file_sha256(Path(args.tokenizer)),
        "generator_sha256": file_sha256(Path(args.generator)),
        "data_sha256": file_sha256(Path(args.data)) if Path(args.data).is_file() else None,
        "manifest_sha256": file_sha256(manifest_path),
        "episodes": len(episodes),
    }
    with output_path.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(metadata, ensure_ascii=False) + "\n")
        for episode in episodes:
            stream.write(json.dumps(episode, ensure_ascii=False) + "\n")

    model_outputs = sum(item["decision_count"] for item in episodes)
    invalid_actions = sum(item["invalid_action_count"] for item in episodes)
    parser_recoveries = sum(item["parser_recovery_count"] for item in episodes)
    per_family: dict[str, dict[str, int]] = {}
    for episode in episodes:
        row = per_family.setdefault(
            episode["family"], {"episodes": 0, "successes": 0, "repairs": 0}
        )
        row["episodes"] += 1
        row["successes"] += int(episode["task_success"])
        row["repairs"] += int(episode["repair"] and episode["task_success"])
    summary = {
        "episodes": len(episodes),
        "exact_output_successes": sum(item["exact_output"] for item in episodes),
        "task_successes": sum(item["task_success"] for item in episodes),
        "finals_accepted": sum(item["final_accepted"] for item in episodes),
        "false_finals": sum(
            item["final_accepted"] and not item["exact_output"] for item in episodes
        ),
        "model_outputs": model_outputs,
        "raw_valid_action_rate": round(
            max(0, model_outputs - invalid_actions - parser_recoveries)
            / max(1, model_outputs),
            4,
        ),
        "invalid_actions": invalid_actions,
        "parser_recoveries": parser_recoveries,
        "repair_episodes": sum(item["repair"] for item in episodes),
        "per_family": per_family,
        "output": str(output_path),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
