from __future__ import annotations

import argparse
import json
import random
import tempfile
from pathlib import Path
from typing import Any

from ncm_lab.agent_tools import (
    compact_json,
    execute_tool,
    format_agent_prompt,
    make_final_action,
    make_tool_action,
)

TASKS = {
    "count_lines": [
        "Write a Python script that prints the number of non-empty input lines.",
        "Create a Python script to count nonblank lines from standard input.",
        "Read stdin and report how many lines contain non-whitespace text.",
    ],
    "sum_numbers": [
        "Write a Python script that reads integers from stdin and prints their sum.",
        "Create a script to total the whitespace-separated input integers.",
        "Read the numbers from standard input and print their sum using Python.",
    ],
    "count_words": [
        "Write a Python script that prints the number of words in stdin.",
        "Count whitespace-separated words from standard input and print the count.",
        "Create a script to report how many words the input contains.",
    ],
    "sort_numbers": [
        "Write a Python script that sorts input integers, printing one per line.",
        "Sort the whitespace-separated integers from stdin in ascending order.",
        "Create a script that reads numbers and prints them sorted on separate lines.",
    ],
    "reverse_lines": [
        "Write a Python script that reverses each input line and prints the results.",
        "Reverse the characters on every stdin line using Python.",
        "Create a script that prints each input line backwards, preserving line order.",
    ],
    "mean_numbers": [
        "Write a Python script that prints the mean of the numbers read from stdin.",
        "Read whitespace-separated values and calculate their arithmetic average.",
        "Create a script to print the average of the input numbers.",
    ],
    "csv_total": [
        "Read CSV from stdin with a header and an amount column; print the amount total.",
        "Write a Python script to sum the amount column in a CSV table with headers.",
        "Sum the CSV amount values from standard input, skipping the header row.",
    ],
    "json_max": [
        "Read a JSON object from stdin and print the largest value in its scores list.",
        "Write a script that loads JSON input and prints max(data['scores']).",
        "Find and print the maximum score from the input JSON object's scores array.",
    ],
    "unique_sorted": [
        "Print unique input integers in ascending order, one number on each line.",
        "Write a script that deduplicates stdin integers and prints them sorted.",
        "Read integers from stdin and output the sorted distinct values line by line.",
    ],
}

CODE = {
    "count_lines": (
        "import sys;print(sum(bool(x.strip()) for x in sys.stdin.read().splitlines()))",
        "import sys;print(sum(bool(x.strip()) for x in sys.stdin.read().splitlines))",
    ),
    "sum_numbers": (
        "import sys;print(sum(map(int,sys.stdin.read().split())))",
        "import sys;print(sum(map(int,sys.stdin.read())))",
    ),
    "count_words": (
        "import sys;print(len(sys.stdin.read().split()))",
        "import sys;print(len(sys.stdin.read().split))",
    ),
    "sort_numbers": (
        'import sys;print(*sorted(map(int,sys.stdin.read().split())),sep="\\n")',
        'import sys;print(*sorted(map(int,sys.stdin.read())),sep="\\n")',
    ),
    "reverse_lines": (
        'import sys;print("\\n".join(x[::-1] for x in sys.stdin.read().splitlines()))',
        'import sys;print("\\n".join(x[::-1] for x in sys.stdin.read().splitlines))',
    ),
    "mean_numbers": (
        "import sys;v=list(map(float,sys.stdin.read().split()));print(sum(v)/len(v))",
        "import sys;v=list(map(float,sys.stdin.read().split()));print(sum(v)/len(v)+1)",
    ),
    "csv_total": (
        "import csv,sys;r=csv.reader(sys.stdin);next(r);print(sum(int(x[1]) for x in r))",
        "import csv,sys;r=csv.reader(sys.stdin);print(sum(int(x[1]) for x in r))",
    ),
    "json_max": (
        'import json,sys;print(max(json.load(sys.stdin)["scores"]))',
        'import json,sys;print(max(json.load(sys.stdin)["score"]))',
    ),
    "unique_sorted": (
        'import sys;print(*sorted(set(map(int,sys.stdin.read().split()))),sep="\\n")',
        'import sys;print(*sorted(set(map(int,sys.stdin.read()))),sep="\\n")',
    ),
}
OBSERVATION_CACHE: dict[tuple[str, str], dict[str, Any]] = {}


def make_case(kind: str, index: int, rng: random.Random) -> tuple[str, str]:
    if kind == "count_lines":
        words = ["alpha", "", "  ", "beta", "gamma", ""]
        rng.shuffle(words)
        return "\n".join(words), "\n"
    if kind == "sum_numbers":
        values = [rng.randint(-25, 90) for _ in range(5)]
        return " ".join(map(str, values)), "\n"
    if kind == "count_words":
        words = [rng.choice(["red", "blue", "green", "small", "quick"]) for _ in range(3 + index % 8)]
        return " ".join(words), "\n"
    if kind == "sort_numbers":
        values = [rng.randint(-30, 80) for _ in range(5)]
        return " ".join(map(str, values)), "\n"
    if kind == "reverse_lines":
        words = [rng.choice(["north", "river", "maple", "cedar", "stone"]) for _ in range(3)]
        return "\n".join(words), "\n"
    if kind == "mean_numbers":
        values = [round(rng.uniform(-8, 25), 1) for _ in range(4)]
        return " ".join(map(str, values)), "\n"
    if kind == "csv_total":
        values = [rng.randint(1, 500) for _ in range(4)]
        rows = ["item,amount"] + [f"item{i},{value}" for i, value in enumerate(values)]
        return "\n".join(rows), "\n"
    if kind == "json_max":
        values = [rng.randint(-50, 100) for _ in range(5)]
        return json.dumps({"scores": values}), "\n"
    values = [rng.randint(-25, 25) for _ in range(6)]
    return " ".join(map(str, values)), "\n"


def add_episode(
    pairs: list[dict[str, str]],
    group: str,
    task: str,
    stdin: str,
    correct_code: str,
    initial_code: str,
    repair: bool,
    prefix_bytes: int,
    seq_len: int,
) -> None:
    history: list[dict[str, str]] = []

    def decide(action: dict[str, Any]) -> dict[str, Any] | None:
        prompt = format_agent_prompt(task, history, prefix_bytes)
        completion = compact_json(action)
        if len(prompt.encode("utf-8")) > prefix_bytes:
            raise ValueError(f"prompt exceeds prefix window in {group}")
        if len(completion.encode("utf-8")) + 1 > seq_len:
            raise ValueError(f"action exceeds output window in {group}: {completion}")
        pairs.append({"prompt": prompt, "completion": completion, "group": group})
        if "tool" not in action:
            return None
        if action["tool"] == "run_script":
            key = (
                (environment["workdir"] / "main.py").read_text(encoding="utf-8"),
                environment["stdin"],
            )
            observation = OBSERVATION_CACHE.get(key)
            if observation is None:
                observation = execute_tool(action["tool"], action["args"], environment)
                OBSERVATION_CACHE[key] = observation
        else:
            observation = execute_tool(action["tool"], action["args"], environment)
        history.append({"role": "assistant", "content": completion})
        history.append(
            {"role": f"tool:{action['tool']}", "content": compact_json(observation)}
        )
        return observation

    with tempfile.TemporaryDirectory(prefix="ncm-agent-data-") as directory:
        environment = {"workdir": Path(directory), "stdin": stdin}
        if repair:
            (environment["workdir"] / "main.py").write_text(initial_code, encoding="utf-8")
            first_result = decide(make_tool_action("run_script", {}))
            written = decide(make_tool_action("write_script", {"code": correct_code}))
            if not written or written.get("written") is not True:
                raise RuntimeError(f"failed to write corrected script for {group}")
            result = decide(make_tool_action("run_script", {}))
        else:
            written = decide(make_tool_action("write_script", {"code": correct_code}))
            if not written or written.get("written") is not True:
                raise RuntimeError(f"failed to write generated script for {group}: {written}")
            result = decide(make_tool_action("run_script", {}))

        if not isinstance(result, dict) or result.get("exit_code") != 0:
            raise RuntimeError(f"generated script did not run successfully for {group}: {result}")
        if repair and isinstance(first_result, dict):
            failed = first_result.get("exit_code") != 0
            wrong_output = first_result.get("stdout") != result.get("stdout")
            if not failed and not wrong_output:
                raise RuntimeError(f"repair example had no observable failure for {group}")
        decide(make_final_action("Done. main.py runs successfully on the sample input."))


def build_dataset(
    episodes_per_task: int,
    seed: int,
    prefix_bytes: int,
    seq_len: int,
) -> list[dict[str, str]]:
    rng = random.Random(seed)
    pairs: list[dict[str, str]] = []
    case_banks: dict[str, list[tuple[str, str]]] = {}
    for kind_index, kind in enumerate(TASKS):
        case_rng = random.Random(seed + 1009 * kind_index)
        case_banks[kind] = [
            make_case(kind, case_index, case_rng) for case_index in range(16)
        ]

    for kind, task_templates in TASKS.items():
        for index in range(episodes_per_task):
            task = rng.choice(task_templates)
            sample_index = index % len(case_banks[kind])
            stdin, _ = case_banks[kind][sample_index]
            correct_code, buggy_code = CODE[kind]
            repair = index % 4 == 0
            if repair:
                task += "\nCURRENT_SCRIPT:" + json.dumps(
                    buggy_code, ensure_ascii=False
                )
            task = f"{task}\nSAMPLE_STDIN:{json.dumps(stdin, ensure_ascii=False)}"
            add_episode(
                pairs,
                f"{kind}:sample{sample_index:02d}",
                task,
                stdin,
                correct_code,
                buggy_code if repair else correct_code,
                repair,
                prefix_bytes,
                seq_len,
            )
    rng.shuffle(pairs)
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic Python coding-agent tool-use trajectories for NCM."
    )
    parser.add_argument("--output", default="data/agent_pairs.jsonl")
    parser.add_argument("--episodes-per-task", type=int, default=256)
    parser.add_argument("--prefix-bytes", type=int, default=510)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.episodes_per_task < 4 or args.prefix_bytes < 128 or args.seq_len < 64:
        raise SystemExit(
            "Need at least 4 episodes per task, 128 prompt bytes, and 64 output bytes"
        )

    pairs = build_dataset(
        args.episodes_per_task,
        args.seed,
        args.prefix_bytes,
        args.seq_len,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in pairs),
        encoding="utf-8",
    )
    groups = {row["group"] for row in pairs}
    counts: dict[str, int] = {}
    for row in pairs:
        action = json.loads(row["completion"])
        key = action.get("tool", "final")
        counts[key] = counts.get(key, 0) + 1
    print(
        f"synthetic_episodes={args.episodes_per_task * len(TASKS)} "
        f"sample_groups={len(groups)} decision_pairs={len(pairs)} "
        f"actions={json.dumps(counts, sort_keys=True)} output={output_path}"
    )


if __name__ == "__main__":
    main()
