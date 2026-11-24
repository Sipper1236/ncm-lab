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
        "Count non-empty lines from stdin and print the count.",
        "How many input lines contain text? Read them from stdin and print the number.",
        "Write a script that ignores blank or whitespace-only lines and counts the rest.",
        "Use Python to count lines with at least one non-space character.",
        "Print the number of nonblank lines in the input text.",
    ],
    "sum_numbers": [
        "Write a Python script that reads integers from stdin and prints their sum.",
        "Create a script to total the whitespace-separated input integers.",
        "Read the numbers from standard input and print their sum using Python.",
        "Add all integers from stdin and output the total.",
        "Write a program that sums the numbers in the input text.",
        "Read whitespace-separated integer values and print their sum.",
        "Calculate the total of every input integer, then print it.",
        "Create a Python script to add the integers supplied on standard input.",
    ],
    "count_words": [
        "Write a Python script that prints the number of words in stdin.",
        "Count whitespace-separated words from standard input and print the count.",
        "Create a script to report how many words the input contains.",
        "Write a Python script that counts words in stdin.",
        "Read text from standard input and print how many words it has.",
        "Count the words in the input text and output the number.",
        "How many whitespace-separated words were provided? Print the count.",
        "Create a Python program that counts the words supplied on stdin.",
    ],
    "sort_numbers": [
        "Write a Python script that sorts input integers, printing one per line.",
        "Sort the whitespace-separated integers from stdin in ascending order.",
        "Create a script that reads numbers and prints them sorted on separate lines.",
        "Read integers from stdin, sort them smallest to largest, and print them.",
        "Output the input numbers in ascending numeric order, one on each line.",
        "Write a Python program to sort all supplied integers in increasing order.",
        "Sort the numbers in the input text and print the sorted list.",
        "Put the stdin integers in order and print one value per line.",
    ],
    "reverse_lines": [
        "Write a Python script that reverses each input line and prints the results.",
        "Reverse the characters on every stdin line using Python.",
        "Create a script that prints each input line backwards, preserving line order.",
        "For every line from stdin, reverse its characters and print it.",
        "Read all lines and output each one with its text reversed.",
        "Reverse each line separately; keep the lines in their original order.",
        "Write a program that flips the characters in every input line.",
        "Print each input line backwards without changing their order.",
    ],
    "mean_numbers": [
        "Write a Python script that prints the mean of the numbers read from stdin.",
        "Read whitespace-separated values and calculate their arithmetic average.",
        "Create a script to print the average of the input numbers.",
        "Compute the arithmetic mean of all numbers from standard input.",
        "Read the input values and print their average.",
        "Add the stdin numbers and divide by how many there are; print the result.",
        "Write a Python program that outputs the mean of the supplied values.",
        "Calculate and print the average of the whitespace-separated numbers.",
    ],
    "csv_total": [
        "Read CSV from stdin with a header and an amount column; print the amount total.",
        "Write a Python script to sum the amount column in a CSV table with headers.",
        "Sum the CSV amount values from standard input, skipping the header row.",
        "Add the values under the amount heading in the input CSV.",
        "Read a headered CSV from stdin and print the sum of its amount column.",
        "Calculate the total in the CSV column named amount; ignore the header.",
        "Write a script that totals every data row's amount field in the CSV input.",
        "Print the sum of CSV amounts supplied on standard input.",
    ],
    "json_max": [
        "Read a JSON object from stdin and print the largest value in its scores list.",
        "Write a script that loads JSON input and prints max(data['scores']).",
        "Find and print the maximum score from the input JSON object's scores array.",
        "Parse the input JSON and output the highest number in scores.",
        "Print the largest entry in the scores array from the JSON on stdin.",
        "Read JSON data and find the maximum value in its scores field.",
        "Write a Python script to print max of the input object's scores list.",
        "Output the greatest score in the supplied JSON object.",
    ],
    "unique_sorted": [
        "Print unique input integers in ascending order, one number on each line.",
        "Write a script that deduplicates stdin integers and prints them sorted.",
        "Read integers from standard input and output the sorted distinct values line by line.",
        "Remove duplicate integers from stdin, sort the rest, and print them.",
        "Output each different input number once, in increasing order.",
        "Create a Python program that prints the unique numbers smallest first.",
        "Sort the distinct whitespace-separated integers and print one per line.",
        "Read the numbers, discard repeats, then print them in ascending order.",
    ],
}

CODE = {
    "count_lines": (
        "import sys;print(sum(1 for x in sys.stdin if x.strip()))",
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

# Each task gets three repair states: the established failure, a script that
# runs but returns a wrong answer, and a syntax error. Keeping the first fault
# preserves the existing evaluation fixture; the extra faults broaden only the
# training trajectories.
REPAIR_FAULTS = {
    "count_lines": [
        ("runtime_error", CODE["count_lines"][1]),
        (
            "wrong_output",
            "import sys;print(len(sys.stdin.read().splitlines()))",
        ),
        (
            "syntax_error",
            "import sys;print(sum(bool(x.strip()) for x in sys.stdin.read().splitlines())",
        ),
    ],
    "sum_numbers": [
        ("runtime_error", CODE["sum_numbers"][1]),
        (
            "wrong_output",
            "import sys;print(sum(map(int,sys.stdin.read().split()))+1)",
        ),
        (
            "syntax_error",
            "import sys;print(sum(map(int,sys.stdin.read().split()))",
        ),
    ],
    "count_words": [
        ("runtime_error", CODE["count_words"][1]),
        (
            "wrong_output",
            "import sys;print(len(sys.stdin.read().splitlines()))",
        ),
        ("syntax_error", "import sys;print(len(sys.stdin.read().split())"),
    ],
    "sort_numbers": [
        ("runtime_error", CODE["sort_numbers"][1]),
        (
            "wrong_output",
            'import sys;print(*sorted(list(map(int,sys.stdin.read().split()))+[999]),sep="\\n")',
        ),
        (
            "syntax_error",
            'import sys;print(*sorted(map(int,sys.stdin.read().split()),sep="\\n")',
        ),
    ],
    "reverse_lines": [
        ("runtime_error", CODE["reverse_lines"][1]),
        (
            "wrong_output",
            'import sys;print("\\n".join(x[::-1]+"!" for x in sys.stdin.read().splitlines()))',
        ),
        (
            "syntax_error",
            'import sys;print("\\n".join(x[::-1] for x in sys.stdin.read().splitlines())',
        ),
    ],
    "mean_numbers": [
        ("wrong_output", CODE["mean_numbers"][1]),
        (
            "wrong_output",
            "import sys;v=list(map(float,sys.stdin.read().split()));print(sum(v)/len(v)+0.5)",
        ),
        (
            "syntax_error",
            "import sys;v=list(map(float,sys.stdin.read().split()));print(sum(v)/len(v)",
        ),
    ],
    "csv_total": [
        ("runtime_error", CODE["csv_total"][1]),
        (
            "wrong_output",
            "import csv,sys;r=csv.reader(sys.stdin);next(r);print(sum(int(x[1]) for x in r)+1)",
        ),
        (
            "syntax_error",
            "import csv,sys;r=csv.reader(sys.stdin);next(r);print(sum(int(x[1]) for x in r)",
        ),
    ],
    "json_max": [
        ("runtime_error", CODE["json_max"][1]),
        (
            "wrong_output",
            'import json,sys;print(max(json.load(sys.stdin)["scores"])+1)',
        ),
        (
            "syntax_error",
            'import json,sys;print(max(json.load(sys.stdin)["scores"])',
        ),
    ],
    "unique_sorted": [
        ("runtime_error", CODE["unique_sorted"][1]),
        (
            "wrong_output",
            'import sys;print(*sorted(set(map(int,sys.stdin.read().split())))[:-1],sep="\\n")',
        ),
        (
            "syntax_error",
            'import sys;print(*sorted(set(map(int,sys.stdin.read().split())),sep="\\n")',
        ),
    ],
}
OBSERVATION_CACHE: dict[tuple[str, str], dict[str, Any]] = {}


def make_case(kind: str, index: int, rng: random.Random) -> tuple[str, str]:
    if kind == "count_lines":
        words = [
            rng.choice(["alpha", "beta", "gamma", "delta", "epsilon", "zeta"])
            for _ in range(rng.randint(1, 6))
        ]
        lines = [*words, *("" for _ in range(rng.randint(1, 3)))]
        lines.extend(" " * rng.randint(1, 3) for _ in range(rng.randint(0, 2)))
        rng.shuffle(lines)
        return "\n".join(lines), "\n"
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


def load_reserved_inputs(path: Path) -> dict[str, set[str]]:
    reserved: dict[str, set[str]] = {}
    if not path.is_file():
        return reserved
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
            family = row.get("family")
            stdin = row.get("stdin")
            if isinstance(family, str) and isinstance(stdin, str):
                reserved.setdefault(family, set()).add(stdin)
    return reserved


def build_case_bank(
    kind: str,
    count: int,
    rng: random.Random,
    reserved: set[str],
) -> list[tuple[str, str]]:
    bank: list[tuple[str, str]] = []
    seen = set(reserved)
    for case_index in range(count * 1000):
        case = make_case(kind, case_index, rng)
        if case[0] in seen:
            continue
        seen.add(case[0])
        bank.append(case)
        if len(bank) == count:
            return bank
    raise ValueError(
        f"Could only generate {len(bank)} unique cases for {kind}; "
        f"requested {count} with {len(reserved)} reserved inputs"
    )


def add_episode(
    pairs: list[dict[str, str]],
    group: str,
    task: str,
    stdin: str,
    correct_code: str,
    initial_code: str,
    repair: bool,
    repair_fault: str,
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
        pairs.append(
            {
                "prompt": prompt,
                "completion": completion,
                "group": group,
                "repair_fault": repair_fault,
            }
        )
        if "tool" in action:
            invalid_completion = completion.replace('"tool"', '"toop"', 1)
        else:
            invalid_completion = completion.replace('"final"', '"finol"', 1)
        retry_history = [
            *history,
            {"role": "assistant", "content": invalid_completion[:48]},
            {
                "role": "tool:runtime",
                "content": compact_json({"error": "invalid_action"}),
            },
        ]
        retry_prompt = format_agent_prompt(task, retry_history, prefix_bytes)
        if len(retry_prompt.encode("utf-8")) > prefix_bytes:
            raise ValueError(f"retry prompt exceeds prefix window in {group}")
        pairs.append(
            {
                "prompt": retry_prompt,
                "completion": completion,
                "group": group,
                "repair_fault": repair_fault,
            }
        )
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
            if isinstance(first_result, dict) and first_result.get("exit_code") == 0:
                cache_key = (correct_code, stdin)
                expected_result = OBSERVATION_CACHE.get(cache_key)
                if expected_result is None:
                    script_path = environment["workdir"] / "main.py"
                    script_path.write_text(correct_code, encoding="utf-8")
                    expected_result = execute_tool(
                        "run_script", {}, environment
                    )
                    OBSERVATION_CACHE[cache_key] = expected_result
                    script_path.write_text(initial_code, encoding="utf-8")
                if (
                    expected_result.get("exit_code") == 0
                    and first_result.get("stdout") != expected_result.get("stdout")
                ):
                    history.append(
                        {
                            "role": "tool:runtime",
                            "content": compact_json(
                                {"error": "expected_output_mismatch"}
                            ),
                        }
                    )
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
    cases_per_task: int,
    reserved_inputs: dict[str, set[str]],
) -> list[dict[str, str]]:
    rng = random.Random(seed)
    pairs: list[dict[str, str]] = []
    case_banks: dict[str, list[tuple[str, str]]] = {}
    for kind_index, kind in enumerate(TASKS):
        case_rng = random.Random(seed + 1009 * kind_index)
        case_banks[kind] = build_case_bank(
            kind,
            cases_per_task,
            case_rng,
            reserved_inputs.get(kind, set()),
        )

    for kind, task_templates in TASKS.items():
        for index in range(episodes_per_task):
            task = rng.choice(task_templates)
            sample_index = index % len(case_banks[kind])
            stdin, _ = case_banks[kind][sample_index]
            correct_code, _ = CODE[kind]
            repair = index % 4 == 0
            fault_index = (index // 4) % len(REPAIR_FAULTS[kind])
            repair_fault, faulty_code = REPAIR_FAULTS[kind][fault_index]
            if repair:
                task += "\nCURRENT_SCRIPT:" + json.dumps(
                    faulty_code, ensure_ascii=False
                )
            task = f"{task}\nSAMPLE_STDIN:{json.dumps(stdin, ensure_ascii=False)}"
            add_episode(
                pairs,
                f"{kind}:sample{sample_index:02d}",
                task,
                stdin,
                correct_code,
                faulty_code if repair else correct_code,
                repair,
                repair_fault if repair else "none",
                prefix_bytes,
                seq_len,
            )
    rng.shuffle(pairs)
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic Python coding-agent tool-use trajectories for NCM."
    )
    parser.add_argument(
        "--output", default="data/agent_pairs_varied_lines_v2.jsonl"
    )
    parser.add_argument("--episodes-per-task", type=int, default=256)
    parser.add_argument("--prefix-bytes", type=int, default=510)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--cases-per-task", type=int, default=64)
    parser.add_argument(
        "--exclude-manifest", default="data/agent_eval_manifest.jsonl"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if (
        args.episodes_per_task < 4
        or args.prefix_bytes < 128
        or args.seq_len < 64
        or args.cases_per_task < 4
    ):
        raise SystemExit(
            "Need at least 4 episodes and cases per task, 128 prompt bytes, and 64 output bytes"
        )

    pairs = build_dataset(
        args.episodes_per_task,
        args.seed,
        args.prefix_bytes,
        args.seq_len,
        args.cases_per_task,
        load_reserved_inputs(Path(args.exclude_manifest)),
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
