from __future__ import annotations

import hashlib
import json
import resource
import re
import shutil
import subprocess
import tempfile
from difflib import get_close_matches
from pathlib import Path
from typing import Any

TOOL_SPEC = "write_script(code),run_script() uses SAMPLE_STDIN"
TOOL_ARGUMENTS = {
    "write_script": {"code"},
    "run_script": set(),
}
MAX_SCRIPT_BYTES = 4096
MAX_STDIN_BYTES = 4096
MAX_STDOUT_CHARS = 80
MAX_STDERR_CHARS = 100
MAX_VERIFICATION_CASES = 8


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def format_agent_prompt(
    task: str,
    history: list[dict[str, str]],
    max_bytes: int,
) -> str:
    """Serialize one next-action decision, dropping oldest complete events if needed."""
    task = task.strip()
    events = list(history)

    def render() -> str:
        history_text = "\n".join(
            f"{event['role']}:{event['content']}" for event in events
        ) or "none"
        return (
            f"TOOLS:{TOOL_SPEC}\n"
            'FORMAT:{"tool":"name","args":{...}} or {"final":"text"}\n'
            f"TASK:{task}\nHISTORY:{history_text}\nNEXT:"
        )

    while events and len(render().encode("utf-8")) > max_bytes:
        del events[:2]

    prompt = render()
    if len(prompt.encode("utf-8")) <= max_bytes:
        return prompt

    fixed = (
        f"TOOLS:{TOOL_SPEC}\n"
        'FORMAT:{"tool":"name","args":{...}} or {"final":"text"}\n'
        "TASK:\nHISTORY:none\nNEXT:"
    )
    task_budget = max(0, max_bytes - len(fixed.encode("utf-8")))
    task = task.encode("utf-8")[:task_budget].decode("utf-8", errors="ignore")
    return (
        f"TOOLS:{TOOL_SPEC}\n"
        'FORMAT:{"tool":"name","args":{...}} or {"final":"text"}\n'
        f"TASK:{task}\nHISTORY:none\nNEXT:"
    )


def make_tool_action(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {"tool": name, "args": arguments}


def make_final_action(answer: str) -> dict[str, str]:
    return {"final": answer}


def parse_action(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for offset, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[offset:])
        except json.JSONDecodeError:
            continue
        if not isinstance(value, dict):
            continue
        if isinstance(value.get("final"), str):
            return {"final": value["final"]}
        name = value.get("tool")
        arguments = value.get("args")
        if (
            isinstance(name, str)
            and name in TOOL_ARGUMENTS
            and isinstance(arguments, dict)
            and set(arguments) == TOOL_ARGUMENTS[name]
        ):
            return {"tool": name, "args": arguments}

        # NCM occasionally changes one byte in a short tool name or argument
        # key. Recover only the no-argument run tool when its JSON object is
        # otherwise unambiguous.
        if isinstance(name, str) and len(value) == 2:
            near_name = get_close_matches(name, ["run_script"], n=1, cutoff=0.75)
            other_values = [item for key, item in value.items() if key != "tool"]
            if near_name and other_values == [{}]:
                return {"tool": "run_script", "args": {}}

    # Also tolerate punctuation damage around these two fixed outer keys.
    # A final response is still gated by run_agent's successful-run check.
    near_run = re.match(
        r"\s*\{+\s*[\"']?tool[\"']?[\s:\"']*([A-Za-z_]{6,20})", text
    )
    if near_run:
        name = get_close_matches(
            near_run.group(1), ["run_script"], n=1, cutoff=0.75
        )
        if name:
            return {"tool": "run_script", "args": {}}

    near_final = re.match(r"\s*\{+\s*[\"']?final[\"']?[\s:\"']*", text)
    if near_final:
        return {"final": "Done. The script ran successfully on the sample input."}

    raise ValueError("model output did not contain a valid tool action or final answer")


def _limit_child_resources() -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (2, 3))
    resource.setrlimit(resource.RLIMIT_AS, (384 * 1024 * 1024, 384 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024, 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))


def _run_in_sandbox(workdir: Path, stdin: str) -> dict[str, Any]:
    bubblewrap = shutil.which("bwrap")
    if bubblewrap is None:
        return {"error": "sandbox_unavailable"}

    command = [
        bubblewrap,
        "--die-with-parent",
        "--unshare-all",
        "--new-session",
        "--ro-bind",
        "/usr",
        "/usr",
    ]
    for system_path in ("/lib", "/lib64"):
        if Path(system_path).exists():
            command.extend(["--ro-bind", system_path, system_path])
    command.extend(
        [
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--tmpfs",
            "/tmp",
            "--bind",
            str(workdir),
            "/work",
            "--chdir",
            "/work",
            "--clearenv",
            "--setenv",
            "PATH",
            "/usr/bin:/bin",
            "--setenv",
            "HOME",
            "/work",
            "--setenv",
            "PYTHONDONTWRITEBYTECODE",
            "1",
            "--",
            "/usr/bin/python3",
            "-I",
            "-S",
            "/work/main.py",
        ]
    )
    try:
        completed = subprocess.run(
            command,
            input=stdin,
            text=True,
            capture_output=True,
            timeout=3,
            check=False,
            preexec_fn=_limit_child_resources,
            env={"PATH": "/usr/bin:/bin"},
        )
    except subprocess.TimeoutExpired:
        return {"error": "execution_timeout"}
    except OSError as error:
        return {"error": "sandbox_failed", "detail": str(error)[:160]}

    stderr_lines = completed.stderr.strip().splitlines()
    short_stderr = stderr_lines[-1] if stderr_lines else ""
    return {
        "exit_code": completed.returncode,
        "stdout": completed.stdout[:MAX_STDOUT_CHARS],
        "stderr": short_stderr[-MAX_STDERR_CHARS:],
    }


def _run_script_snapshot(source: bytes, stdin: str) -> dict[str, Any]:
    """Run an immutable copy so a script cannot change between test inputs."""
    with tempfile.TemporaryDirectory(prefix="ncm-script-run-") as directory:
        workdir = Path(directory)
        (workdir / "main.py").write_bytes(source)
        return _run_in_sandbox(workdir, stdin)


def _validate_verification_cases(value: Any) -> list[dict[str, str]] | None:
    if not isinstance(value, list) or len(value) > MAX_VERIFICATION_CASES:
        return None
    cases: list[dict[str, str]] = []
    for case in value:
        if (
            not isinstance(case, dict)
            or set(case) != {"stdin", "expected_stdout"}
            or not isinstance(case["stdin"], str)
            or not isinstance(case["expected_stdout"], str)
            or len(case["stdin"].encode("utf-8")) > MAX_STDIN_BYTES
            or len(case["expected_stdout"].encode("utf-8")) > MAX_STDOUT_CHARS
        ):
            return None
        cases.append(case)
    return cases


def execute_tool(
    name: str, arguments: dict[str, Any], environment: dict[str, Any]
) -> dict[str, Any]:
    if name not in TOOL_ARGUMENTS or set(arguments) != TOOL_ARGUMENTS[name]:
        return {"error": "invalid_tool_arguments"}
    workdir = environment.get("workdir")
    if not isinstance(workdir, Path):
        return {"error": "coding_workspace_unavailable"}

    script_path = workdir / "main.py"
    if name == "write_script":
        code = arguments["code"]
        if not isinstance(code, str):
            return {"error": "code_must_be_text"}
        if len(code.encode("utf-8")) > MAX_SCRIPT_BYTES:
            return {"error": "script_too_large"}
        script_path.write_text(code, encoding="utf-8")
        return {"written": True, "bytes": len(code.encode("utf-8"))}

    stdin = environment.get("stdin", "")
    if not isinstance(stdin, str):
        return {"error": "stdin_must_be_text"}
    if len(stdin.encode("utf-8")) > MAX_STDIN_BYTES:
        return {"error": "stdin_too_large"}
    if not script_path.is_file():
        return {"error": "script_not_written"}
    try:
        source = script_path.read_bytes()
    except OSError:
        return {"error": "script_unreadable"}
    result = _run_script_snapshot(source, stdin)

    verification_cases = environment.get("verification_cases", [])
    if verification_cases:
        cases = _validate_verification_cases(verification_cases)
        if cases is None:
            return {"error": "invalid_verification_cases"}
        expected_stdout = environment.get("expected_stdout")
        sample_matches = (
            expected_stdout is None or result.get("stdout") == expected_stdout
        )
        if result.get("exit_code") == 0 and sample_matches:
            cache = environment.setdefault("verification_cache", {})
            if not isinstance(cache, dict):
                cache = {}
                environment["verification_cache"] = cache
            digest = hashlib.sha256(source).hexdigest()
            verification = cache.get(digest)
            if not isinstance(verification, dict):
                passed = True
                for case in cases:
                    hidden_result = _run_script_snapshot(source, case["stdin"])
                    if (
                        hidden_result.get("exit_code") != 0
                        or hidden_result.get("stdout") != case["expected_stdout"]
                    ):
                        passed = False
                verification = {"passed": passed, "case_count": len(cases)}
                cache[digest] = verification
            result["verification"] = verification
    return result
