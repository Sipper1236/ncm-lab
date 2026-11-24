from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import torch

from ncm_lab.agent_tools import (
    compact_json,
    execute_tool,
    format_agent_prompt,
    parse_action,
)
from ncm_lab.checkpoint import load_tokenizer
from ncm_lab.data import decode_bytes, encode_prefix
from ncm_lab.model import NCMGenerator
from ncm_lab.train import choose_device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the NCM coding agent with isolated Python-script tools."
    )
    parser.add_argument("--tokenizer", default="artifacts/agent/tokenizer.pt")
    parser.add_argument("--generator", default="artifacts/agent/generator.pt")
    parser.add_argument("--task", required=True)
    parser.add_argument("--stdin", default="", help="sample input used by run_script()")
    parser.add_argument("--initial-script", default="", help="existing main.py source to debug")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=16)
    return parser


def run_agent(args: argparse.Namespace) -> str | None:
    if len(args.stdin.encode("utf-8")) > 100:
        raise SystemExit("--stdin is limited to 100 UTF-8 bytes")
    if len(args.initial_script.encode("utf-8")) > 160:
        raise SystemExit("--initial-script is limited to 160 UTF-8 bytes")
    device = choose_device(args.device)
    tokenizer, _ = load_tokenizer(args.tokenizer, device)
    payload = torch.load(args.generator, map_location=device, weights_only=True)
    generator = NCMGenerator(tokenizer, **payload["generator_config"])
    generator.load_state_dict(payload["model"])
    generator.to(device)
    tokenizer.eval()
    generator.eval()

    history: list[dict[str, str]] = []
    last_run_succeeded = False
    with tempfile.TemporaryDirectory(prefix="ncm-agent-") as directory:
        environment = {"workdir": Path(directory), "stdin": args.stdin}
        task = args.task
        if args.initial_script:
            (environment["workdir"] / "main.py").write_text(
                args.initial_script, encoding="utf-8"
            )
            task += "\nCURRENT_SCRIPT:" + json.dumps(
                args.initial_script, ensure_ascii=False
            )
        task += "\nSAMPLE_STDIN:" + json.dumps(args.stdin, ensure_ascii=False)
        initial_prompt = format_agent_prompt(
            task,
            history,
            max_bytes=generator.prefix_len - 2,
        )
        if f"TASK:{task}\nHISTORY:none\nNEXT:" not in initial_prompt:
            raise SystemExit("task context exceeds the model prompt window; shorten it")
        for index in range(args.max_steps):
            prompt = format_agent_prompt(
                task,
                history,
                max_bytes=generator.prefix_len - 2,
            )
            prefix = encode_prefix(prompt, generator.prefix_len).unsqueeze(0).to(device)
            with torch.no_grad():
                latents = generator.generate_latents(
                    prefix,
                    temperature=args.temperature,
                    top_k=args.top_k,
                )
                token_ids = (
                    tokenizer.decode_latents(latents).argmax(dim=-1)[0].cpu().tolist()
                )

            raw = decode_bytes(token_ids)
            try:
                action = parse_action(raw)
            except ValueError as error:
                print(f"step {index + 1}: invalid action: {error}")
                print(f"model output: {raw!r}")
                return None

            print(f"step {index + 1} action: {compact_json(action)}")
            if "final" in action:
                if not last_run_succeeded:
                    print("final rejected: no successful run of the current script")
                    return None
                print(f"final: {action['final']}")
                return action["final"]

            result = execute_tool(action["tool"], action["args"], environment)
            print(f"step {index + 1} observation: {compact_json(result)}")
            if action["tool"] == "write_script":
                last_run_succeeded = False
            elif action["tool"] == "run_script":
                last_run_succeeded = result.get("exit_code") == 0
                if not last_run_succeeded:
                    script_path = environment["workdir"] / "main.py"
                    current_script = (
                        script_path.read_text(encoding="utf-8")
                        if script_path.is_file()
                        else ""
                    )
                    task = (
                        f"{args.task}\nCURRENT_SCRIPT:"
                        f"{json.dumps(current_script, ensure_ascii=False)}\n"
                        f"SAMPLE_STDIN:{json.dumps(args.stdin, ensure_ascii=False)}"
                    )
            history.append({"role": "assistant", "content": compact_json(action)})
            history.append(
                {
                    "role": f"tool:{action['tool']}",
                    "content": compact_json(result),
                }
            )

    print(f"stopped after {args.max_steps} decisions without a final answer")
    return None


def main() -> None:
    args = build_parser().parse_args()
    if args.max_steps < 1:
        raise SystemExit("--max-steps must be positive")
    if args.temperature < 0:
        raise SystemExit("--temperature must be nonnegative")
    run_agent(args)


if __name__ == "__main__":
    main()
