from __future__ import annotations

import argparse

import torch

from ncm_lab.checkpoint import load_tokenizer
from ncm_lab.data import decode_bytes, encode_prefix
from ncm_lab.model import NCMGenerator
from ncm_lab.train import choose_device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate Python from an instruction.")
    parser.add_argument("--tokenizer", default="artifacts/tokenizer.pt")
    parser.add_argument("--generator", default="artifacts/generator.pt")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-samples", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=16)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.num_samples < 1:
        raise SystemExit("--num-samples must be positive")
    device = choose_device(args.device)
    tokenizer, _ = load_tokenizer(args.tokenizer, device)
    payload = torch.load(args.generator, map_location=device, weights_only=True)
    generator = NCMGenerator(tokenizer, **payload["generator_config"])
    generator.load_state_dict(payload["model"])
    generator.to(device)
    tokenizer.eval()
    generator.eval()
    prefix = encode_prefix(args.prompt, generator.prefix_len).unsqueeze(0)
    prefix = prefix.repeat(args.num_samples, 1)

    with torch.no_grad():
        latents = generator.generate_latents(
            prefix.to(device),
            temperature=args.temperature,
            top_k=args.top_k,
        )
        token_ids = tokenizer.decode_latents(latents).argmax(dim=-1).cpu().tolist()
    for index, ids in enumerate(token_ids, start=1):
        print(f"--- sample {index} ---")
        print(decode_bytes(ids))
        print()


if __name__ == "__main__":
    main()
