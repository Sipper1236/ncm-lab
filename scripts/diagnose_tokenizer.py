from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

from ncm_lab.checkpoint import load_tokenizer
from ncm_lab.data import decode_bytes, load_pair_datasets
from ncm_lab.model import resize_sequence
from ncm_lab.train import choose_device


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def char_class(value: int | None) -> str:
    if value is None:
        return "end"
    char = chr(value)
    if char.isspace():
        return "whitespace"
    if char.isalpha():
        return "letter"
    if char.isdigit():
        return "digit"
    if char in "{}[]()\\\"'`:;,.":
        return "punctuation"
    return "other"


def add_reconstruction(
    stats: dict[str, Any], expected: str, actual: str
) -> None:
    expected_bytes = expected.encode("utf-8")
    actual_bytes = actual.encode("utf-8")
    width = max(len(expected_bytes), len(actual_bytes))
    matches = sum(
        left == right
        for left, right in zip(expected_bytes, actual_bytes, strict=False)
    )
    stats["examples"] += 1
    stats["exact"] += int(expected_bytes == actual_bytes)
    stats["expected_bytes"] += len(expected_bytes)
    stats["aligned_matches"] += matches
    stats["comparison_bytes"] += width
    stats["length_mismatches"] += int(len(expected_bytes) != len(actual_bytes))
    for index in range(min(len(expected_bytes), len(actual_bytes))):
        if expected_bytes[index] != actual_bytes[index]:
            stats["mismatches_by_expected_class"][char_class(expected_bytes[index])] += 1
    stats["mismatches_by_expected_class"]["missing"] += max(
        0, len(expected_bytes) - len(actual_bytes)
    )
    stats["mismatches_by_expected_class"]["extra"] += max(
        0, len(actual_bytes) - len(expected_bytes)
    )


def empty_level_stats() -> dict[str, Any]:
    return {
        "examples": 0,
        "exact": 0,
        "expected_bytes": 0,
        "aligned_matches": 0,
        "comparison_bytes": 0,
        "length_mismatches": 0,
        "mismatches_by_expected_class": Counter(),
    }


def finish_level_stats(stats: dict[str, Any]) -> dict[str, Any]:
    return {
        "examples": stats["examples"],
        "exact_reconstructions": stats["exact"],
        "exact_rate": round(stats["exact"] / max(1, stats["examples"]), 4),
        "expected_bytes": stats["expected_bytes"],
        "aligned_byte_accuracy": round(
            stats["aligned_matches"] / max(1, stats["comparison_bytes"]), 4
        ),
        "length_mismatches": stats["length_mismatches"],
        "mismatches_by_expected_class": dict(stats["mismatches_by_expected_class"]),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure exact-byte and per-level tokenizer reconstruction."
    )
    parser.add_argument("--tokenizer", default="artifacts/agent/tokenizer.pt")
    parser.add_argument("--data", default="data/agent_pairs.jsonl")
    parser.add_argument("--output", default="artifacts/eval/tokenizer-diagnostics.json")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--prefix-len", type=int, default=512)
    parser.add_argument("--max-pairs", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.max_pairs < 1 or args.batch_size < 1:
        raise SystemExit("--max-pairs and --batch-size must be positive")
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    tokenizer_path = Path(args.tokenizer)
    data_path = Path(args.data)
    tokenizer, config = load_tokenizer(tokenizer_path, device)
    tokenizer.eval()

    _, validation = load_pair_datasets(
        data_path,
        args.prefix_len,
        tokenizer.seq_len,
        seed=args.seed,
    )
    sample_count = min(args.max_pairs, len(validation))
    sample_indices = (
        [
            round(index * (len(validation) - 1) / (sample_count - 1))
            for index in range(sample_count)
        ]
        if sample_count > 1
        else [0]
    )
    loader = DataLoader(
        Subset(validation, sample_indices),
        batch_size=args.batch_size,
        shuffle=False,
    )
    per_level = [empty_level_stats() for _ in tokenizer.levels]
    code_counts = [torch.zeros(book.num_embeddings, dtype=torch.long) for book in tokenizer.codebooks]
    examples: list[dict[str, str]] = []

    with torch.no_grad():
        for _, targets in loader:
            targets = targets.to(device)
            codes = tokenizer.encode_codes(targets)
            for level, ids in enumerate(codes):
                code_counts[level] += torch.bincount(
                    ids.detach().cpu().reshape(-1),
                    minlength=tokenizer.codebooks[level].num_embeddings,
                )

            latent = torch.zeros(
                targets.shape[0], tokenizer.seq_len, tokenizer.latent_dim, device=device
            )
            for level, ids in enumerate(codes):
                latent += resize_sequence(tokenizer.codebooks[level](ids), tokenizer.seq_len)
                decoded = tokenizer.decode_latents(latent).argmax(dim=-1).cpu().tolist()
                for row_index, ids_row in enumerate(decoded):
                    expected = decode_bytes(targets[row_index].detach().cpu().tolist())
                    actual = decode_bytes(ids_row)
                    add_reconstruction(per_level[level], expected, actual)
                    if level == len(codes) - 1 and len(examples) < 12:
                        if expected != actual:
                            examples.append({"expected": expected, "actual": actual})

    codebook_usage = []
    for level, counts in enumerate(code_counts):
        probabilities = counts.double() / counts.sum().clamp_min(1)
        nonzero = probabilities[probabilities > 0]
        entropy = -(nonzero * nonzero.log2()).sum().item()
        codebook_usage.append(
            {
                "level": level,
                "positions": int(counts.sum()),
                "active_codes": int((counts > 0).sum()),
                "codebook_size": int(counts.numel()),
                "entropy_bits": round(entropy, 4),
                "perplexity": round(2**entropy, 4),
            }
        )

    selected_device = str(device)
    if device.type == "vulkan":
        import torchvulkan

        selected_device = torchvulkan.get_device_name()
    report = {
        "tokenizer_sha256": sha256(tokenizer_path),
        "data_sha256": sha256(data_path),
        "device": selected_device,
        "validation_pairs_available": len(validation),
        "pairs_evaluated": sample_count,
        "tokenizer_config": config,
        "levels": [
            {"level": level, **finish_level_stats(stats)}
            for level, stats in enumerate(per_level)
        ],
        "codebook_usage": codebook_usage,
        "full_depth_mismatch_examples": examples,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"saved diagnostics: {output_path}")


if __name__ == "__main__":
    main()
