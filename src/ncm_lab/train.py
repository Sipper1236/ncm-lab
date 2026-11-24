from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import Iterator

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from ncm_lab.checkpoint import load_tokenizer
from ncm_lab.data import PAD_ID, VOCAB_SIZE, load_pair_datasets
from ncm_lab.model import (
    NCMGenerator,
    TextVQTokenizer,
    resize_sequence,
    resize_valid_weights,
)


def choose_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        try:
            import torchvulkan
        except ImportError:
            torchvulkan = None
        if torchvulkan is not None and torchvulkan.is_available():
            torchvulkan.set_device(0)
            return torch.device("vulkan")
        return torch.device("cpu")
    if name == "vulkan" or name.startswith("vulkan:"):
        try:
            import torchvulkan
        except ImportError as error:
            raise RuntimeError(
                "Vulkan support requires `uv sync --extra vulkan`"
            ) from error
        if not torchvulkan.is_available():
            raise RuntimeError("torchvulkan did not find a supported Vulkan device")
        try:
            index = int(name.split(":", 1)[1]) if ":" in name else 0
        except ValueError as error:
            raise RuntimeError("Vulkan device must be `vulkan` or `vulkan:<index>`") from error
        if index < 0 or index >= torchvulkan.device_count():
            raise RuntimeError(f"Vulkan device index {index} is not available")
        torchvulkan.set_device(index)
        return torch.device("vulkan")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available to PyTorch")
    return device


def repeat_batches(
    loader: DataLoader,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    while True:
        yield from loader


def parse_levels(value: str) -> list[int]:
    try:
        levels = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("Levels must be comma-separated integers") from error
    if not levels:
        raise argparse.ArgumentTypeError("Provide comma-separated level lengths")
    return levels


def save_tokenizer(
    path: Path, model: TextVQTokenizer, config: dict[str, int | list[int]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": config}, path)


def validation_loader(data, args: argparse.Namespace) -> DataLoader:
    count = min(args.eval_pairs, len(data))
    return DataLoader(
        Subset(data, range(count)), batch_size=args.batch_size, shuffle=False
    )


@torch.no_grad()
def evaluate_tokenizer(
    model: TextVQTokenizer,
    data,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    reconstruction_sum = 0.0
    vq_sum = 0.0
    batches = 0
    for _, targets in validation_loader(data, args):
        result = model(targets.to(device), cutoff=len(model.levels))
        reconstruction_sum += result["reconstruction_loss"].item()
        vq_sum += result["vq_loss"].item()
        batches += 1
    return reconstruction_sum / batches, vq_sum / batches


@torch.no_grad()
def evaluate_generator(
    model: NCMGenerator,
    tokenizer: TextVQTokenizer,
    data,
    args: argparse.Namespace,
    device: torch.device,
) -> list[float]:
    model.eval()
    loss_sums = [0.0 for _ in tokenizer.levels]
    for prefixes, targets in validation_loader(data, args):
        prefixes = prefixes.to(device)
        targets = targets.to(device)
        codes = tokenizer.encode_codes(targets)
        predictions = model(codes, prefixes)
        valid_tokens = targets.ne(PAD_ID).float().unsqueeze(-1)
        for level, (length, logits, target_codes) in enumerate(
            zip(tokenizer.levels, predictions, codes, strict=True)
        ):
            valid = resize_valid_weights(valid_tokens, length).squeeze(-1).gt(0.0)
            per_code = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                target_codes.reshape(-1),
                reduction="none",
            ).view_as(target_codes)
            loss_sums[level] += (
                (per_code * valid).sum() / valid.sum().clamp_min(1)
            ).item()
    batches = len(validation_loader(data, args))
    return [loss / batches for loss in loss_sums]


def train_tokenizer(args: argparse.Namespace, device: torch.device) -> None:
    train_data, validation_data = load_pair_datasets(
        args.data, args.prefix_len, args.seq_len, seed=args.seed
    )
    print(
        f"device={device} train_pairs={len(train_data)} "
        f"validation_pairs={len(validation_data)}"
    )
    config = {
        "seq_len": args.seq_len,
        "levels": args.levels,
        "latent_dim": args.latent_dim,
        "codebook_size": args.codebook_size,
        "vocab_size": VOCAB_SIZE,
        "decoder_layers": args.tokenizer_layers,
        "heads": args.heads,
    }
    tokenizer = TextVQTokenizer(**config).to(device)
    optimizer = torch.optim.AdamW(
        tokenizer.parameters(), lr=args.learning_rate, weight_decay=0.01
    )
    loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, drop_last=False
    )
    tokenizer.train()
    for step, (_, targets) in enumerate(repeat_batches(loader), start=1):
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        result = tokenizer(targets)
        loss = result["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(tokenizer.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            print(
                f"[tokenizer] step={step}/{args.steps} "
                f"loss={loss.item():.4f} "
                f"recon={result['reconstruction_loss'].item():.4f} "
                f"vq={result['vq_loss'].item():.4f}"
            )
        if step >= args.steps:
            break

    val_recon, val_vq = evaluate_tokenizer(tokenizer, validation_data, args, device)
    print(
        f"[tokenizer] validation pairs={min(args.eval_pairs, len(validation_data))} "
        f"recon={val_recon:.4f} vq={val_vq:.4f} total={val_recon + val_vq:.4f}"
    )
    save_tokenizer(Path(args.output) / "tokenizer.pt", tokenizer.cpu(), config)
    print(f"saved tokenizer: {Path(args.output) / 'tokenizer.pt'}")


def train_generator(args: argparse.Namespace, device: torch.device) -> None:
    tokenizer, tokenizer_config = load_tokenizer(args.tokenizer, device)
    tokenizer.eval()
    for parameter in tokenizer.parameters():
        parameter.requires_grad_(False)

    generator_config = {
        "prefix_len": args.prefix_len,
        "generator_width": args.generator_width,
        "generator_layers": args.generator_layers,
        "heads": args.heads,
        "dropout": args.dropout,
    }
    train_data, validation_data = load_pair_datasets(
        args.data, args.prefix_len, tokenizer.seq_len, seed=args.seed
    )
    print(
        f"device={device} train_pairs={len(train_data)} "
        f"validation_pairs={len(validation_data)}"
    )
    generator = NCMGenerator(tokenizer, **generator_config).to(device)
    if args.resume_generator:
        payload = torch.load(
            args.resume_generator, map_location=device, weights_only=True
        )
        if payload.get("generator_config") != generator_config:
            raise ValueError("resume checkpoint generator configuration does not match")
        generator.load_state_dict(payload["model"])
        print(f"resumed generator weights: {args.resume_generator}")
    optimizer = torch.optim.AdamW(
        [parameter for parameter in generator.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=0.01,
    )
    loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, drop_last=False
    )
    generator.train()

    for step, (prefixes, targets) in enumerate(repeat_batches(loader), start=1):
        prefixes = prefixes.to(device)
        targets = targets.to(device)
        with torch.no_grad():
            codes = tokenizer.encode_codes(targets)
        optimizer.zero_grad(set_to_none=True)
        predictions = generator(codes, prefixes, corruption=args.corruption)

        valid_tokens = targets.ne(PAD_ID).float().unsqueeze(-1)
        level_losses: list[torch.Tensor] = []
        for length, logits, target_codes in zip(
            tokenizer.levels, predictions, codes, strict=True
        ):
            valid = resize_valid_weights(valid_tokens, length).squeeze(-1).gt(0.0)
            per_code = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                target_codes.reshape(-1),
                reduction="none",
            ).view_as(target_codes)
            level_loss = (per_code * valid).sum() / valid.sum().clamp_min(1)
            level_losses.append(level_loss)
        loss = torch.stack(level_losses).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in generator.parameters() if parameter.requires_grad],
            1.0,
        )
        optimizer.step()

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            print(
                f"[generator] step={step}/{args.steps} "
                f"loss={loss.item():.4f} "
                f"levels={[round(item.item(), 3) for item in level_losses]}"
            )
        if step >= args.steps:
            break

    val_levels = evaluate_generator(
        generator, tokenizer, validation_data, args, device
    )
    print(
        f"[generator] validation pairs={min(args.eval_pairs, len(validation_data))} "
        f"loss={sum(val_levels) / len(val_levels):.4f} levels="
        f"{[round(item, 3) for item in val_levels]}"
    )
    path = Path(args.output) / "generator.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": generator.cpu().state_dict(),
            "tokenizer_config": tokenizer_config,
            "generator_config": generator_config,
        },
        path,
    )
    print(f"saved generator: {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the tokenizer or NCM generator from JSONL pairs."
    )
    parser.add_argument("--stage", choices=("tokenizer", "generator"), required=True)
    parser.add_argument("--data", default="data/pairs.jsonl")
    parser.add_argument("--output", default="artifacts")
    parser.add_argument("--tokenizer", default="artifacts/tokenizer.pt")
    parser.add_argument("--resume-generator", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-pairs", type=int, default=32)
    parser.add_argument("--prefix-len", type=int, default=256)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument(
        "--levels",
        type=parse_levels,
        default=parse_levels("1,2,4,8,16,32,64,128,192,256"),
    )
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--codebook-size", type=int, default=64)
    parser.add_argument("--tokenizer-layers", type=int, default=1)
    parser.add_argument("--generator-width", type=int, default=64)
    parser.add_argument("--generator-layers", type=int, default=1)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--corruption",
        type=float,
        default=0.0,
        help="mean code corruption rate; linearly decreases from coarse to fine levels",
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.steps < 1:
        raise SystemExit("--steps must be positive")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
    if args.eval_pairs < 1:
        raise SystemExit("--eval-pairs must be positive")
    if not 0.0 <= args.corruption <= 1.0:
        raise SystemExit("--corruption must be between 0 and 1")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    device = choose_device(args.device)
    if device.type == "vulkan":
        import torchvulkan

        torchvulkan.manual_seed(args.seed)
        print(f"vulkan_device={torchvulkan.get_device_name()}")
    if args.stage == "tokenizer":
        if args.levels[-1] != args.seq_len:
            raise SystemExit("The final --levels value must equal --seq-len")
        train_tokenizer(args, device)
    else:
        train_generator(args, device)


if __name__ == "__main__":
    main()
