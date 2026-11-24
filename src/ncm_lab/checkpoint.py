from __future__ import annotations

from pathlib import Path

import torch

from ncm_lab.model import TextVQTokenizer


def load_tokenizer(
    path: str | Path, device: str | torch.device = "cpu"
) -> tuple[TextVQTokenizer, dict]:
    payload = torch.load(path, map_location=device, weights_only=True)
    config = payload["config"]
    tokenizer = TextVQTokenizer(**config)
    tokenizer.load_state_dict(payload["model"])
    tokenizer.to(device)
    return tokenizer, config
