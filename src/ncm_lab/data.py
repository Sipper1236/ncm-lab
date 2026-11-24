from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

PAD_ID = 256
BOS_ID = 257
EOS_ID = 258
VOCAB_SIZE = 259


def encode_prefix(text: str, prefix_len: int) -> torch.Tensor:
    if prefix_len < 4:
        raise ValueError("prefix_len must be at least 4")
    payload = list(text.encode("utf-8")[: prefix_len - 2])
    ids = [BOS_ID, *payload, EOS_ID]
    ids.extend([PAD_ID] * (prefix_len - len(ids)))
    return torch.tensor(ids, dtype=torch.long)


def encode_completion(text: str, seq_len: int) -> torch.Tensor:
    if seq_len < 8:
        raise ValueError("seq_len must be at least 8")
    payload = list(text.encode("utf-8")[: seq_len - 1])
    ids = [*payload, EOS_ID]
    ids.extend([PAD_ID] * (seq_len - len(ids)))
    return torch.tensor(ids, dtype=torch.long)


class PromptCompletionDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self, records: list[dict[str, str]], prefix_len: int, seq_len: int
    ) -> None:
        self.examples = [
            (encode_prefix(record["prompt"], prefix_len),
             encode_completion(record["completion"], seq_len))
            for record in records
        ]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.examples[index]


def load_pair_datasets(
    path: str | Path,
    prefix_len: int,
    seq_len: int,
    validation_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[PromptCompletionDataset, PromptCompletionDataset]:
    records: list[dict[str, str]] = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on line {line_number}: {error}") from error
            if not isinstance(item, dict):
                raise ValueError(f"Line {line_number} must contain a JSON object")
            prompt = item.get("prompt")
            completion = item.get("completion")
            if not isinstance(prompt, str) or not isinstance(completion, str):
                raise ValueError(
                    f"Line {line_number} needs string 'prompt' and 'completion' fields"
                )
            if not completion.strip():
                continue
            record = {"prompt": prompt, "completion": completion}
            group = item.get("group")
            if isinstance(group, str) and group:
                record["group"] = group
            records.append(record)

    if not records:
        raise ValueError(
            f"No prompt/completion pairs found in {path}; expected one JSON object per line"
        )

    rng = random.Random(seed)
    if any("group" in record for record in records):
        groups: dict[str, list[dict[str, str]]] = {}
        for index, record in enumerate(records):
            group = record.get("group", f"ungrouped:{index}")
            groups.setdefault(group, []).append(record)
        group_ids = list(groups)
        rng.shuffle(group_ids)
        if len(group_ids) == 1:
            training_records = validation_records = records
        else:
            n_validation_groups = max(1, round(len(group_ids) * validation_fraction))
            n_validation_groups = min(n_validation_groups, len(group_ids) - 1)
            validation_groups = set(group_ids[:n_validation_groups])
            validation_records = [
                record
                for group, items in groups.items()
                if group in validation_groups
                for record in items
            ]
            training_records = [
                record
                for group, items in groups.items()
                if group not in validation_groups
                for record in items
            ]
    elif len(records) == 1:
        training_records = validation_records = records
    else:
        rng.shuffle(records)
        n_validation = max(1, round(len(records) * validation_fraction))
        n_validation = min(n_validation, len(records) - 1)
        validation_records = records[:n_validation]
        training_records = records[n_validation:]

    training = PromptCompletionDataset(training_records, prefix_len, seq_len)
    validation = PromptCompletionDataset(validation_records, prefix_len, seq_len)
    return training, validation


def decode_bytes(ids: list[int]) -> str:
    payload: list[int] = []
    for token in ids:
        if token == EOS_ID:
            break
        if 0 <= token < 256:
            payload.append(token)
    return bytes(payload).decode("utf-8", errors="replace")
