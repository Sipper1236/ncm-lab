from __future__ import annotations

import random

import torch
from torch import nn
from torch.nn import functional as F

from ncm_lab.data import PAD_ID


def resize_sequence(x: torch.Tensor, length: int) -> torch.Tensor:
    if x.shape[1] == length:
        return x
    return F.interpolate(
        x.transpose(1, 2), size=length, mode="linear", align_corners=False
    ).transpose(1, 2)


def resize_valid_weights(valid: torch.Tensor, length: int) -> torch.Tensor:
    """Pool token validity over each code's receptive-field bin."""
    if valid.shape[1] == length:
        return valid
    pooled = F.adaptive_avg_pool2d(
        valid.transpose(1, 2).unsqueeze(-1), (length, 1)
    )
    return pooled.squeeze(-1).transpose(1, 2)


def weighted_mse(
    prediction: torch.Tensor, target: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    per_position = (prediction - target).square().mean(dim=-1)
    return (per_position * weights).sum() / weights.sum().clamp_min(1.0)


class TextVQTokenizer(nn.Module):
    """Fixed-length byte tokenizer with a multiscale residual VQ hierarchy."""

    def __init__(
        self,
        seq_len: int,
        levels: list[int],
        latent_dim: int,
        codebook_size: int,
        vocab_size: int = 259,
        decoder_layers: int = 1,
        heads: int = 4,
        commitment_weight: float = 0.25,
    ) -> None:
        super().__init__()
        if not levels or levels[-1] != seq_len:
            raise ValueError("The last hierarchy level must equal seq_len")
        if any(a >= b for a, b in zip(levels, levels[1:])):
            raise ValueError("levels must be strictly increasing")
        if latent_dim % heads:
            raise ValueError("latent_dim must be divisible by heads")

        self.seq_len = seq_len
        self.levels = list(levels)
        self.latent_dim = latent_dim
        self.codebook_size = codebook_size
        self.vocab_size = vocab_size
        self.commitment_weight = commitment_weight

        self.token_embedding = nn.Embedding(
            vocab_size, latent_dim, padding_idx=PAD_ID
        )
        self.position_embedding = nn.Parameter(torch.empty(1, seq_len, latent_dim))
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)
        self.codebooks = nn.ModuleList(
            [nn.Embedding(codebook_size, latent_dim) for _ in levels]
        )
        for table in self.codebooks:
            nn.init.normal_(table.weight, mean=0.0, std=latent_dim**-0.5)

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=heads,
            dim_feedforward=4 * latent_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(
            decoder_layer, num_layers=decoder_layers, enable_nested_tensor=False
        )
        self.output = nn.Linear(latent_dim, vocab_size)

    def _quantize(
        self,
        vectors: torch.Tensor,
        level: int,
        valid_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        table = self.codebooks[level]
        flat = vectors.reshape(-1, self.latent_dim)
        weights = table.weight
        distances = (
            flat.square().sum(dim=1, keepdim=True)
            - 2.0 * flat @ weights.transpose(0, 1)
            + weights.square().sum(dim=1).unsqueeze(0)
        )
        ids = distances.argmin(dim=1).view(vectors.shape[:2])
        quantized = table(ids)
        straight_through = vectors + (quantized - vectors).detach()
        codebook_loss = weighted_mse(quantized, vectors.detach(), valid_weights)
        commitment_loss = weighted_mse(vectors, quantized.detach(), valid_weights)
        vq_loss = codebook_loss + self.commitment_weight * commitment_loss
        return ids, straight_through, vq_loss

    def forward(
        self, tokens: torch.Tensor, cutoff: int | None = None
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        if tokens.shape[1] != self.seq_len:
            raise ValueError(f"Expected sequences of length {self.seq_len}")
        if cutoff is None:
            cutoff = random.randint(1, len(self.levels))
        cutoff = max(1, min(cutoff, len(self.levels)))

        pad_mask = tokens.eq(PAD_ID)
        valid = (~pad_mask).float()
        z = self.token_embedding(tokens) + self.position_embedding
        z_hat = torch.zeros_like(z)
        vq_losses: list[torch.Tensor] = []
        codes: list[torch.Tensor] = []

        for level, length in enumerate(self.levels[:cutoff]):
            residual = z - z_hat.detach()
            small = resize_sequence(residual, length)
            valid_small = resize_valid_weights(valid.unsqueeze(-1), length).squeeze(-1)
            ids, quantized, vq_loss = self._quantize(small, level, valid_small)
            vq_losses.append(vq_loss)
            codes.append(ids)
            z_hat = z_hat + resize_sequence(quantized, self.seq_len)

        hidden = self.decoder(z_hat, src_key_padding_mask=pad_mask)
        logits = self.output(hidden)
        reconstruction_loss = F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            tokens.reshape(-1),
            ignore_index=PAD_ID,
        )
        vq_loss = torch.stack(vq_losses).mean()
        return {
            "loss": reconstruction_loss + vq_loss,
            "reconstruction_loss": reconstruction_loss,
            "vq_loss": vq_loss,
            "logits": logits,
            "codes": codes,
        }

    @torch.no_grad()
    def encode_codes(self, tokens: torch.Tensor) -> list[torch.Tensor]:
        z = self.token_embedding(tokens) + self.position_embedding
        z_hat = torch.zeros_like(z)
        codes: list[torch.Tensor] = []
        valid = tokens.ne(PAD_ID).float()
        for level, length in enumerate(self.levels):
            residual = z - z_hat
            small = resize_sequence(residual, length)
            valid_small = resize_valid_weights(valid.unsqueeze(-1), length).squeeze(-1)
            ids, quantized, _ = self._quantize(small, level, valid_small)
            codes.append(ids)
            z_hat = z_hat + resize_sequence(quantized, self.seq_len)
        return codes

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        hidden = self.decoder(latents)
        return self.output(hidden)


class NCMGenerator(nn.Module):
    """Condition on a text prefix, then predict each code level in parallel."""

    def __init__(
        self,
        tokenizer: TextVQTokenizer,
        prefix_len: int = 256,
        generator_width: int = 64,
        generator_layers: int = 1,
        heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if generator_width % heads:
            raise ValueError("generator_width must be divisible by heads")
        if prefix_len < 4:
            raise ValueError("prefix_len must be at least 4")
        self.seq_len = tokenizer.seq_len
        self.levels = list(tokenizer.levels)
        self.latent_dim = tokenizer.latent_dim
        self.prefix_len = prefix_len

        self.codebooks = nn.ModuleList(
            [
                nn.Embedding.from_pretrained(book.weight.detach().clone(), freeze=True)
                for book in tokenizer.codebooks
            ]
        )
        self.prefix_token_embedding = nn.Embedding.from_pretrained(
            tokenizer.token_embedding.weight.detach().clone(), freeze=True
        )
        self.input_projection = nn.Linear(self.latent_dim, generator_width)
        self.position_embedding = nn.Embedding(max(self.levels), generator_width)
        self.prefix_position_embedding = nn.Embedding(prefix_len, generator_width)
        self.level_embedding = nn.Embedding(len(self.levels), generator_width)
        layer = nn.TransformerEncoderLayer(
            d_model=generator_width,
            nhead=heads,
            dim_feedforward=4 * generator_width,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=generator_layers, enable_nested_tensor=False
        )
        self.output_heads = nn.ModuleList(
            [
                nn.Linear(generator_width, book.num_embeddings)
                for book in self.codebooks
            ]
        )
        self.register_buffer("block_mask", self._make_block_mask(), persistent=False)

    def _make_block_mask(self) -> torch.Tensor:
        total_target = sum(self.levels)
        total = self.prefix_len + total_target
        mask = torch.ones((total, total), dtype=torch.bool)
        mask[: self.prefix_len, : self.prefix_len] = False
        start = self.prefix_len
        for length in self.levels:
            mask[start : start + length, : self.prefix_len] = False
            mask[start : start + length, start : start + length] = False
            start += length
        return mask

    def _prefix_embeddings(self, prefix_tokens: torch.Tensor) -> torch.Tensor:
        if prefix_tokens.shape[1] != self.prefix_len:
            raise ValueError(f"Expected prefixes of length {self.prefix_len}")
        positions = torch.arange(self.prefix_len, device=prefix_tokens.device)
        token_vectors = self.prefix_token_embedding(prefix_tokens)
        return (
            self.input_projection(token_vectors)
            + self.prefix_position_embedding(positions).unsqueeze(0)
        )

    def _input_block(self, latents: torch.Tensor, level: int) -> torch.Tensor:
        length = self.levels[level]
        context = resize_sequence(latents, length)
        positions = torch.arange(length, device=latents.device)
        level_ids = torch.full((length,), level, device=latents.device)
        return (
            self.input_projection(context)
            + self.position_embedding(positions).unsqueeze(0)
            + self.level_embedding(level_ids).unsqueeze(0)
        )

    def forward(
        self,
        codes: list[torch.Tensor],
        prefix_tokens: torch.Tensor,
        corruption: float = 0.0,
    ) -> list[torch.Tensor]:
        if len(codes) != len(self.levels):
            raise ValueError("Code hierarchy does not match generator levels")
        if not 0.0 <= corruption <= 1.0:
            raise ValueError("corruption must be between 0 and 1")
        batch = codes[0].shape[0]
        device = codes[0].device
        z_hat = torch.zeros(batch, self.seq_len, self.latent_dim, device=device)
        prefix = self._prefix_embeddings(prefix_tokens)
        blocks: list[torch.Tensor] = []

        for level, target_ids in enumerate(codes):
            blocks.append(self._input_block(z_hat, level))
            condition_ids = target_ids
            if corruption > 0.0:
                max_rate = min(1.0, 2.0 * corruption)
                rate = (
                    max_rate * (1.0 - level / (len(self.levels) - 1))
                    if len(self.levels) > 1
                    else corruption
                )
                replace = torch.rand(target_ids.shape, device=device) < rate
                random_ids = torch.randint(
                    self.codebooks[level].num_embeddings,
                    target_ids.shape,
                    device=device,
                )
                condition_ids = torch.where(replace, random_ids, target_ids)
            contribution = self.codebooks[level](condition_ids)
            z_hat = z_hat + resize_sequence(contribution, self.seq_len)

        target_input = torch.cat(blocks, dim=1)
        hidden = self.transformer(
            torch.cat([prefix, target_input], dim=1),
            mask=self.block_mask,
            src_key_padding_mask=torch.cat(
                [
                    prefix_tokens.eq(PAD_ID),
                    torch.zeros(
                        batch, target_input.shape[1], dtype=torch.bool, device=device
                    ),
                ],
                dim=1,
            ),
        )
        predictions: list[torch.Tensor] = []
        start = self.prefix_len
        for level, length in enumerate(self.levels):
            predictions.append(
                self.output_heads[level](hidden[:, start : start + length])
            )
            start += length
        return predictions

    def predict_level(
        self,
        latents: torch.Tensor,
        prefix_tokens: torch.Tensor,
        level: int,
    ) -> torch.Tensor:
        prefix = self._prefix_embeddings(prefix_tokens)
        target = self._input_block(latents, level)
        length = target.shape[1]
        total = self.prefix_len + length
        mask = torch.ones((total, total), dtype=torch.bool, device=latents.device)
        mask[: self.prefix_len, : self.prefix_len] = False
        mask[self.prefix_len :, : self.prefix_len] = False
        mask[self.prefix_len :, self.prefix_len :] = False
        key_padding = torch.cat(
            [
                prefix_tokens.eq(PAD_ID),
                torch.zeros(
                    prefix_tokens.shape[0],
                    length,
                    dtype=torch.bool,
                    device=latents.device,
                ),
            ],
            dim=1,
        )
        hidden = self.transformer(
            torch.cat([prefix, target], dim=1),
            mask=mask,
            src_key_padding_mask=key_padding,
        )
        return self.output_heads[level](hidden[:, self.prefix_len :])

    @torch.no_grad()
    def generate_latents(
        self,
        prefix_tokens: torch.Tensor,
        temperature: float = 1.0,
        top_k: int = 0,
    ) -> torch.Tensor:
        if temperature < 0:
            raise ValueError("temperature must be nonnegative")
        if prefix_tokens.shape[1] != self.prefix_len:
            raise ValueError(f"Expected prefixes of length {self.prefix_len}")
        batch_size = prefix_tokens.shape[0]
        device = next(self.parameters()).device
        prefix_tokens = prefix_tokens.to(device)
        z_hat = torch.zeros(batch_size, self.seq_len, self.latent_dim, device=device)
        for level in range(len(self.levels)):
            logits = self.predict_level(z_hat, prefix_tokens, level)
            if temperature == 0.0:
                ids = logits.argmax(dim=-1)
            else:
                logits = logits / temperature
                if top_k > 0:
                    keep = min(top_k, logits.shape[-1])
                    values, indices = logits.topk(keep, dim=-1)
                    filtered = torch.full_like(logits, float("-inf"))
                    logits = filtered.scatter(-1, indices, values)
                probabilities = torch.softmax(logits, dim=-1)
                ids = torch.multinomial(
                    probabilities.reshape(-1, probabilities.shape[-1]), 1
                ).view(batch_size, self.levels[level])
            contribution = self.codebooks[level](ids)
            z_hat = z_hat + resize_sequence(contribution, self.seq_len)
        return z_hat
