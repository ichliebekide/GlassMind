from __future__ import annotations

import random

import torch
from torch import Tensor
from torch.utils.data import Dataset


class SyntheticSequenceDataset(Dataset[tuple[Tensor, Tensor]]):
    """Deterministische wiederholte Muster für kausale Stage-0/1-Tests."""

    def __init__(self, *, samples: int = 256, sequence_length: int = 32, vocab_size: int = 32, seed: int = 17) -> None:
        if vocab_size < 8:
            raise ValueError("vocab_size muss mindestens 8 sein")
        generator = random.Random(seed)
        self.samples: list[tuple[Tensor, Tensor]] = []
        for _ in range(samples):
            motif_length = generator.randint(2, min(6, sequence_length // 2))
            motif = [generator.randrange(4, vocab_size) for _ in range(motif_length)]
            tokens = (motif * ((sequence_length + 1 + motif_length - 1) // motif_length))[: sequence_length + 1]
            sequence = torch.tensor(tokens, dtype=torch.long)
            self.samples.append((sequence[:-1], sequence[1:]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return self.samples[index]


def make_overfit_batch(vocab_size: int = 32, sequence_length: int = 24, batch_size: int = 8) -> tuple[Tensor, Tensor]:
    dataset = SyntheticSequenceDataset(samples=batch_size, sequence_length=sequence_length, vocab_size=vocab_size, seed=3)
    inputs = torch.stack([dataset[index][0] for index in range(batch_size)])
    targets = torch.stack([dataset[index][1] for index in range(batch_size)])
    return inputs, targets

