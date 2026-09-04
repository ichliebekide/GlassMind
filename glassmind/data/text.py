from __future__ import annotations

import torch
from torch import Tensor
from torch.utils.data import Dataset

from glassmind.data.tokenizer import ByteTokenizer


DEFAULT_TINY_CORPUS = (
    "Mira sieht einen roten Vogel. Der Vogel sitzt auf einem alten Baum. "
    "Am Abend fliegt der rote Vogel nach Hause. "
    "Noah baut ein kleines Boot. Das Boot schwimmt auf dem ruhigen See. "
    "Am Morgen fährt Noah mit dem kleinen Boot zurück.\n"
)


class TextChunkDataset(Dataset[tuple[Tensor, Tensor]]):
    """Speicherschonende Sicht auf überlappende Tokenabschnitte."""

    def __init__(self, text: str, tokenizer: ByteTokenizer, *, sequence_length: int = 64, stride: int | None = None) -> None:
        tokens = tokenizer.encode(text, add_bos=True, add_eos=True)
        if len(tokens) <= sequence_length:
            raise ValueError("Text ist für die gewählte Sequenzlänge zu kurz")
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        self.sequence_length = sequence_length
        self.stride = stride or sequence_length
        self.starts = list(range(0, len(tokens) - sequence_length, self.stride))

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        start = self.starts[index]
        chunk = self.tokens[start : start + self.sequence_length + 1]
        return chunk[:-1], chunk[1:]

