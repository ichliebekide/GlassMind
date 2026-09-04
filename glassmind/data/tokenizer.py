from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable


@dataclass(frozen=True)
class TokenizerMetadata:
    kind: str = "byte"
    version: int = 1
    vocab_size: int = 260
    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2
    unk_id: int = 3


class ByteTokenizer:
    """Deterministischer UTF-8-Byte-Tokenizer für frühe Experimente."""

    PAD = 0
    BOS = 1
    EOS = 2
    UNK = 3
    BYTE_OFFSET = 4

    def __init__(self) -> None:
        self.metadata = TokenizerMetadata()

    @property
    def vocab_size(self) -> int:
        return self.metadata.vocab_size

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        tokens = [byte + self.BYTE_OFFSET for byte in text.encode("utf-8")]
        if add_bos:
            tokens.insert(0, self.BOS)
        if add_eos:
            tokens.append(self.EOS)
        return tokens

    def decode(self, tokens: Iterable[int], *, skip_special: bool = True) -> str:
        values: list[int] = []
        for token in tokens:
            token = int(token)
            if self.BYTE_OFFSET <= token < self.vocab_size:
                values.append(token - self.BYTE_OFFSET)
            elif not skip_special:
                values.extend(f"<{self.token_name(token)}>".encode("utf-8"))
        return bytes(values).decode("utf-8", errors="replace")

    def token_name(self, token: int) -> str:
        return {self.PAD: "PAD", self.BOS: "BOS", self.EOS: "EOS", self.UNK: "UNK"}.get(token, f"BYTE_{token - self.BYTE_OFFSET}")

    def to_dict(self) -> dict[str, object]:
        return asdict(self.metadata)

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "ByteTokenizer":
        if data.get("kind") != "byte" or int(data.get("version", -1)) != 1:
            raise ValueError("Nicht unterstützte Tokenizer-Metadaten")
        return cls()

