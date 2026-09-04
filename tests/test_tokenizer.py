from glassmind.data.tokenizer import ByteTokenizer


def test_utf8_roundtrip() -> None:
    tokenizer = ByteTokenizer()
    text = "Glas, Grüße und 🧠"
    assert tokenizer.decode(tokenizer.encode(text, add_bos=True, add_eos=True)) == text

