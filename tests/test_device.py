from glassmind.utils.device import detect_device


def test_cpu_fallback() -> None:
    capabilities = detect_device("cpu")
    assert capabilities.backend == "cpu"
    assert capabilities.precision == "float32"

