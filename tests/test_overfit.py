from glassmind.testing import run_overfit_test


def test_tiny_overfit() -> None:
    result = run_overfit_test(steps=240)
    assert result["passed"] is True

