from glassmind.testing import run_smoke_test


def test_numerical_smoke() -> None:
    result = run_smoke_test()
    assert result["finite"] is True

