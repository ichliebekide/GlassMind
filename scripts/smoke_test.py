#!/usr/bin/env python3
from glassmind.testing import run_smoke_test


if __name__ == "__main__":
    result = run_smoke_test()
    print("[PASS] Smoke-Test bestanden")
    print(result)

