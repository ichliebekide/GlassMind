#!/usr/bin/env python3
import argparse

from glassmind.testing import run_overfit_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tiny-Overfit-Test für GlassMind")
    parser.add_argument("--steps", type=int, default=240)
    args = parser.parse_args()
    result = run_overfit_test(steps=args.steps)
    print("[PASS] Tiny-Overfit-Test bestanden")
    print(result)

