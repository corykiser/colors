"""N raters for a paired comparison: python scripts/power_stub.py --sd_diff 0.8 --delta 0.3 [--alpha 0.05 --power 0.8]"""
import argparse
from scipy.stats import norm


def n_paired(sd_diff, delta, alpha=0.05, power=0.8):
    z = norm.ppf(1 - alpha / 2) + norm.ppf(power)
    return int((z * sd_diff / delta) ** 2 + 0.5) + 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--sd_diff", type=float, required=True); ap.add_argument("--delta", type=float, required=True)
    ap.add_argument("--alpha", type=float, default=0.05); ap.add_argument("--power", type=float, default=0.8)
    a = ap.parse_args(); print("N raters ≈", n_paired(a.sd_diff, a.delta, a.alpha, a.power))
