"""N raters for a within-subject method contrast, accounting for rater, context and sampled-item variance.

Model: rating = method + rater + context + item(sample) + noise. The contrast between two methods, averaged
over the same C contexts × S samples per method, has variance
    Var(Δ̂) = 2σ_item² /(C·S) + 2σ_resid² /(N·C·S) + σ_rater×method² / N
(context effects cancel in a shared-context design). We solve for N given the other terms.
Usage: python scripts/power_stub.py --delta 0.3 --sd_item 0.5 --sd_resid 1.2 --sd_rater_method 0.4 --contexts 26 --samples 2
The rater-level-only formula from the pilot (sd_diff) is kept as --sd_diff.
"""
import argparse
from scipy.stats import norm


def n_raters(delta, sd_item, sd_resid, sd_rater_method, C, S, alpha=0.05, power=0.8):
    z = norm.ppf(1 - alpha / 2) + norm.ppf(power)
    fixed = 2 * sd_item**2 / (C * S)                # does not shrink with N: more items, not more raters
    target_var = (delta / z) ** 2
    if target_var <= fixed:
        return None, fixed
    per_n = 2 * sd_resid**2 / (C * S) + sd_rater_method**2
    return int(per_n / (target_var - fixed) + 0.999), fixed


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--delta", type=float, required=True); ap.add_argument("--alpha", type=float, default=0.05); ap.add_argument("--power", type=float, default=0.8)
    ap.add_argument("--sd_diff", type=float, default=None, help="pilot rater-level SD of the method difference (simple paired formula)")
    ap.add_argument("--sd_item", type=float, default=0.5); ap.add_argument("--sd_resid", type=float, default=1.2); ap.add_argument("--sd_rater_method", type=float, default=0.4)
    ap.add_argument("--contexts", type=int, default=26); ap.add_argument("--samples", type=int, default=2)
    a = ap.parse_args()
    z = norm.ppf(1 - a.alpha / 2) + norm.ppf(a.power)
    if a.sd_diff is not None:
        print("simple paired formula: N ≈", int((z * a.sd_diff / a.delta) ** 2 + 0.5) + 1)
    n, fixed = n_raters(a.delta, a.sd_item, a.sd_resid, a.sd_rater_method, a.contexts, a.samples, a.alpha, a.power)
    if n is None:
        print(f"unreachable: item variance alone ({fixed:.4f}) exceeds the target variance; add contexts/samples, not raters")
    else:
        print(f"N raters ≈ {n}  (item-variance floor {fixed:.4f} of target {(a.delta / z) ** 2:.4f})")
