#!/usr/bin/env python3
"""How many age-matched controls does the confirmatory cohort need?

Under the predictive framing, age is legitimate information and is not adjusted
away. The obvious reviewer objection is then that the probe is an age detector.
A control arm matched on age answers that by design instead of by regression:
if discrimination survives when the groups are the same age, it is not age.

Two questions here.
  1. How far do the existing controls already go? Greedy nearest-neighbour
     matching within a caliper says which age bands are already covered and
     which are empty.
  2. How many per group does the new cohort need? The effect to power for is the
     age-adjusted AUC measured in `age_analysis.py`, 0.538-0.585, not the
     unadjusted 0.64-0.71.

Writes `results/matched_cohort_plan.json` and `tables/table5_matching.tex`, both
of which `verify_manuscript.py` checks against the prose.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path("/scratch/hl106/80_workspace/foye/paper")
RES = BASE / "results"
TAB = BASE / "tables"
SEED = 42


def hanley_var(a, n1, n0):
    """Hanley-McNeil variance of an AUC estimate."""
    q1 = a / (2 - a)
    q2 = 2 * a * a / (1 + a)
    return (a * (1 - a) + (n1 - 1) * (q1 - a * a) + (n0 - 1) * (q2 - a * a)) / (n1 * n0)


def n_per_group(auc, power=0.80, alpha=0.05):
    """Smallest n per group at which AUC is distinguishable from 0.5."""
    from scipy.stats import norm
    za, zb = norm.ppf(1 - alpha / 2), norm.ppf(power)
    for n in range(20, 20001):
        se0 = np.sqrt((2 * n + 1) / (12.0 * n * n))       # under the null
        se1 = np.sqrt(hanley_var(auc, n, n))              # under the alternative
        if abs(auc - 0.5) >= za * se0 + zb * se1:
            return n
    return None


def greedy_match(case_age, ctrl_age, caliper):
    """Nearest-neighbour 1:1 matching without replacement, within `caliper`."""
    ctrl = np.sort(ctrl_age.copy())
    used = np.zeros(len(ctrl), bool)
    matched, unmatched = [], []
    for a in np.sort(case_age):
        d = np.where(used, np.inf, np.abs(ctrl - a))
        j = int(np.argmin(d))
        if d[j] <= caliper:
            used[j] = True
            matched.append((a, ctrl[j]))
        else:
            unmatched.append(a)
    return np.array(matched), np.array(unmatched)


BANDS = [(0, 12), (12, 24), (24, 48), (48, 72), (72, 96),
         (96, 120), (120, 156), (156, 10 ** 4)]


def band_label(lo, hi):
    if hi > 1000:
        return f"{lo//12}y+"
    if hi <= 24:
        return f"{lo}-{hi} mo"
    return f"{lo//12}-{hi//12} y"


def main():
    allrows = pd.read_csv(RES / "burned_in_age.csv")
    n_case_all = int((allrows.label == 1).sum())
    d = allrows.dropna(subset=["age_months"])
    case = d[d.label == 1].age_months.to_numpy()
    ctrl = d[d.label == 0].age_months.to_numpy()
    print(f"ages read from the overlay: {len(case)} hernia, {len(ctrl)} control")
    print(f"  hernia  median {np.median(case):5.0f} mo   IQR "
          f"{np.percentile(case,25):.0f}-{np.percentile(case,75):.0f}")
    print(f"  control median {np.median(ctrl):5.0f} mo   IQR "
          f"{np.percentile(ctrl,25):.0f}-{np.percentile(ctrl,75):.0f}")

    # ---- 1. what the existing controls already cover -------------------------
    print("\nAge distribution, and the control shortfall per band")
    print(f"{'band':>10} {'hernia':>8} {'control':>8} {'ratio':>7} {'short':>7}")
    coverage = []
    for lo, hi in BANDS:
        nh = int(((case >= lo) & (case < hi)).sum())
        nc = int(((ctrl >= lo) & (ctrl < hi)).sum())
        short = max(0, nh - nc)
        ratio = (nc / nh) if nh else np.nan
        coverage.append((band_label(lo, hi), nh, nc, ratio, short))
        print(f"{band_label(lo,hi):>10} {nh:8d} {nc:8d} "
              f"{'' if np.isnan(ratio) else f'{ratio:7.2f}'} {short:7d}")
    total_short = sum(c[4] for c in coverage)
    print(f"{'':>10} {len(case):8d} {len(ctrl):8d} {'':>7} {total_short:7d}"
          "   <- 1:1 shortfall on the readable subset")

    # ---- 2. matching yield with what is already collected --------------------
    print("\nGreedy 1:1 matching against the existing controls")
    matching = []
    for cal in (3, 6, 12):
        m, u = greedy_match(case, ctrl, cal)
        smd = ((m[:, 0].mean() - m[:, 1].mean())
               / np.sqrt((m[:, 0].var(ddof=1) + m[:, 1].var(ddof=1)) / 2)) if len(m) else np.nan
        under6 = int((m[:, 0] < 72).sum()) if len(m) else 0
        matching.append({"caliper_months": cal, "pairs": int(len(m)),
                         "pairs_under_6y": under6,
                         "unmatched_cases": int(len(u)), "abs_smd": float(abs(smd))})
        print(f"  caliper +/-{cal:2d} mo: {len(m):4d} pairs matched "
              f"({under6} with the case under 6 y), "
              f"{len(u):4d} hernia cases unmatched   |SMD| after matching {abs(smd):.3f}")

    # ---- 3. how big the new cohort has to be --------------------------------
    print("\nSample size per group, AUC vs 0.5, two-sided alpha=0.05")
    print(f"{'target AUC':>11} {'80% power':>10} {'90% power':>10}")
    power = []
    for a in (0.55, 0.56, 0.58, 0.60, 0.62, 0.65):
        n80, n90 = n_per_group(a, 0.80), n_per_group(a, 0.90)
        power.append({"auc": a, "n80": n80, "n90": n90})
        print(f"{a:11.2f} {n80:10d} {n90:10d}")

    # ---- 4. how many of those are new ---------------------------------------
    print("\nScaling the shortfall to the full cohort")
    read_rate_case = len(case) / n_case_all
    scaled = int(round(total_short / read_rate_case))
    print(f"  age was readable on {100*read_rate_case:.1f}% of hernia images, so a "
          f"shortfall of {total_short} on the readable subset")
    print(f"  corresponds to roughly {scaled} across all {n_case_all} hernia cases")

    # ---- 5. persist, so the manuscript can be verified against this ---------
    out = {
        "n_case_with_age": int(len(case)), "n_control_with_age": int(len(ctrl)),
        "median_months_hernia": float(np.median(case)),
        "median_months_control": float(np.median(ctrl)),
        "bands": [{"band": b, "hernia": nh, "control": nc,
                   "ratio": None if np.isnan(r) else float(r), "shortfall": s}
                  for b, nh, nc, r, s in coverage],
        "total_shortfall_readable": int(total_short),
        "total_shortfall_scaled": scaled,
        "matching": matching,
        "sample_size": power,
    }
    (RES / "matched_cohort_plan.json").write_text(json.dumps(out, indent=1))

    def tex_band(b):
        return "$\\ge$13 y" if b.endswith("y+") else b.replace("-", "--")

    rows = "\n".join(
        f"{tex_band(b)} & {nh} & {nc} & {'--' if np.isnan(r) else f'{r:.2f}'} & {s} \\\\"
        for b, nh, nc, r, s in coverage)
    TAB.joinpath("table5_matching.tex").write_text(
        "\\begin{tabular}{lrrrr}\n\\toprule\n"
        "Age band & Hernia & Control & Control/hernia & Control shortfall \\\\\n"
        "\\midrule\n" + rows + "\n\\midrule\n"
        f"All & {len(case)} & {len(ctrl)} & {len(ctrl)/len(case):.2f} & {total_short} \\\\\n"
        "\\bottomrule\n\\end{tabular}\n")
    print(f"\nwrote {RES/'matched_cohort_plan.json'} and {TAB/'table5_matching.tex'}")


if __name__ == "__main__":
    main()
