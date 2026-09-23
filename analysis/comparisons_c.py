#!/usr/bin/env python3
"""Paired comparisons for the 'signal present, origin unresolved' framing.

Reuses the out-of-fold scores already saved by run_paper_analysis.py, so no
model is refitted. Each comparison is a paired stratified bootstrap on the same
children.
"""

import numpy as np
import pandas as pd

RES = "/scratch/hl106/80_workspace/foye/paper/results"
N_BOOT, SEED = 2000, 42
ENC = ["biomedclip", "torchxrayvision", "rad-dino", "imagenet"]
DISP = {"biomedclip": "BiomedCLIP", "torchxrayvision": "TorchXRayVision",
        "rad-dino": "RAD-DINO", "imagenet": "ImageNet-DenseNet121"}

z = np.load(f"{RES}/oof_scores.npz")
y = z["y"]


def bag(model, ns):
    return np.nanmean(z[f"{model}|{ns}"], axis=0)


def auc(s):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y, s)


def boot_diff(sa, sb, seed=SEED):
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    v = np.empty(N_BOOT)
    for b in range(N_BOOT):
        i = np.concatenate([rng.choice(pos, pos.size, True), rng.choice(neg, neg.size, True)])
        v[b] = roc_auc_score(y[i], sa[i]) - roc_auc_score(y[i], sb[i])
    lo, hi = np.percentile(v, [2.5, 97.5])
    p = 2 * min((v <= 0).mean(), (v >= 0).mean())
    return float(lo), float(hi), float(min(p, 1.0))


rows = []


def add(label, sa, sb, seed=SEED):
    d = auc(sa) - auc(sb)
    lo, hi, p = boot_diff(sa, sb, seed)
    rows.append(dict(comparison=label, delta_auc=d, ci_lo=lo, ci_hi=hi, p_bootstrap=p))


# 1. what each adjustment layer costs
for i, m in enumerate(ENC):
    add(f"{DISP[m]}: raw - demographic-adjusted", bag(m, "none"), bag(m, "S"), SEED + i)
for i, m in enumerate(ENC):
    add(f"{DISP[m]}: raw - acquisition-adjusted", bag(m, "none"), bag(m, "A"), SEED + 10 + i)

# 2. does the image model beat an acquisition-only model that never sees pixels?
px = bag("proxy_acquisition", "none")
for i, m in enumerate(ENC):
    add(f"{DISP[m]} (acquisition-adjusted) - acquisition-only proxy",
        bag(m, "A"), px, SEED + 20 + i)

# 3. the full 29-variable proxy model, reported for completeness
px29 = bag("proxy_all_proxy_factors", "none")
for i, m in enumerate(ENC):
    add(f"29-variable proxy - {DISP[m]} (raw)", px29, bag(m, "none"), SEED + 30 + i)

out = pd.DataFrame(rows)
out.to_csv(f"{RES}/comparisons_c.csv", index=False)
pd.set_option("display.width", 200)
print(out.to_string(index=False))
print("\n->", f"{RES}/comparisons_c.csv")
