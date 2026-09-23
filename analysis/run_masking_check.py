#!/usr/bin/env python3
"""
Burned-in-annotation control.

An independent 609-image cohort from the same institution was embedded twice:
once as delivered, and once after the burned-in text/metadata corners were
masked out. If the apparent hernia signal came from burned-in patient or
study text, masking should remove it.

Same probe protocol as the main analysis (repeated stratified 5x5 CV,
logistic regression on standardised embeddings).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

UNMASKED = Path("/scratch/hl106/foye/results")
MASKED = Path("/scratch/hl106/foye/results_masked")
OUT = Path("/scratch/hl106/80_workspace/foye/paper/results")

MODEL_DISPLAY = {
    "biomedclip": "BiomedCLIP",
    "torchxrayvision": "TorchXRayVision",
    "rad-dino": "RAD-DINO",
    "imagenet": "ImageNet-DenseNet121",
}
N_SPLITS, N_REPEATS, SEED, N_BOOT = 5, 5, 42, 2000


def boot_ci(y, s, seed=SEED):
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    v = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = np.concatenate(
            [rng.choice(pos, pos.size, replace=True), rng.choice(neg, neg.size, replace=True)]
        )
        v[b] = roc_auc_score(y[idx], s[idx])
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def probe(X, y):
    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    aucs = []
    oof = np.full((N_REPEATS, len(y)), np.nan)
    for i, (tr, te) in enumerate(rskf.split(X, y)):
        sc = StandardScaler()
        Xtr, Xte = sc.fit_transform(X[tr]), sc.transform(X[te])
        clf = LogisticRegression(max_iter=5000, C=1.0, random_state=SEED).fit(Xtr, y[tr])
        s = clf.predict_proba(Xte)[:, 1]
        oof[i // N_SPLITS, te] = s
        aucs.append(roc_auc_score(y[te], s))
    lo, hi = boot_ci(y, oof[0])
    return dict(
        auc_mean=float(np.mean(aucs)),
        auc_sd=float(np.std(aucs, ddof=1)),
        auc_oof=float(np.mean([roc_auc_score(y, oof[r]) for r in range(N_REPEATS)])),
        auc_ci_lo=lo,
        auc_ci_hi=hi,
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for cond, root in (("unmasked", UNMASKED), ("annotation-masked", MASKED)):
        for mk, disp in MODEL_DISPLAY.items():
            f = root / f"embeddings_{mk}.npz"
            if not f.exists():
                print(f"[skip] {cond} {mk}: missing")
                continue
            z = np.load(f)
            X, y = z["features"].astype(np.float32), z["labels"].astype(int)
            r = probe(X, y)
            print(f"[{cond}] {disp}: n={len(y)} AUC={r['auc_mean']:.3f}", flush=True)
            rows.append(dict(condition=cond, model=mk, model_display=disp, n=int(len(y)),
                             n_hernia=int((y == 1).sum()), **r))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "masking_check.csv", index=False)
    print(df.to_string(index=False))
    with open(OUT / "masking_check_manifest.json", "w") as f:
        json.dump({"unmasked_dir": str(UNMASKED), "masked_dir": str(MASKED),
                   "note": "Independent 609-image cohort (all 2025) from the same institution; "
                           "the masked build lost 38 images during preprocessing, so n differs.",
                   "cv": f"RepeatedStratifiedKFold({N_SPLITS}x{N_REPEATS}), seed={SEED}"}, f, indent=2)


if __name__ == "__main__":
    main()
