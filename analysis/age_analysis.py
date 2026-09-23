#!/usr/bin/env python3
"""What the burned-in age explains.

`ocr_age.py` recovers a real age for 1,208 of the 2,009 primary-cohort images by
reading the console overlay. This script asks what that variable does to the
paper's claims:

  1. how well age alone separates hernia from control;
  2. whether the encoder probes still separate them once age is adjusted for,
     within the subcohort where age is known;
  3. whether the "the mask failed here" flag carries label information beyond age.

Reads the published (leaky) embeddings by default; pass --emb-dir to point it at
the remasked ones.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RES = Path("/scratch/hl106/80_workspace/foye/paper/results")
PROXY_CSV = "/scratch/hl106/foye/cohort_1960/proxy_metadata.csv"  # 1960-image cohort, see build_clean_cohort.py
MODELS = ["biomedclip", "torchxrayvision", "rad-dino", "imagenet"]
DISPLAY = {"biomedclip": "BiomedCLIP", "torchxrayvision": "TorchXRayVision",
           "rad-dino": "RAD-DINO", "imagenet": "ImageNet-DenseNet121"}
N_SPLITS, N_REPEATS, SEED, N_BOOT = 5, 5, 42, 2000


def boot_ci(y, s, n=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    out = []
    for _ in range(n):
        idx = np.concatenate([rng.choice(pos, len(pos), True),
                              rng.choice(neg, len(neg), True)])
        if len(np.unique(y[idx])) == 2:
            out.append(roc_auc_score(y[idx], s[idx]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def oof(X, y, nuisance=None):
    """Repeat-averaged out-of-fold probe score, nuisance removed within folds."""
    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                                 random_state=SEED)
    acc = np.full((N_REPEATS, len(y)), np.nan)
    for k, (tr, te) in enumerate(cv.split(X, y)):
        Xtr, Xte = X[tr], X[te]
        if nuisance is not None:
            zs = StandardScaler().fit(nuisance[tr])
            Ztr, Zte = zs.transform(nuisance[tr]), zs.transform(nuisance[te])
            xs = StandardScaler().fit(Xtr)
            Str, Ste = xs.transform(Xtr), xs.transform(Xte)
            r = Ridge(alpha=10.0).fit(Ztr, Str)
            Xtr, Xte = Str - r.predict(Ztr), Ste - r.predict(Zte)
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(C=1.0, max_iter=5000))
        clf.fit(Xtr, y[tr])
        acc[k // N_SPLITS, te] = clf.predict_proba(Xte)[:, 1]
    return np.nanmean(acc, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb-dir", default="/scratch/hl106/foye/cohort_1960/emb_remasked")
    ap.add_argument("--tag", default="published")
    args = ap.parse_args()

    meta = pd.read_csv(PROXY_CSV)
    y = meta["label"].astype(int).to_numpy()
    age = pd.read_csv(RES / "burned_in_age.csv")
    assert len(age) == len(meta)
    assert (age["label"].to_numpy() == y).all(), "age table is out of order"

    have = age["age_months"].notna().to_numpy()
    a = age["age_months"].to_numpy()
    print(f"age readable on {have.sum()} / {len(y)} images "
          f"({100*have.mean():.1f}%); hernia {100*have[y==1].mean():.1f}%, "
          f"control {100*have[y==0].mean():.1f}%")

    out = {"n": int(len(y)), "n_age": int(have.sum()), "emb_dir": args.emb_dir}

    # ---- 1. age alone ----
    ya, aa = y[have], a[have]
    auc_age = roc_auc_score(ya, -aa)
    lo, hi = boot_ci(ya, -aa)
    med_h, med_c = np.median(aa[ya == 1]), np.median(aa[ya == 0])
    print(f"\nage alone            AUC {auc_age:.3f} [{lo:.3f}, {hi:.3f}]   "
          f"median age hernia {med_h:.0f} mo, control {med_c:.0f} mo")
    out["age_alone"] = {"auc": auc_age, "ci": [lo, hi],
                        "median_months_hernia": float(med_h),
                        "median_months_control": float(med_c)}

    # ---- 2. encoders on the age subcohort, before and after adjusting for age ----
    Z = np.column_stack([aa, np.log1p(aa)])
    rows = []
    for m in MODELS:
        X = np.load(Path(args.emb_dir) / f"embeddings_{m}.npz")["features"]
        X = X.astype(np.float32)[have]
        s_raw = oof(X, ya)
        s_adj = oof(X, ya, nuisance=Z)
        r = {"model": DISPLAY[m],
             "auc_raw": roc_auc_score(ya, s_raw),
             "auc_age_adjusted": roc_auc_score(ya, s_adj)}
        r["ci_raw"] = boot_ci(ya, s_raw)
        r["ci_age_adjusted"] = boot_ci(ya, s_adj)
        r["delta"] = r["auc_raw"] - r["auc_age_adjusted"]
        rows.append(r)
        print(f"{DISPLAY[m]:22s} raw {r['auc_raw']:.3f} "
              f"[{r['ci_raw'][0]:.3f}, {r['ci_raw'][1]:.3f}]   "
              f"age-adjusted {r['auc_age_adjusted']:.3f} "
              f"[{r['ci_age_adjusted'][0]:.3f}, {r['ci_age_adjusted'][1]:.3f}]   "
              f"cost {r['delta']:+.3f}")
    out["encoders_age_subcohort"] = rows

    # ---- 3. the mask-failure flag ----
    hits = RES / "mask_failure_flag.csv"
    if hits.exists():
        f = pd.read_csv(hits)["residual_text"].to_numpy()
        auc_f = roc_auc_score(y, f.astype(float))
        print(f"\nmask-failure flag alone   AUC {auc_f:.3f}  "
              f"(hernia {100*f[y==1].mean():.1f}%, control {100*f[y==0].mean():.1f}%)")
        out["mask_failure_flag"] = {
            "auc": auc_f,
            "rate_hernia": float(f[y == 1].mean()),
            "rate_control": float(f[y == 0].mean()),
        }

    p = RES / f"age_analysis_{args.tag}.json"
    p.write_text(json.dumps(out, indent=2, default=float))
    print("\nwrote", p)


if __name__ == "__main__":
    main()
