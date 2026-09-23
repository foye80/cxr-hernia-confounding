#!/usr/bin/env python3
"""Acquisition-matched analysis: the strongest device control available
without DICOM headers.

Residualization removes a linear projection and leaves the sample intact;
matching instead builds a subsample in which the two groups were imaged under
comparable conditions, then re-runs the probe. Matching is on the five
acquisition attributes only, so body habitus is left untouched.

1:1 nearest-neighbour matching without replacement on the Mahalanobis distance
over the four continuous acquisition attributes, exact on modality. Mahalanobis
matching is used in preference to a propensity-score caliper because the
propensity score here is dominated by modality and leaves the remaining
attributes imbalanced.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

PROXY = "/scratch/hl106/80_workspace/foye/radiographic_proxy_probe/primary_2k_radiographic_proxy_metadata.csv"
EMB = "/scratch/hl106/foye/results_2k"
OUT = "/scratch/hl106/80_workspace/foye/paper/results"

ACQ = ["modality_cr", "exam_days_since_min", "image_width_px", "image_height_px", "image_aspect"]
MODELS = {"biomedclip": "BiomedCLIP", "torchxrayvision": "TorchXRayVision",
          "rad-dino": "RAD-DINO", "imagenet": "ImageNet-DenseNet121"}
N_SPLITS, N_REPEATS, SEED, N_BOOT = 5, 5, 42, 2000


def boot_ci(y, s, seed=SEED):
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    v = np.empty(N_BOOT)
    for b in range(N_BOOT):
        i = np.concatenate([rng.choice(pos, pos.size, True), rng.choice(neg, neg.size, True)])
        v[b] = roc_auc_score(y[i], s[i])
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def smd(a, b):
    sp = np.sqrt((a.std(ddof=1) ** 2 + b.std(ddof=1) ** 2) / 2)
    return float((a.mean() - b.mean()) / sp) if sp > 0 else 0.0


def probe(X, y):
    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    oof = np.full((N_REPEATS, len(y)), np.nan)
    for i, (tr, te) in enumerate(rskf.split(X, y)):
        sc = StandardScaler()
        c = LogisticRegression(max_iter=5000, C=1.0, random_state=SEED).fit(
            sc.fit_transform(X[tr]), y[tr])
        oof[i // N_SPLITS, te] = c.predict_proba(sc.transform(X[te]))[:, 1]
    s = np.nanmean(oof, axis=0)
    lo, hi = boot_ci(y, s)
    return roc_auc_score(y, s), lo, hi


def main():
    m = pd.read_csv(PROXY)
    y = m.label.astype(int).to_numpy()
    Z = m[ACQ].apply(pd.to_numeric, errors="coerce").fillna(m[ACQ].median()).to_numpy(float)
    Zs = StandardScaler().fit_transform(Z)

    cont = [ACQ.index(c) for c in ACQ if c != "modality_cr"]
    W = Zs[:, cont]
    cov = np.cov(W, rowvar=False)
    Winv = np.linalg.pinv(cov)
    mod = m.modality_cr.to_numpy()
    # caliper: 0.2 SD of the pooled Mahalanobis distance to the opposite group mean
    rng = np.random.default_rng(SEED)
    pairs = []
    for mv in (0, 1):
        t_idx = np.flatnonzero((y == 1) & (mod == mv))
        c_idx = np.flatnonzero((y == 0) & (mod == mv))
        if len(t_idx) == 0 or len(c_idx) == 0:
            continue
        D = W[t_idx][:, None, :] - W[c_idx][None, :, :]
        dist = np.sqrt(np.einsum("ijk,kl,ijl->ij", D, Winv, D))
        cap = 0.2 * dist.std(ddof=1)
        avail = set(range(len(c_idx)))
        for a in rng.permutation(len(t_idx)):
            row = dist[a]
            cand = [j for j in avail if row[j] <= cap]
            if not cand:
                continue
            best = min(cand, key=lambda j: row[j])
            pairs.append((t_idx[a], c_idx[best]))
            avail.discard(best)
        print(f"[match] modality_cr={mv}: {len(t_idx)} hernia, {len(c_idx)} control, "
              f"caliper={cap:.4f}")
    idx = np.array([i for p in pairs for i in p])
    ym = y[idx]
    print(f"[match] {len(pairs)} pairs, n = {len(idx)} "
          f"({int(ym.sum())} hernia / {int((ym == 0).sum())} control)")

    bal = []
    for j, f in enumerate(ACQ):
        before = smd(Z[y == 1, j], Z[y == 0, j])
        after = smd(Z[idx][ym == 1, j], Z[idx][ym == 0, j])
        bal.append(dict(variable=f, smd_before=before, smd_after=after))
        print(f"  {f:24s} SMD {before:+.3f} -> {after:+.3f}")
    pd.DataFrame(bal).to_csv(f"{OUT}/matched_balance.csv", index=False)

    rows = []
    for k, disp in MODELS.items():
        X = np.load(f"{EMB}/embeddings_{k}.npz")["features"].astype(np.float32)
        a, lo, hi = probe(X[idx], ym)
        flag = "above chance" if lo > 0.5 else "includes 0.5"
        print(f"  {disp:22s} {a:.3f} [{lo:.3f}, {hi:.3f}]  {flag}")
        rows.append(dict(model=disp, n=len(idx), n_hernia=int(ym.sum()),
                         auc=a, ci_lo=lo, ci_hi=hi))
    pd.DataFrame(rows).to_csv(f"{OUT}/matched_analysis.csv", index=False)
    print("\n->", f"{OUT}/matched_analysis.csv")


if __name__ == "__main__":
    main()
