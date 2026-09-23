#!/usr/bin/env python3
"""External validation with confidence intervals.

The April run reported point estimates only. This re-extracts embeddings for
the second-centre images (CPU, cached weights), refits the probe on the full
primary cohort, and adds a stratified bootstrap CI. The extracted external
embeddings are cached so this only has to run once.

Also reports the acquisition-only proxy model on the external set is not
possible (those images carry no acquisition metadata), which is itself part of
the finding: the external set cannot adjudicate the acquisition explanation.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "/scratch/hl106/foye")
from generate_mlhc_figures import extract_embeddings, load_image_paths_labels  # noqa: E402

EXT_DIR = "/scratch/hl106/foye/center2_CR"
EMB_DIR = "/scratch/hl106/foye/cohort_1960/emb_remasked"  # primary training set, 1960 images
CACHE = Path("/scratch/hl106/80_workspace/foye/paper/results/external_embeddings.npz")
OUT = "/scratch/hl106/80_workspace/foye/paper/results/external_validation.csv"

MODELS = ["biomedclip", "torchxrayvision", "rad-dino", "imagenet"]
DISP = {"biomedclip": "BiomedCLIP", "torchxrayvision": "TorchXRayVision",
        "rad-dino": "RAD-DINO", "imagenet": "ImageNet-DenseNet121"}
N_BOOT, SEED = 2000, 42


def boot_ci(y, s, seed=SEED):
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    v = np.empty(N_BOOT)
    for b in range(N_BOOT):
        i = np.concatenate([rng.choice(pos, pos.size, True), rng.choice(neg, neg.size, True)])
        v[b] = roc_auc_score(y[i], s[i])
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def main():
    paths, y_ext = load_image_paths_labels(EXT_DIR)
    print(f"[ext] {len(paths)} images, hernia={int(y_ext.sum())}, non-hernia={int((y_ext==0).sum())}")

    if CACHE.exists():
        z = np.load(CACHE)
        ext = {m: z[m] for m in MODELS}
        print("[ext] loaded cached embeddings")
    else:
        ext = {}
        for m in MODELS:
            print(f"[ext] extracting {m} on CPU ...", flush=True)
            ext[m] = extract_embeddings(paths, y_ext, m, device="cpu", batch_size=8)
            print(f"       {ext[m].shape}")
        np.savez_compressed(CACHE, **ext, labels=y_ext)

    rows = []
    for m in MODELS:
        z = np.load(f"{EMB_DIR}/embeddings_{m}.npz")
        Xp, yp = z["features"].astype(np.float32), z["labels"].astype(int)
        sc = StandardScaler().fit(Xp)
        clf = LogisticRegression(max_iter=5000, C=1.0, random_state=SEED).fit(sc.transform(Xp), yp)
        s = clf.predict_proba(sc.transform(ext[m].astype(np.float32)))[:, 1]
        a = roc_auc_score(y_ext, s)
        lo, hi = boot_ci(y_ext, s)
        print(f"  {DISP[m]:22s} AUC {a:.3f} [{lo:.3f}, {hi:.3f}]")
        rows.append(dict(model=m, model_display=DISP[m], n=len(y_ext),
                         n_hernia=int(y_ext.sum()), auc=a, ci_lo=lo, ci_hi=hi))
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print("\n->", OUT)


if __name__ == "__main__":
    main()
