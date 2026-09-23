#!/usr/bin/env python3
"""Did the VLMs use the burned-in text?

The four frozen encoders demonstrably did not: re-extracting them on the rebuilt
cohort moved every AUC by at most 0.005. But VLMs are the model class that reads
text, and their evaluation ran on the leaky build. This is the cheap test that
does not need the GPU: split each VLM's own test set by whether the mask failed
on that image, and compare. If a VLM is reading the age, it should do markedly
better where the age is legible.

Also correlates the VLM's hernia probability against the age itself.
"""

from __future__ import annotations

import ast
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import burned_in_text as B  # noqa: E402

PRED = Path("/scratch/hl106/80_workspace/foye/vlm_qlora/results_shufflefix")
RES = Path("/scratch/hl106/80_workspace/foye/paper/results")


def flag(path):
    im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if im is None:
        return str(path), -1
    return str(path), int(len(B.overlay_boxes(im)) > 0)


# Internal test images are restricted to the 1960-image analysis cohort
# (build_clean_cohort.py). The adapters were trained before that cohort was
# defined, so their training split may include some of the 49 dropped images.
CLEAN_IDS = set(pd.read_csv("/scratch/hl106/foye/cohort_1960/proxy_metadata.csv").image_id.astype(str))


def in_clean_cohort(image_path):
    return Path(str(image_path)).stem in CLEAN_IDS


def main():
    age = pd.read_csv(RES / "burned_in_age.csv").set_index("filename")["age_months"]

    files = sorted(PRED.glob("foye_*_on_foye_*_test.csv"))
    paths = set()
    frames = {}
    for f in files:
        d = pd.read_csv(f)
        if "_on_foye_hernia_" in f.name:
            d = d[d.image_path.map(in_clean_cohort)].reset_index(drop=True)
        frames[f.name] = d
        paths.update(d.image_path.dropna().tolist())
    paths = sorted(paths)
    print(f"{len(files)} prediction files, {len(paths)} distinct images")

    with Pool(24) as pool:
        res = dict(pool.map(flag, paths, chunksize=8))

    rows = []
    for name, d in frames.items():
        d = d.dropna(subset=["image_path"]).copy()
        d["text"] = d.image_path.map(res)
        d["p_hernia"] = [ast.literal_eval(p)[0] for p in d.probabilities]
        d["y"] = (d.gold_idx == 0).astype(int)
        d["age"] = [age.get(os.path.basename(p), np.nan) for p in d.image_path]

        if d.y.nunique() < 2 or d.p_hernia.nunique() < 2:
            continue
        r = {"file": name.replace("foye_", "").replace("_test.csv", ""),
             "n": len(d), "auc_all": roc_auc_score(d.y, d.p_hernia),
             "n_text": int((d.text == 1).sum())}
        for key, sub in [("text", d[d.text == 1]), ("notext", d[d.text == 0])]:
            r[f"auc_{key}"] = (roc_auc_score(sub.y, sub.p_hernia)
                               if len(sub) > 15 and sub.y.nunique() == 2 else np.nan)
        a = d.dropna(subset=["age"])
        # does the VLM's score track age among controls, where the label cannot
        # explain the association?
        ctrl = a[a.y == 0]
        r["rho_age_ctrl"] = (spearmanr(ctrl.age, ctrl.p_hernia).correlation
                             if len(ctrl) > 20 else np.nan)
        r["n_ctrl_age"] = len(ctrl)
        rows.append(r)

    out = pd.DataFrame(rows).sort_values("file")
    pd.set_option("display.width", 200)
    print(out.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    out.to_csv(RES / "vlm_text_check.csv", index=False)
    print("\nwrote", RES / "vlm_text_check.csv")


if __name__ == "__main__":
    main()
