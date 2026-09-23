#!/usr/bin/env python3
"""How much burned-in text survived the original fixed-rectangle mask.

Runs the detector over a cohort directory and writes a per-image flag in the
same order the analysis uses, so the flag can be treated as just another
candidate predictor.

    python audit_text.py                                    # as-published cohort
    python audit_text.py --dir /scratch/hl106/foye/ChestCR_remasked --tag remasked
"""

from __future__ import annotations

import argparse
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import burned_in_text as B  # noqa: E402


RES = Path("/scratch/hl106/80_workspace/foye/paper/results")
EXTS = {".jpg", ".jpeg", ".png"}


def scan(path):
    """Flag any overlay text at all.

    Deliberately *not* the anchored rule that `remask_build` uses to decide what
    to blank. Anchoring assumes the text still touches the frame edge, which is
    exactly what the original mask cut away: on the published cohort the visible
    remnant starts at x = 12% of the width, so anchoring would score it clean.
    """
    im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if im is None:
        return str(path), 0
    return str(path), int(len(B.overlay_boxes(im)) > 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/scratch/hl106/foye/ChestCR_prepared")
    ap.add_argument("--tag", default="published")
    ap.add_argument("--procs", type=int, default=24)
    args = ap.parse_args()

    root = Path(args.dir)
    paths, y, names = [], [], []
    for folder, lab in [("0", 1), ("1", 0)]:
        for f in sorted(os.listdir(root / folder)):
            if os.path.splitext(f)[1].lower() in EXTS:
                paths.append(root / folder / f)
                y.append(lab)
                names.append(f)
    y = np.array(y)

    with Pool(args.procs) as pool:
        res = pool.map(scan, paths, chunksize=8)
    hit = np.array([r[1] for r in res])

    print(f"{root}")
    print(f"  images with residual burned-in text: {hit.sum()} / {len(hit)} "
          f"({100*hit.mean():.1f}%)")
    print(f"    hernia  {hit[y==1].sum():4d}/{(y==1).sum()} ({100*hit[y==1].mean():.1f}%)")
    print(f"    control {hit[y==0].sum():4d}/{(y==0).sum()} ({100*hit[y==0].mean():.1f}%)")
    if 0 < hit.sum() < len(hit):
        print(f"  AUC of the flag alone: {roc_auc_score(y, hit.astype(float)):.4f}")

    out = RES / ("mask_failure_flag.csv" if args.tag == "published"
                 else f"mask_failure_flag_{args.tag}.csv")
    with open(out, "w") as fh:
        fh.write("label,filename,residual_text\n")
        for lab, nm, h in zip(y, names, hit):
            fh.write(f"{lab},{nm},{h}\n")
    print("  wrote", out)


if __name__ == "__main__":
    main()
