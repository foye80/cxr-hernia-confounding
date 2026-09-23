#!/usr/bin/env python3
"""Write the 1960-image analysis cohort (revision, 2026-09).

`audit_units.py` decides which of the 2009 radiographs have an unambiguous
unit: one child, one image, one clinical record, and a label that agrees with
the diagnosis text of that admission. This script applies its mask to every
row-aligned input the analysis scripts read, so each of them only needs its
paths changed:

  proxy table      -> /scratch/hl106/foye/cohort_1960/proxy_metadata.csv
  embeddings       -> /scratch/hl106/foye/cohort_1960/emb_published/
                      /scratch/hl106/foye/cohort_1960/emb_remasked/
  per-image tables -> paper/results/burned_in_age.csv,
                      mask_failure_flag.csv, mask_failure_flag_remasked.csv

The 2009-row originals stay where they were (embeddings, proxy table) or in
/scratch/hl106/foye/cohort_2009_derived/ (per-image tables). Everything with a
file name or identifier in it is written under /scratch/hl106/foye/, never
under paper/, except the three per-image tables the scripts already kept in
paper/results/.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FOYE = Path("/scratch/hl106/foye")
PAPER = Path("/scratch/hl106/80_workspace/foye/paper")
FULL_PROXY = Path("/scratch/hl106/80_workspace/foye/radiographic_proxy_probe/"
                  "primary_2k_radiographic_proxy_metadata.csv")
FULL_EMB = {"emb_published": FOYE / "results_2k",
            "emb_remasked": FOYE / "results_2k_remasked"}
FULL_DERIVED = FOYE / "cohort_2009_derived"
OUT = FOYE / "cohort_1960"
MASK = PAPER / "results" / "clean_cohort_mask.npz"
MODELS = ["biomedclip", "torchxrayvision", "rad-dino", "imagenet"]


def main():
    keep = np.load(MASK)["keep"]
    proxy = pd.read_csv(FULL_PROXY)
    assert len(proxy) == len(keep) == 2009
    y_full = proxy["label"].astype(int).to_numpy()

    OUT.mkdir(parents=True, exist_ok=True)
    clean = proxy[keep].reset_index(drop=True)
    clean.to_csv(OUT / "proxy_metadata.csv", index=False)
    print(f"proxy table: {len(clean)} rows, labels {clean.label.value_counts().to_dict()}")

    for name, src in FULL_EMB.items():
        (OUT / name).mkdir(exist_ok=True)
        for m in MODELS:
            z = np.load(src / f"embeddings_{m}.npz")
            X, y = z["features"], z["labels"].astype(int)
            assert len(X) == 2009 and (y == y_full).all(), f"{name}/{m} out of order"
            np.savez(OUT / name / f"embeddings_{m}.npz", features=X[keep], labels=y[keep])
        print(f"{name}: {len(MODELS)} encoders, {int(keep.sum())} rows")

    for fname in ["burned_in_age.csv", "mask_failure_flag.csv",
                  "mask_failure_flag_remasked.csv"]:
        d = pd.read_csv(FULL_DERIVED / fname)
        assert len(d) == 2009 and (d.label.to_numpy() == y_full).all(), fname
        # same image in the same row: file name stem equals the proxy image_id
        stem = d.filename.astype(str).str.replace(r"\.[a-z]+$", "", regex=True)
        stem = stem.str.split("/").str[-1]
        assert (stem.to_numpy() == proxy.image_id.astype(str).to_numpy()).all(), fname
        d[keep].reset_index(drop=True).to_csv(PAPER / "results" / fname, index=False)
        print(f"{fname}: {int(keep.sum())} rows")


if __name__ == "__main__":
    main()
