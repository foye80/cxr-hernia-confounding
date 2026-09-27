#!/usr/bin/env python3
"""Does the vision-language-model split contaminate its own test set?

The split was made before the unit audit, on the 2009-image cohort, and it was
grouped by admission number rather than by child. Two questions follow. How much
of the training data did the audit later exclude? And does any child in the
reported internal test set also appear in the training or calibration split?

The reported internal test set is the part of the test split that belongs to the
1960-image cohort. Children are identified by the radiology number that prefixes
each file name.

Output: results/vlm_split_audit.json (counts only).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ITEMS = Path("/scratch/hl106/80_workspace/foye/vlm_qlora/data/foye_vqa_items.csv")
CLEAN = Path("/scratch/hl106/foye/cohort_1960/proxy_metadata.csv")
RES = Path("/scratch/hl106/80_workspace/foye/paper/results")


def main():
    items = pd.read_csv(ITEMS)
    h = items[items.dataset == "foye_hernia"].copy()
    h["stem"] = h.image_path.str.split("/").str[-1].str.replace(".jpg", "", regex=False)
    h["child"] = h.stem.str.split("_").str[0]
    clean = set(pd.read_csv(CLEAN).image_id.astype(str))
    h["in_clean"] = h.stem.isin(clean)

    out = {"splits": {}}
    for sp in ("train", "calib", "test"):
        d = h[h.split == sp]
        out["splits"][sp] = {"images": int(len(d)),
                             "in_analysis_cohort": int(d.in_clean.sum()),
                             "excluded_by_the_audit": int((~d.in_clean).sum()),
                             "children": int(d.child.nunique())}
    test = h[(h.split == "test") & h.in_clean]
    train, calib = h[h.split == "train"], h[h.split == "calib"]
    out["reported_internal_test"] = {
        "images": int(len(test)),
        "children": int(test.child.nunique()),
        "children_also_in_training": int(len(set(test.child) & set(train.child))),
        "children_also_in_calibration": int(len(set(test.child) & set(calib.child))),
    }
    out["any_child_in_both_train_and_test_split"] = int(
        len(set(train.child) & set(h[h.split == "test"].child)))
    RES.mkdir(exist_ok=True)
    (RES / "vlm_split_audit.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
