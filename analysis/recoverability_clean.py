#!/usr/bin/env python3
"""Table S1 / Figure S1 on the 1960-image cohort: how well each proxy variable
can be predicted from each frozen embedding.

Same probe suite and task list as ../../radiographic_proxy_probe.py, which
built the 2009-image version; only the inputs change (the proxy table and
embeddings written by build_clean_cohort.py).

Output: results/recoverability_summary.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

FOYE_CODE = Path("/scratch/hl106/80_workspace/foye")
sys.path.insert(0, str(FOYE_CODE))

from phenotype_probe import load_embeddings, run_probe_suite, write_csv  # noqa: E402

PROXY_CSV = "/scratch/hl106/foye/cohort_1960/proxy_metadata.csv"
EMB_DIR = "/scratch/hl106/foye/cohort_1960/emb_remasked"
RES = Path("/scratch/hl106/80_workspace/foye/paper/results")
WORK = Path("/scratch/hl106/foye/cohort_1960/recoverability_work")

CLASSIFICATION = [("sex_male", "sex_male"), ("modality_cr", "CR_vs_DX")]
REGRESSION = [
    "exam_days_since_min", "image_width_px", "image_height_px", "image_aspect",
    "body_bbox_width_frac", "body_bbox_height_frac", "body_bbox_area_frac",
    "body_bbox_aspect", "body_mask_area_frac", "upper_body_area_share",
    "mid_body_area_share", "lower_body_area_share", "upper_body_width_frac",
    "mid_body_width_frac", "lower_body_width_frac", "shoulder_width_frac",
    "thorax_width_frac", "shoulder_to_lower_width_ratio",
    "thorax_to_lower_width_ratio", "thorax_mean_intensity",
    "thorax_p10_intensity", "thorax_p90_intensity", "central_lucency_frac",
    "left_lucency_frac", "right_lucency_frac", "lucency_lr_asymmetry",
    "thorax_body_mask_frac",
]


def main():
    df = pd.read_csv(PROXY_CSV)
    rows = []
    for rec in df.to_dict("records"):
        r = {}
        for k, v in rec.items():
            if isinstance(v, float) and not np.isfinite(v):
                r[k] = None
            else:
                r[k] = v
        for k, _ in CLASSIFICATION:
            if r.get(k) is not None:
                r[k] = int(r[k])
        rows.append(r)
    emb = load_embeddings(EMB_DIR)
    tasks = ([{"target": t, "name": n, "type": "classification"} for t, n in CLASSIFICATION]
             + [{"target": t, "name": t, "type": "regression"} for t in REGRESSION])
    summary, _ = run_probe_suite("primary_1960_radiographic", rows, emb, tasks, str(WORK))
    write_csv(str(RES / "recoverability_summary.csv"), summary)
    print("wrote", RES / "recoverability_summary.csv", len(summary), "rows")


if __name__ == "__main__":
    main()
