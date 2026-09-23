#!/usr/bin/env python3
"""
Summarise the vision-language-model arm (6 VLMs, zero-shot and QLoRA
fine-tuned, evaluated on three test sets) from the per-image prediction CSVs.

Reads the shuffle-corrected run only (results_shufflefix); the first
fine-tuning batch used a class-ordered training CSV and is excluded.
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

RES = Path("/scratch/hl106/80_workspace/foye/vlm_qlora/results_shufflefix")
OUT = Path("/scratch/hl106/80_workspace/foye/paper/results")

MODEL_DISPLAY = {
    "medgemma": "MedGemma-4B-IT",
    "huatuo": "HuatuoGPT-V-7B",
    "internvl": "InternVL",
    "qwen25vl": "Qwen2.5-VL",
    "llavaov": "LLaVA-OneVision",
    "smolvlm": "SmolVLM",
}
SET_DISPLAY = {
    "foye_hernia": "Internal test (same centre)",
    "foye_center2": "External centre",
    "foye_311_masked": "Metadata-masked cohort",
}
SET_ORDER = ["foye_hernia", "foye_center2", "foye_311_masked"]
COND_DISPLAY = {"zero_shot": "Zero-shot", "ft": "QLoRA fine-tuned"}


# Internal test images are restricted to the 1960-image analysis cohort
# (build_clean_cohort.py). The adapters were trained before that cohort was
# defined, so their training split may include some of the 49 dropped images.
CLEAN_IDS = set(pd.read_csv("/scratch/hl106/foye/cohort_1960/proxy_metadata.csv").image_id.astype(str))


def in_clean_cohort(image_path):
    return Path(str(image_path)).stem in CLEAN_IDS


def boot_auc_ci(y, s, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    if pos.size == 0 or neg.size == 0:
        return np.nan, np.nan
    v = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate(
            [rng.choice(pos, pos.size, replace=True), rng.choice(neg, neg.size, replace=True)]
        )
        v[b] = roc_auc_score(y[idx], s[idx])
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for csv_path in sorted(RES.glob("foye_*_on_foye_*_test.csv")):
        stem = csv_path.stem  # foye_<model>_<cond>_on_foye_<set>_test
        body = stem[len("foye_"):]
        left, right = body.split("_on_foye_", 1)
        model = left.split("_")[0]
        cond = "ft" if left.endswith("_ft") else "zero_shot"
        test_set = "foye_" + right[: -len("_test")]

        df = pd.read_csv(csv_path)
        if test_set == "foye_hernia":
            df = df[df["image_path"].map(in_clean_cohort)].reset_index(drop=True)
        # gold_idx / pred_idx: 0 = "yes" (hernia), 1 = "no". Recode to hernia=1.
        y = (df["gold_idx"].to_numpy() == 0).astype(int)
        pred = (df["pred_idx"].to_numpy() == 0).astype(int)
        # P(hernia) = probability assigned to option index 0
        probs = df["probabilities"].apply(lambda s: json.loads(s)[0]).to_numpy(dtype=float)

        acc = float((pred == y).mean())
        bacc = float(balanced_accuracy_score(y, pred))
        try:
            auc = float(roc_auc_score(y, probs))
            lo, hi = boot_auc_ci(y, probs)
        except ValueError:
            auc, lo, hi = np.nan, np.nan, np.nan
        rows.append(
            dict(
                model=model,
                model_display=MODEL_DISPLAY.get(model, model),
                condition=cond,
                condition_display=COND_DISPLAY[cond],
                test_set=test_set,
                test_set_display=SET_DISPLAY.get(test_set, test_set),
                n=int(len(df)),
                n_hernia=int(y.sum()),
                accuracy=acc,
                balanced_accuracy=bacc,
                auc=auc,
                auc_ci_lo=lo,
                auc_ci_hi=hi,
                pred_hernia_rate=float(pred.mean()),
                single_class_output=bool(pred.min() == pred.max()),
                mean_confidence=float(df["confidence"].mean()),
            )
        )

    out = pd.DataFrame(rows)
    out["set_rank"] = out["test_set"].map({k: i for i, k in enumerate(SET_ORDER)})
    out = out.sort_values(["model_display", "condition", "set_rank"]).drop(columns="set_rank")
    out.to_csv(OUT / "vlm_summary.csv", index=False)
    print(out.to_string(index=False))
    print("\n[done]", OUT / "vlm_summary.csv")


if __name__ == "__main__":
    main()
