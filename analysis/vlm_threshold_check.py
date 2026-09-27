#!/usr/bin/env python3
"""What fails at the external center: the decision or the ranking?

Every fine-tuned vision-language model answers "no" for all 100 external
images, so its balanced accuracy is exactly 0.500 whatever its scores rank.
That is a property of the decision threshold, not of the representation. This
script asks what a site would obtain if it kept the model and re-fitted only
the threshold on a few labelled local images.

For each model: the AUC, the balanced accuracy at the model's own threshold,
the best threshold in hindsight, and an honest estimate that chooses the
threshold on a random half of the external images and evaluates it on the other
half (200 repetitions).

Temperature scaling cannot do this. Dividing both logits by a positive
temperature is monotone, so it leaves every decision unchanged; only a shift of
the threshold, or an intercept as in Platt scaling, can.

Output: results/vlm_threshold.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

PRED = Path("/scratch/hl106/80_workspace/foye/vlm_qlora/results_shufflefix")
RES = Path("/scratch/hl106/80_workspace/foye/paper/results")
DISPLAY = {"medgemma": "MedGemma-4B-IT", "huatuo": "HuatuoGPT-V-7B",
           "internvl": "InternVL", "qwen25vl": "Qwen2.5-VL",
           "llavaov": "LLaVA-OneVision", "smolvlm": "SmolVLM"}
SEED, N_SPLIT = 42, 200


def best_threshold(y, p):
    out = max(((balanced_accuracy_score(y, (p >= t).astype(int)), float(t))
               for t in np.unique(p)))
    return out[1], float(out[0])


def main():
    rng = np.random.default_rng(SEED)
    rows = []
    for f in sorted(PRED.glob("foye_*_ft_on_foye_center2_test.csv")):
        key = f.name.split("_")[1]
        d = pd.read_csv(f)
        p = d.probabilities.apply(lambda s: json.loads(s)[0]).to_numpy(float)
        y = (d.gold_idx == 0).astype(int).to_numpy()
        thr, bacc_best = best_threshold(y, p)
        half = []
        for _ in range(N_SPLIT):
            idx = rng.permutation(len(y))
            a, b = idx[:len(y) // 2], idx[len(y) // 2:]
            t, _ = best_threshold(y[a], p[a])
            half.append(balanced_accuracy_score(y[b], (p[b] >= t).astype(int)))
        rows.append({
            "model": DISPLAY.get(key, key),
            "n": int(len(y)),
            "auc": float(roc_auc_score(y, p)),
            "balanced_accuracy_at_model_threshold":
                float(balanced_accuracy_score(y, (p >= 0.5).astype(int))),
            "predicted_hernia_rate": float((p >= 0.5).mean()),
            "p_hernia_min": float(p.min()), "p_hernia_max": float(p.max()),
            "best_threshold_in_hindsight": thr,
            "balanced_accuracy_at_best_threshold": bacc_best,
            "balanced_accuracy_threshold_fitted_on_half": float(np.mean(half)),
            "sd_over_splits": float(np.std(half)),
        })
    out = {"note": "external cohort, fine-tuned models; the threshold is the only "
                   "thing re-fitted, and it is fitted on labelled external images",
           "n_splits": N_SPLIT, "seed": SEED, "models": rows}
    (RES / "vlm_threshold.json").write_text(json.dumps(out, indent=1))
    for r in sorted(rows, key=lambda r: -r["auc"]):
        print(f"{r['model']:22s} AUC {r['auc']:.3f}  at model threshold "
              f"{r['balanced_accuracy_at_model_threshold']:.3f}  threshold fitted on half "
              f"{r['balanced_accuracy_threshold_fitted_on_half']:.3f} "
              f"+/- {r['sd_over_splits']:.3f}")
    print("\nwrote", RES / "vlm_threshold.json")


if __name__ == "__main__":
    main()
