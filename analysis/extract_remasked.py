#!/usr/bin/env python3
"""Re-extract the four frozen encoders on the properly masked cohort.

Same encoders, same preprocessing and same image order as the published run --
`extract_embeddings` is imported from the original script rather than
reimplemented, so the only thing that differs is which pixels the images carry.

Writes `/scratch/hl106/foye/results_2k_remasked/embeddings_<model>.npz`, matching
the layout of `results_2k/` so `run_paper_analysis.py --emb-dir` can consume it.

    CUDA_VISIBLE_DEVICES=4 HF_HOME=/scratch/hl106/huggingface \
      /scratch/hl106/conda_envs/medshortcut/bin/python extract_remasked.py
"""

import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/scratch/hl106/foye")
from generate_mlhc_figures import (extract_embeddings,  # noqa: E402
                                   load_image_paths_labels)

SRC = Path("/scratch/hl106/foye/ChestCR_remasked")
REF = Path("/scratch/hl106/foye/results_2k")
OUT = Path("/scratch/hl106/foye/results_2k_remasked")
MODELS = ["torchxrayvision", "rad-dino", "biomedclip", "imagenet"]


def main():
    paths, labels = load_image_paths_labels(str(SRC))
    print(f"{len(paths)} images, hernia={int(labels.sum())}, "
          f"control={int((labels == 0).sum())}", flush=True)

    ref = np.load(REF / "embeddings_biomedclip.npz")["labels"]
    assert len(ref) == len(labels) and (ref == labels).all(), \
        "image order does not match the published run"
    print("image order matches results_2k", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    for m in MODELS:
        dst = OUT / f"embeddings_{m}.npz"
        if dst.exists():
            print(f"[{m}] exists, skipping", flush=True)
            continue
        t0 = time.time()
        feats = extract_embeddings(paths, labels, m, device="cuda", batch_size=32)
        np.savez_compressed(dst, features=feats.astype(np.float32), labels=labels)
        old = np.load(REF / f"embeddings_{m}.npz")["features"]
        d = np.linalg.norm(feats - old, axis=1) / (np.linalg.norm(old, axis=1) + 1e-9)
        print(f"[{m}] {feats.shape} in {time.time()-t0:.0f}s -> {dst.name}; "
              f"relative change vs published embeddings: median {np.median(d):.4f}, "
              f"p90 {np.percentile(d, 90):.4f}", flush=True)


if __name__ == "__main__":
    main()
