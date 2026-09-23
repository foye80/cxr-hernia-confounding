#!/usr/bin/env python3
"""Read the patient age out of the burned-in overlay.

The manuscript states that age is unavailable: `ChestCR_prepared/metadata.json`
carries sex for all 2,020 clinical records and age for none. But the console
burns a four-line block into the pixels -- ID / name / sex / age -- and the age
line survived the original mask on a third of the cohort. It can simply be read.

The overlay is a fixed bitmap font at a fixed grey level, so no OCR engine is
needed: cluster the glyph bitmaps, label the clusters once, then match. Ages are
rendered as three digits plus a unit, `003Y` or `014M`.

    python ocr_age.py --templates   # pass 1: cluster glyphs, write a montage
    python ocr_age.py               # pass 2: read ages, write burned_in_age.csv
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import burned_in_text as B  # noqa: E402

FOYE = Path("/scratch/hl106/foye")
RAW = FOYE / "ChestCR"
PREPARED = FOYE / "ChestCR_prepared"
RES = Path("/scratch/hl106/80_workspace/foye/paper/results")
ART = Path(__file__).resolve().parent / "ocr_templates"

GH, GW = 20, 14          # canonical glyph size
AGE_RE = re.compile(r"^(\d{3})([YMD])$")


def members():
    out = []
    for folder in ("0", "1"):
        for f in sorted(os.listdir(PREPARED / folder)):
            if os.path.splitext(f)[1].lower() in {".jpg", ".jpeg", ".png"}:
                out.append((folder, f))
    return out


def norm(patch: np.ndarray) -> np.ndarray:
    """Glyph bitmap, size- and contrast-normalised."""
    if patch.size == 0:
        return np.zeros((GH, GW), np.float32)
    g = cv2.resize(patch.astype(np.float32), (GW, GH), interpolation=cv2.INTER_AREA)
    lo, hi = g.min(), g.max()
    return (g - lo) / (hi - lo) if hi > lo else g * 0


def left_block_rows(img):
    h, w = img.shape
    rs = [r for r in B.overlay_rows(img)
          if r["x0"] <= w * 0.06 and r["y0"] <= h * 0.30]
    rs.sort(key=lambda r: r["y0"])
    return rs


def age_row(img):
    """The overlay's age line: the bottom line of the left block, 3-5 glyphs."""
    rs = left_block_rows(img)
    if not rs:
        return None
    r = rs[-1]
    return r if 3 <= r["n_glyphs"] <= 5 else None


def row_glyphs(img, row):
    out = []
    for gx, gy, gw, gh in sorted(row["glyphs"], key=lambda b: b[0]):
        out.append(norm(img[gy:gy + gh, gx:gx + gw]))
    return out


def collect(path):
    im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if im is None:
        return []
    r = age_row(im)
    return row_glyphs(im, r) if r else []


def read_one(job):
    path, templates, labels = job
    im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if im is None:
        return str(path), None, None, None
    r = age_row(im)
    if r is None:
        return str(path), None, None, None
    text, worst = "", 0.0
    for g in row_glyphs(im, r):
        d = np.sqrt(((templates - g[None]) ** 2).sum(axis=(1, 2)))
        i = int(np.argmin(d))
        text += labels[i]
        worst = max(worst, float(d[i]))
    return str(path), text, worst, r["n_glyphs"]


def months(text):
    m = AGE_RE.match(text)
    if not m:
        return None
    n, unit = int(m.group(1)), m.group(2)
    return {"Y": 12 * n, "M": n, "D": n / 30.44}[unit]


def build_templates(procs):
    mem = members()
    paths = [RAW / f / n for f, n in mem]
    with Pool(procs) as pool:
        chunks = pool.map(collect, paths, chunksize=8)
    G = np.array([g for c in chunks for g in c], np.float32)
    print(f"collected {len(G)} glyphs from {sum(1 for c in chunks if c)} age rows")

    X = G.reshape(len(G), -1)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 0.1)
    K = 24
    _, lab, cen = cv2.kmeans(X, K, None, crit, 8, cv2.KMEANS_PP_CENTERS)
    lab = lab.ravel()

    ART.mkdir(parents=True, exist_ok=True)
    np.save(ART / "centres.npy", cen.reshape(K, GH, GW))
    counts = np.bincount(lab, minlength=K)

    S = 5
    sheet = np.zeros((GH * S + 30, GW * S * K), np.uint8)
    for k in range(K):
        tile = cv2.resize((cen[k].reshape(GH, GW) * 255).astype(np.uint8),
                          (GW * S, GH * S), interpolation=cv2.INTER_NEAREST)
        sheet[:GH * S, k * GW * S:(k + 1) * GW * S] = tile
        cv2.putText(sheet, str(k), (k * GW * S + 4, GH * S + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 1)
    cv2.imwrite(str(ART / "clusters.png"), sheet)
    print("cluster sizes:", dict(enumerate(counts.tolist())))
    print("wrote", ART / "clusters.png", "- label it into labels.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--templates", action="store_true")
    ap.add_argument("--procs", type=int, default=24)
    args = ap.parse_args()

    if args.templates:
        build_templates(args.procs)
        return

    cen = np.load(ART / "centres.npy")
    lab = json.loads((ART / "labels.json").read_text())
    # Junk clusters are kept, not dropped: they catch smeared and clipped glyphs
    # that would otherwise be forced onto a digit template and read as a real age.
    labels = [lab[str(i)] for i in range(len(cen))]

    mem = members()
    jobs = [(RAW / f / n, cen, labels) for f, n in mem]
    with Pool(args.procs) as pool:
        res = pool.map(read_one, jobs, chunksize=8)

    RES.mkdir(parents=True, exist_ok=True)
    out = RES / "burned_in_age.csv"
    n_ok = 0
    with open(out, "w") as fh:
        fh.write("label,filename,age_text,age_months,glyph_distance,n_glyphs\n")
        for (folder, name), (_, text, worst, ng) in zip(mem, res):
            lab = 1 if folder == "0" else 0
            am = months(text) if text else None
            if am is not None:
                n_ok += 1
            fh.write(f"{lab},{name},{text or ''},"
                     f"{'' if am is None else f'{am:.2f}'},"
                     f"{'' if worst is None else f'{worst:.3f}'},"
                     f"{'' if ng is None else ng}\n")
    print(f"read a valid age for {n_ok} / {len(mem)} images ({100*n_ok/len(mem):.1f}%)")
    print("wrote", out)


if __name__ == "__main__":
    main()
