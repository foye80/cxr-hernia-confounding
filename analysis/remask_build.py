#!/usr/bin/env python3
"""Rebuild the image cohorts with the burned-in overlay actually removed.

The original `prepare_dataset.py:mask_image` blanks fixed rectangles sized
8% x 12% (top-left), 5% x 15% (top-right) and two small bottom corners. The
console overlay is a four-line block (ID / name / sex / age) that is taller than
that, so the age line, and often the study date, survived. 32.5% of the prepared
primary cohort still carried legible text, and it survived more often in hernia
images (38.8%) than in controls (26.2%).

Blanking only the *detected* text would swap one leak for another: hernia images
would have more pixels blanked than controls, and the blanked area would carry
the label. So the mask is a fixed rectangle per corner, sized from the whole
cohort's text extent, identical for every image; detected text outside those
rectangles is blanked as well, but by construction that is rare.

Pass 1 measures the extents, pass 2 writes the images, pass 3 audits the output.

    python remask_build.py            # both cohorts
    python remask_build.py --measure  # pass 1 only, print the extents
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import burned_in_text as B  # noqa: E402

FOYE = Path("/scratch/hl106/foye")
RES = Path("/scratch/hl106/80_workspace/foye/paper/results")

COHORTS = {
    # name:      (originals,      cohort as analysed,        output)
    "primary": (FOYE / "ChestCR", FOYE / "ChestCR_prepared", FOYE / "ChestCR_remasked"),
}
# The external cohort is not rebuilt. Its 100 files are screen captures already
# cropped to the radiation field: the detector finds burned-in text on 0 of 100,
# only the etched laterality marker. Leaving them untouched also keeps the
# external comparison against the published numbers like for like.
EXTS = {".jpg", ".jpeg", ".png"}
QUANTILE = 99.5          # of the per-image text extent, per corner
PAD = 0.012              # extra margin, as a fraction of width/height
ANCHOR = 0.06            # overlay text starts this close to the frame edge
TOP = 0.30               # ...and this high up


def anchored(boxes, h, w):
    """Overlay blocks are pinned to a top corner of the frame. Anatomy that
    happens to match the glyph signature is not, so it is dropped here."""
    out = []
    for x0, y0, x1, y1 in boxes:
        if y0 > h * TOP:
            continue
        if x0 <= w * ANCHOR or x1 >= w * (1 - ANCHOR):
            out.append((x0, y0, x1, y1))
    return out


def members(prepared: Path) -> list[tuple[str, str]]:
    """(label_folder, filename) for every image in the cohort as analysed."""
    out = []
    for folder in ("0", "1"):
        d = prepared / folder
        if not d.is_dir():
            continue
        for f in sorted(os.listdir(d)):
            if os.path.splitext(f)[1].lower() in EXTS:
                out.append((folder, f))
    return out


def extents(job):
    """Per-image text extent in each corner, as fractions of the image."""
    src, = job
    im = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
    if im is None:
        return None
    h, w = im.shape
    boxes = anchored(B.overlay_boxes(im), h, w)
    tl = [0.0, 0.0]   # right edge, bottom edge of top-left text
    tr = [1.0, 0.0]   # left edge, bottom edge of top-right text
    for x0, y0, x1, y1 in boxes:
        if x0 < w * 0.5:
            tl[0] = max(tl[0], x1 / w)
            tl[1] = max(tl[1], y1 / h)
        else:
            tr[0] = min(tr[0], x0 / w)
            tr[1] = max(tr[1], y1 / h)
    return tl, tr, len(boxes)


def rectangles(ext: list) -> dict:
    """Fixed corner rectangles covering `QUANTILE` of the cohort's text."""
    tl_x = np.array([e[0][0] for e in ext if e[0][0] > 0])
    tl_y = np.array([e[0][1] for e in ext if e[0][1] > 0])
    tr_x = np.array([e[1][0] for e in ext if e[1][0] < 1])
    tr_y = np.array([e[1][1] for e in ext if e[1][1] > 0])
    return {
        "top_left":  {"w": float(np.percentile(tl_x, QUANTILE)) + PAD,
                      "h": float(np.percentile(tl_y, QUANTILE)) + PAD},
        "top_right": {"w": 1.0 - float(np.percentile(tr_x, 100 - QUANTILE)) + PAD,
                      "h": float(np.percentile(tr_y, QUANTILE)) + PAD},
        "n_with_left_text": int(len(tl_x)),
        "n_with_right_text": int(len(tr_x)),
    }


_RECT = None


def _init(rect):
    global _RECT
    _RECT = rect


def write_one(job):
    src, dst = job
    im = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
    if im is None:
        return str(src), None
    h, w = im.shape
    r = _RECT
    im[:int(h * r["top_left"]["h"]), :int(w * r["top_left"]["w"])] = 0
    im[:int(h * r["top_right"]["h"]), int(w * (1 - r["top_right"]["w"])):] = 0
    # laterality marker and scale bar, as before
    im[int(h * 0.88):, int(w * 0.90):] = 0
    im[int(h * 0.95):, :int(w * 0.04)] = 0

    extra = 0
    for x0, y0, x1, y1 in anchored(B.overlay_boxes(im), h, w):
        im[y0:y1, x0:x1] = 0
        extra += 1

    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), im, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return str(dst), extra


def audit_one(job):
    dst, = job
    im = cv2.imread(str(dst), cv2.IMREAD_GRAYSCALE)
    if im is None:
        return str(dst), -1
    return str(dst), len(B.overlay_boxes(im))


def run(name: str, raw: Path, prepared: Path, out: Path, procs: int, measure: bool):
    mem = members(prepared)
    srcs, dsts = [], []
    missing = []
    for folder, f in mem:
        s = raw / folder / f
        if not s.exists():
            missing.append(f"{folder}/{f}")
            continue
        srcs.append(s)
        dsts.append(out / folder / f)

    print(f"[{name}] cohort {len(mem)} images, originals found for {len(srcs)}")
    if missing:
        print(f"[{name}]   no original for {len(missing)}: {missing[:5]}")

    with Pool(procs) as pool:
        ext = [e for e in pool.map(extents, [(s,) for s in srcs], chunksize=8)
               if e is not None]
    rect = rectangles(ext)
    print(f"[{name}] text found in {rect['n_with_left_text']} left / "
          f"{rect['n_with_right_text']} right corners")
    print(f"[{name}] fixed mask: top-left {100*rect['top_left']['w']:.1f}% x "
          f"{100*rect['top_left']['h']:.1f}%   top-right "
          f"{100*rect['top_right']['w']:.1f}% x {100*rect['top_right']['h']:.1f}%")
    print(f"[{name}]   (the old mask was 12.0% x 8.0% and 15.0% x 5.0%)")
    if measure:
        return rect, None

    with Pool(procs, initializer=_init, initargs=(rect,)) as pool:
        res = pool.map(write_one, list(zip(srcs, dsts)), chunksize=8)
    extra = np.array([r[1] for r in res if r[1] is not None])
    print(f"[{name}] wrote {len(res)} images; text outside the fixed rectangles "
          f"on {int((extra > 0).sum())} of them")

    with Pool(procs) as pool:
        aud = pool.map(audit_one, [(d,) for d in dsts], chunksize=8)
    left = np.array([a[1] for a in aud])
    print(f"[{name}] AUDIT: residual text on {int((left > 0).sum())} / {len(left)} "
          f"images ({100*(left > 0).mean():.2f}%)")

    return rect, {
        "n": len(dsts),
        "extra_boxes_images": int((extra > 0).sum()),
        "residual_images": int((left > 0).sum()),
        "residual_paths": [a[0] for a in aud if a[1] > 0][:50],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--measure", action="store_true")
    ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--cohort", default=None, choices=list(COHORTS))
    args = ap.parse_args()

    report = {}
    for name, (raw, prepared, out) in COHORTS.items():
        if args.cohort and name != args.cohort:
            continue
        rect, audit = run(name, raw, prepared, out, args.procs, args.measure)
        report[name] = {"rect": rect, "audit": audit, "out": str(out)}
        print()

    if not args.measure:
        RES.mkdir(parents=True, exist_ok=True)
        p = RES / "remask_report.json"
        p.write_text(json.dumps(report, indent=2))
        print("wrote", p)


if __name__ == "__main__":
    main()
