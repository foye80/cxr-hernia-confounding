#!/usr/bin/env python3
"""Find burned-in annotation text in a radiograph.

The console overlay is a bitmap font rendered at constant intensity on top of
the image. Morphological top-hat isolates small bright structures that sit on a
locally darker background; glyph-sized components sharing a baseline and a
regular character pitch form a text row.

Used three ways:
  * audit    - how much text survived the fixed-rectangle mask (`audit_text.py`)
  * remask   - blank the detected rows instead of fixed rectangles (`remask_build.py`)
  * read age - the fourth overlay line is the patient age (`ocr_age.py`)
"""

from __future__ import annotations

import cv2
import numpy as np

TOPHAT_THRESH = 60          # intensity a glyph must stand above its surroundings
MIN_GLYPHS_PER_ROW = 3


def _glyph_bounds(h: int) -> tuple[int, int]:
    """Plausible glyph height in pixels for an image of height `h`."""
    return max(6, int(h * 0.006)), max(14, int(h * 0.035))


def glyphs(region: np.ndarray, img_h: int) -> list[tuple[int, int, int, int]]:
    """Glyph-like connected components as (x, y, w, h) inside `region`."""
    if region.size == 0:
        return []
    gmin, gmax = _glyph_bounds(img_h)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * gmax + 1, 2 * gmax + 1))
    th = cv2.morphologyEx(region, cv2.MORPH_TOPHAT, k)
    n, _, stats, _ = cv2.connectedComponentsWithStats(
        (th >= TOPHAT_THRESH).astype(np.uint8), 8)

    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if gmin <= h <= gmax and 3 <= w <= gmax * 1.4 and area >= 12 \
                and area >= 0.15 * w * h:
            out.append((int(x), int(y), int(w), int(h)))
    return out


def rows(region: np.ndarray, img_h: int) -> list[dict]:
    """Group glyphs into text rows.

    Returns dicts with the row bounding box in `region` coordinates, the glyph
    boxes it contains, and the median character pitch.
    """
    g = glyphs(region, img_h)
    if len(g) < MIN_GLYPHS_PER_ROW:
        return []
    _, gmax = _glyph_bounds(img_h)

    arr = np.array(g, dtype=float)
    cy = arr[:, 1] + arr[:, 3] / 2.0
    cx = arr[:, 0] + arr[:, 2] / 2.0

    used = np.zeros(len(arr), bool)
    found = []
    for i in np.argsort(cy):
        if used[i]:
            continue
        tol = max(4.0, arr[i, 3] * 0.5)
        same = (~used) & (np.abs(cy - cy[i]) <= tol)
        if same.sum() < MIN_GLYPHS_PER_ROW:
            continue
        order = np.argsort(cx[same])
        idx = np.flatnonzero(same)[order]
        pitch = np.diff(cx[idx])
        if len(pitch) < 2:
            continue
        med = float(np.median(pitch))
        # a rendered string has near-constant pitch; anatomy speckle does not
        if med > gmax * 2.2:
            continue
        if (np.abs(pitch - med) <= gmax * 0.8).mean() < 0.6:
            continue
        used[idx] = True

        # The top-left and top-right overlay blocks often share a baseline. Cut
        # the run wherever the gap jumps, so one row never spans both corners.
        cuts = np.flatnonzero(pitch > max(med * 3.0, gmax * 3.0)) + 1
        for piece in np.split(idx, cuts):
            if len(piece) < MIN_GLYPHS_PER_ROW:
                continue
            boxes = arr[piece].astype(int)
            found.append({
                "x0": int(boxes[:, 0].min()),
                "y0": int(boxes[:, 1].min()),
                "x1": int((boxes[:, 0] + boxes[:, 2]).max()),
                "y1": int((boxes[:, 1] + boxes[:, 3]).max()),
                "n_glyphs": int(len(piece)),
                "pitch": float(np.median(np.diff(cx[piece]))) if len(piece) > 1 else med,
                "glyphs": [tuple(int(v) for v in b) for b in boxes],
            })
    return found


# windows searched for overlay text, as (y0, y1, x0, x1) fractions of the image
WINDOWS = {
    "left":   (0.00, 0.45, 0.00, 0.40),
    "right":  (0.00, 0.40, 0.60, 1.00),
    "bottom": (0.80, 1.00, 0.00, 1.00),
}


def find_all(img: np.ndarray, windows: dict | None = None) -> list[dict]:
    """Text rows anywhere in `img`, in whole-image pixel coordinates."""
    h, w = img.shape[:2]
    out = []
    for name, (fy0, fy1, fx0, fx1) in (windows or WINDOWS).items():
        y0, x0 = int(h * fy0), int(w * fx0)
        reg = img[y0:int(h * fy1), x0:int(w * fx1)]
        for r in rows(reg, h):
            r = dict(r)
            r["window"] = name
            r["x0"] += x0
            r["x1"] += x0
            r["y0"] += y0
            r["y1"] += y0
            r["glyphs"] = [(gx + x0, gy + y0, gw, gh)
                           for gx, gy, gw, gh in r["glyphs"]]
            out.append(r)
    return out


# ---------------------------------------------------------------------------
# Overlay blocks
# ---------------------------------------------------------------------------
# Ribs and clavicles can fake a single row of evenly spaced bright blobs, so a
# lone row is not trusted. The console overlay is a *block*: several rows that
# share an alignment edge and a constant line pitch. That signature anatomy does
# not produce, and it is what gets blanked.

def _aligned(rows_, edge, tol):
    return np.std([r[edge] for r in rows_]) <= tol


def blocks(rows_: list[dict], img_h: int, img_w: int) -> list[dict]:
    """Group text rows into left- or right-aligned multi-line overlay blocks."""
    _, gmax = _glyph_bounds(img_h)
    out = []
    for edge in ("x0", "x1"):
        pool = sorted([r for r in rows_ if r["window"] != "bottom"],
                      key=lambda r: r["y0"])
        used = set()
        for i, r in enumerate(pool):
            if i in used:
                continue
            group, last = [r], r
            for j in range(i + 1, len(pool)):
                if j in used:
                    continue
                c = pool[j]
                if abs(c[edge] - last[edge]) > gmax * 1.5:
                    continue
                gap = c["y0"] - last["y0"]
                if not (gmax * 0.8 <= gap <= gmax * 3.2):
                    continue
                if group[1:]:
                    prev = group[-1]["y0"] - group[-2]["y0"]
                    if abs(gap - prev) > gmax * 0.9:
                        continue
                group.append(c)
                used.add(j)
                last = c
            if len(group) >= 2 and _aligned(group, edge, gmax * 1.2):
                used.add(i)
                out.append({
                    "edge": edge,
                    "n_rows": len(group),
                    "x0": min(g["x0"] for g in group),
                    "x1": max(g["x1"] for g in group),
                    "y0": min(g["y0"] for g in group),
                    "y1": max(g["y1"] for g in group),
                    "rows": group,
                })
    return out


# The console burns its overlay in at one fixed grey level. Measured over 200
# random primary-cohort originals, 801 of 851 low-variance rows sit in the
# 150-155 bin, and the per-row spread of glyph brightness is ~1 grey level.
# Anatomy that happens to line up into an evenly spaced row does not do that:
# its spread is ~7 and its level ranges over 135-203. That is the separator.
LEVEL_SD_MAX = 3.0          # grey levels, within one row
LEVEL_BAND = (140.0, 168.0)  # plausible rendered level
LEVEL_TOL = 6.0             # slack around this image's own overlay level


def row_level(img: np.ndarray, row: dict) -> tuple[float, float] | None:
    """(mean, sd) of per-glyph brightness for one row, or None if unmeasurable."""
    vals = []
    for gx, gy, gw, gh in row["glyphs"]:
        patch = img[gy:gy + gh, gx:gx + gw]
        if patch.size:
            vals.append(float(np.percentile(patch, 90)))
    if len(vals) < MIN_GLYPHS_PER_ROW:
        return None
    return float(np.mean(vals)), float(np.std(vals))


def overlay_rows(img: np.ndarray) -> list[dict]:
    """Text rows whose brightness signature marks them as burned-in overlay."""
    rs = find_all(img)
    lv = {id(r): row_level(img, r) for r in rs}

    strict = [r for r in rs if lv[id(r)] and lv[id(r)][1] <= LEVEL_SD_MAX
              and LEVEL_BAND[0] <= lv[id(r)][0] <= LEVEL_BAND[1]]
    if not strict:
        return []

    # this image's own overlay level, then a second pass that also accepts rows
    # the fixed mask has already clipped (fewer glyphs, noisier)
    level = float(np.median([lv[id(r)][0] for r in strict]))
    cand = [r for r in rs if lv[id(r)]
            and abs(lv[id(r)][0] - level) <= LEVEL_TOL
            and lv[id(r)][1] <= LEVEL_SD_MAX * 2]

    # Anatomy occasionally lines up *and* happens to sit at the overlay level.
    # Real overlay is either stacked into an aligned block or pinned to a margin,
    # so require one of those as well.
    h, w = img.shape[:2]
    in_block = set()
    for b in blocks(cand, h, w):
        for r in b["rows"]:
            in_block.add((r["x0"], r["y0"]))

    def margin(r):
        return (r["x1"] <= w * 0.30 or r["x0"] >= w * 0.70
                or r["y1"] <= h * 0.15 or r["y0"] >= h * 0.85)

    return [r for r in cand if (r["x0"], r["y0"]) in in_block or margin(r)]


def overlay_boxes(img: np.ndarray, pad_frac: float = 0.6) -> list[tuple]:
    """Rectangles to blank, in whole-image pixel coordinates.

    `pad_frac` is padding in units of the maximum glyph height.
    """
    h, w = img.shape[:2]
    _, gmax = _glyph_bounds(h)
    pad = int(round(gmax * pad_frac))
    return [(max(0, r["x0"] - pad), max(0, r["y0"] - pad),
             min(w, r["x1"] + pad), min(h, r["y1"] + pad))
            for r in overlay_rows(img)]
