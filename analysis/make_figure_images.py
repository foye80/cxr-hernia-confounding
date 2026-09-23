#!/usr/bin/env python3
"""Figure 1: the radiographs themselves, and what was printed on them.

Three things a reader of this paper cannot currently see: what the encoders were
shown, that the de-identification mask left the age line exposed, and how far
apart the two groups are in age.

Panel (a) is a real original with all four burned-in lines painted over here,
including the sex and age lines the audit concerned: the black block marks where
they were, and the point is that it reaches below the published mask. Panels
(b)-(c) use the rebuilt cohort. Panel (d) is the age recovered by `ocr_age.py`.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

FOYE = Path("/scratch/hl106/foye")
RAW = FOYE / "ChestCR"
CLEAN = FOYE / "ChestCR_remasked"
RES = Path("/scratch/hl106/80_workspace/foye/paper/results")
FIG = Path("/scratch/hl106/80_workspace/foye/paper/figures")

BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
INK, MUTED = "#1a1a1a", "#5b5b5b"
RED = "#b01c1c"

# the old fixed mask, from prepare_dataset.py:mask_image
OLD_TL = (0.12, 0.08)
# the rebuilt mask, from results/remask_report.json
NEW_TL = (0.177, 0.150)

# Which original to show in panel (a). The file name is a hospital identifier,
# so it is kept outside the code, in a file that is not part of the public
# release (analysis/figure1_example.txt, one relative path).
EXAMPLE = (Path(__file__).resolve().parent / "figure1_example.txt").read_text().strip()

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.7,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.labelcolor": INK,
    "figure.dpi": 150, "pdf.fonttype": 42, "ps.fonttype": 42,
})


def _fills_frame(path, min_frac=0.45):
    """Skip studies collimated to a small island inside a large black frame:
    they are real, and Figure 4 is about them, but they make poor examples."""
    im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if im is None:
        return False
    return float((im > 25).mean()) >= min_frac


def pick_examples(age: pd.DataFrame, n_each=3):
    """Representative images per group, spread across the age range."""
    a = age.dropna(subset=["age_months"])
    out = {}
    for lab in (1, 0):
        folder = "0" if lab == 1 else "1"
        s = a[a.label == lab].sort_values("age_months").reset_index(drop=True)
        picked = []
        for q in np.linspace(0.15, 0.85, n_each):
            i = int(q * (len(s) - 1))
            for j in range(len(s)):                       # walk outwards
                for k in (i + j, i - j):
                    if 0 <= k < len(s) and k not in [p[0] for p in picked] \
                            and _fills_frame(CLEAN / folder / s.filename[k]):
                        picked.append((k, s.iloc[k]))
                        break
                else:
                    continue
                break
        out[lab] = [{"filename": r.filename, "age_months": r.age_months}
                    for _, r in picked]
    return out


def age_band(months, width=36):
    """A three-year band instead of an exact age. Journal policy: an exact age
    printed under an individual radiograph is a potential identifier."""
    lo = int(months // width) * width
    return f"{lo // 12}\u2013{(lo + width) // 12 - 1} y"


def show(ax, img, title=None, sub=None):
    ax.imshow(img, cmap="gray", vmin=0, vmax=255, aspect="equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor("#c8c8c8")
        sp.set_linewidth(0.6)
    if title:
        ax.set_title(title, fontsize=7.5, pad=2.5)
    if sub:
        ax.set_xlabel(sub, fontsize=7, labelpad=2, color=MUTED)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    age = pd.read_csv(RES / "burned_in_age.csv")
    ex = pick_examples(age)

    fig = plt.figure(figsize=(7.2, 5.4))
    gs = fig.add_gridspec(
        3, 6, height_ratios=[1.22, 1.0, 1.0], hspace=0.34, wspace=0.10,
        left=0.065, right=0.985, top=0.905, bottom=0.055)

    def panel(ax, letter, text, dy=1.045):
        ax.text(0.0, dy, f"({letter})  {text}", transform=ax.transAxes,
                fontsize=8.5, ha="left", va="bottom")

    # ---------------- (a) the overlay, and the two masks ----------------
    axa = fig.add_subplot(gs[0, :3])
    im = cv2.imread(str(RAW / EXAMPLE), cv2.IMREAD_GRAYSCALE)
    h, w = im.shape
    ch, cw = int(h * 0.24), int(w * 0.30)
    crop = im[:ch, :cw].copy()
    # Paint over the whole overlay block: hospital number, romanised name, sex
    # and age. The four lines end at 0.098 of the height and 0.093 of the width
    # (analysis above); the block below covers them with a small margin and is
    # deliberately narrower than the published mask, so the only part of it that
    # sticks out of the red rectangle is the part that matters, its lower edge.
    BLANK_W, BLANK_H = 0.115, 0.105
    crop[int(h * 0.005):int(h * BLANK_H), :int(w * BLANK_W)] = 0
    axa.imshow(crop, cmap="gray", vmin=0, vmax=255)
    axa.set_xticks([])
    axa.set_yticks([])
    axa.add_patch(Rectangle((0, 0), OLD_TL[0] * w, OLD_TL[1] * h, fill=False,
                            ec=RED, lw=1.5, ls=(0, (4, 2)), zorder=3))
    axa.add_patch(Rectangle((0, 0), NEW_TL[0] * w, NEW_TL[1] * h, fill=False,
                            ec=AQUA, lw=1.5, zorder=3))
    axa.text(OLD_TL[0] * w + 6, OLD_TL[1] * h - 4, "mask as published",
             color=RED, fontsize=7, va="bottom", ha="left", zorder=4)
    axa.text(NEW_TL[0] * w + 6, NEW_TL[1] * h - 4, "rebuilt mask",
             color=AQUA, fontsize=7, va="bottom", ha="left", zorder=4)
    # The corner is nearly black already, so mark the blanked region rather
    # than relying on the reader seeing black on black. Its lower edge, below
    # the dashed rectangle, is the finding.
    axa.add_patch(Rectangle((0, 0), BLANK_W * w, BLANK_H * h, fc="#ffd166",
                            alpha=0.20, ec="#ffd166", lw=0.8, ls=(0, (1, 1.5)),
                            zorder=2))
    axa.text(int(w * 0.128), int(h * 0.088), "sex and age",
             color="#ffd166", fontsize=7, va="center", ha="left", zorder=4)
    axa.annotate("", xy=(int(w * 0.085), int(h * 0.088)),
                 xytext=(int(w * 0.125), int(h * 0.088)),
                 arrowprops=dict(arrowstyle="->", color="#ffd166", lw=1.0), zorder=4)
    panel(axa, "a", "Overlay on an original")
    axa.set_xlabel("All four overlay lines painted over here for publication",
                   fontsize=7, color=MUTED, labelpad=3, loc="left")

    # ---------------- (b) group age distributions ----------------
    axb = fig.add_subplot(gs[0, 3:])
    a = age.dropna(subset=["age_months"])
    bins = np.arange(0, 200, 8)
    for lab, col, name in [(1, ORANGE, "Hernia"),
                           (0, BLUE, "Control")]:
        v = np.clip(a[a.label == lab].age_months, 0, 195)
        axb.hist(v, bins=bins, color=col, alpha=0.30, edgecolor="none")
        axb.hist(v, bins=bins, histtype="step", color=col, linewidth=1.3,
                 label=f"{name} (n={int((a.label==lab).sum())})")
    axb.set_xlabel("Age at radiograph, months")
    axb.set_ylabel("Images")
    axb.legend(frameon=False, loc="upper right", handlelength=1.4,
               bbox_to_anchor=(1.0, 1.0))
    panel(axb, "b", "Age, read off the overlay")
    axb.spines["top"].set_visible(False)
    axb.spines["right"].set_visible(False)
    aa = json.loads((RES / "age_analysis_published.json").read_text())["age_alone"]
    axb.text(0.03, 0.97, f"age alone\nAUC {aa['auc']:.3f} [{aa['ci'][0]:.3f}, {aa['ci'][1]:.3f}]",
             transform=axb.transAxes,
             ha="left", va="top", fontsize=7.5, color=INK,
             bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#d8dade", lw=0.6))

    # ---------------- (c)-(d) example radiographs ----------------
    for row, (lab, name, col) in enumerate(
            [(1, "Indirect inguinal hernia", ORANGE),
             (0, "Non-hernia surgical control", BLUE)]):
        for j, rec in enumerate(ex[lab]):
            ax = fig.add_subplot(gs[1 + row, j * 2:(j + 1) * 2])
            folder = "0" if lab == 1 else "1"
            img = cv2.imread(str(CLEAN / folder / rec["filename"]),
                             cv2.IMREAD_GRAYSCALE)
            show(ax, img, sub=age_band(rec["age_months"]))
            if j == 0:
                ax.set_ylabel(name, fontsize=7.5, color=col, labelpad=4)
            if row == 0 and j == 0:
                panel(ax, "c", "The rebuilt cohort, across the age range of "
                      "each group", dy=1.055)

    fig.savefig(FIG / "figure1_images.pdf")
    fig.savefig(FIG / "figure1_images.png", dpi=600)
    plt.close(fig)
    print("[fig] figure1_images")


if __name__ == "__main__":
    main()
