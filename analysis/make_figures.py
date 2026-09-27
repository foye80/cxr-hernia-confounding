#!/usr/bin/env python3
"""
Publication figures.

Framing: a reproducible signal is present; its origin is not resolved by the
metadata available. The figures therefore report the observation, each
sensitivity layer, and the device-stratified control that limits interpretation.

Palette: dataviz reference categorical slots 1/2/3/7, validated all-pairs on a
white surface with analysis/validate_palette.py (worst CVD dE 9.2, worst
normal-vision dE 16.3). Aqua carries a contrast WARN against white, so every
aqua mark is directly labelled.

Outputs vector PDF plus 600-dpi PNG into paper/figures/.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

RES = Path("/scratch/hl106/80_workspace/foye/paper/results")
FIG = Path("/scratch/hl106/80_workspace/foye/paper/figures")
FOYE = Path("/scratch/hl106/80_workspace/foye")


# Cohort counts come from the analysed proxy table, never from literals, so the
# figures follow the cohort when it changes (1960 images since the revision).
COHORT_CSV = Path("/scratch/hl106/foye/cohort_1960/proxy_metadata.csv")


def cohort_counts():
    m = pd.read_csv(COHORT_CSV)
    dims = (m.image_width_px.astype(int).astype(str) + "x"
            + m.image_height_px.astype(int).astype(str))
    sq = dims == "3001x3001"
    return dict(n=len(m), n_h=int((m.label == 1).sum()), n_c=int((m.label == 0).sum()),
                n_dx=int((m.modality_cr == 0).sum()), n_cr=int((m.modality_cr == 1).sum()),
                n_sq=int(sq.sum()), n_sq_h=int((sq & (m.label == 1)).sum()),
                n_sq_c=int((sq & (m.label == 0)).sum()), n_dims=int(dims.nunique()))

BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
INK, MUTED, GRID = "#1a1a1a", "#5b5b5b", "#d8dade"
CHANCE = "#b01c1c"

ENCODERS = ["BiomedCLIP", "TorchXRayVision", "RAD-DINO", "ImageNet-DenseNet121"]
ENC_SHORT = {"BiomedCLIP": "BiomedCLIP", "TorchXRayVision": "TorchXRayVision",
             "RAD-DINO": "RAD-DINO", "ImageNet-DenseNet121": "ImageNet-DN121"}
ENC_COLOR = dict(zip(ENCODERS, [BLUE, ORANGE, AQUA, VIOLET]))

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.7,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.labelcolor": INK,
    "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
MM = 1 / 25.4


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=600)
    plt.close(fig)
    print("[fig]", name)


def tidy(ax, xlabel=None, grid_axis="x"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    if xlabel:
        ax.set_xlabel(xlabel)


def ci_bar(ax, y, mean, lo, hi, color, height=0.6, x0=0.5, fs=7.0):
    ax.barh(y, mean - x0, left=x0, height=height, color=color, zorder=3, linewidth=0)
    ax.plot([lo, hi], [y, y], color="#ffffff", linewidth=2.8, zorder=4, solid_capstyle="butt")
    ax.errorbar(mean, y, xerr=[[mean - lo], [hi - mean]], fmt="none", ecolor=INK,
                elinewidth=0.9, capsize=2.0, capthick=0.9, zorder=5)
    ax.text(hi + 0.006, y, f"{mean:.3f} [{lo:.3f}–{hi:.3f}]", va="center",
            fontsize=fs, color=INK)


# ---------------------------------------------------------------- Figure 1
def fig1_design():
    fig, ax = plt.subplots(figsize=(180 * MM, 104 * MM))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    ax.axis("off")
    TITLE_H, LINE_H, PAD = 3.6, 2.65, 1.5

    def box(x, ytop, w, title, lines, edge, fill, align="center"):
        h = TITLE_H + LINE_H * len(lines) + 2 * PAD
        y = ytop - h
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.35,rounding_size=1.0",
                                    linewidth=1.0, edgecolor=edge, facecolor=fill, zorder=2))
        ax.text(x + w / 2, ytop - PAD - TITLE_H / 2, title, ha="center", va="center",
                fontsize=8.0, fontweight="bold", color=edge, zorder=3)
        tx = x + w / 2 if align == "center" else x + 1.6
        ha = "center" if align == "center" else "left"
        fs = 6.6 if align == "center" else 6.3
        for i, ln in enumerate(lines):
            ax.text(tx, ytop - PAD - TITLE_H - LINE_H * (i + 0.5), ln, ha=ha, va="center",
                    fontsize=fs, color=INK, zorder=3)
        return y

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=7,
                                     linewidth=0.9, color=MUTED, zorder=1, shrinkA=0, shrinkB=0))

    C = cohort_counts()
    box(0.5, 59.0, 26, "Primary cohort", [
        "one center, 2024 to 2026",
        "2089 admissions, 69 postponed",
        "2020 records",
        "audit: one child, one image,",
        "one record, label agrees",
        f"{C['n']} radiographs",
        f"{C['n_h']} hernia / {C['n_c']} controls",
        "burned-in text masked"], INK, "#f4f5f7")
    box(0.5, 20.0, 26, "External center", [
        "100 radiographs (50 / 50)", "screen captures,", "no acquisition metadata"],
        INK, "#f4f5f7")

    box(31.0, 59.0, 33, "1  Apparent discrimination", [
        "4 frozen encoders, linear probe",
        "5-fold x 5-repeat CV; external test"], BLUE, "#eaf2fc")
    box(31.0, 45.0, 33, "2  Adjustment by regression", [
        "within each training fold:",
        "sex; acquisition (5); age;",
        "all 29 variables (lower bound)"], ORANGE, "#fdefe8")
    box(31.0, 28.8, 33, "3  Matching (age legible)", [
        "age within 3 months",
        "+ CR/DX mode and image size",
        "+ sex"], VIOLET, "#efedf8")
    box(31.0, 12.6, 33, "4  Fixed-matrix stratum", [
        f"3001x3001, n = {C['n_sq']}",
        f"({C['n_sq_h']} hernia / {C['n_sq_c']} controls)"], AQUA, "#e7f7f1")

    box(66.0, 59.0, 33.5, "What each step can show", [
        "1  whether the groups can be told apart",
        "2  how much each factor accounts for;",
        "    linear adjustment can leave residue",
        "3  whether anything survives when",
        "    the groups are made alike by design",
        "4  equipment held fixed, but small:",
        "    only large effects are detectable"], INK, "#f4f5f7", align="left")
    box(66.0, 31.0, 33.5, "Limits", [
        "no DICOM headers: acquisition is",
        "proxied by image-derived variables",
        "age legible for part of the cohort",
        "controls are other surgical patients"], INK, "#f4f5f7", align="left")

    arrow(27.0, 45.0, 30.7, 52.0)
    arrow(27.0, 12.7, 30.7, 50.0)
    for ytop, ynext in ((47.1, 45.0), (30.45, 28.8), (14.25, 12.6)):
        arrow(47.5, ytop, 47.5, ynext)
    save(fig, "figure1_design")


# ---------------------------------------------------------------- Figure 2
def fig2_discrimination():
    d = pd.read_csv(RES / "pathway_auc.csv")
    ext = pd.read_csv(RES / "external_validation.csv").set_index("model_display")
    fig, axes = plt.subplots(1, 2, figsize=(180 * MM, 62 * MM),
                             gridspec_kw={"wspace": 0.46, "left": 0.15, "right": 0.99,
                                          "top": 0.86, "bottom": 0.22})
    y = np.arange(len(ENCODERS))[::-1]

    ax = axes[0]
    for yi, m in zip(y, ENCODERS):
        r = d[(d.pathway == "image") & (d.nuisance_set == "none") & (d.model_display == m)].iloc[0]
        ci_bar(ax, yi, r.auc_oof_mean, r.auc_oof_ci_lo, r.auc_oof_ci_hi, ENC_COLOR[m])
    ax.axvline(0.5, color=CHANCE, linewidth=1.1, zorder=6)
    ax.set_yticks(y)
    ax.set_yticklabels([ENC_SHORT[m] for m in ENCODERS])
    ax.set_xlim(0.49, 0.85)
    ax.set_xticks([0.5, 0.6, 0.7, 0.8])
    ax.set_title(f"(a)  Primary cohort (n = {cohort_counts()['n']}), cross-validated",
                 loc="left", fontweight="bold", fontsize=8.0)
    tidy(ax, "AUC (95% CI)")
    ax.tick_params(axis="y", length=0)

    ax = axes[1]
    for yi, m in zip(y, ENCODERS):
        r = ext.loc[m]
        ci_bar(ax, yi, r.auc, r.ci_lo, r.ci_hi, ENC_COLOR[m])
    ax.axvline(0.5, color=CHANCE, linewidth=1.1, zorder=6)
    ax.set_yticks(y)
    ax.set_yticklabels([ENC_SHORT[m] for m in ENCODERS])
    ax.set_xlim(0.36, 1.06)
    ax.set_xticks([0.4, 0.6, 0.8, 1.0])
    ax.set_title("(b)  External center (n = 100)", loc="left", fontweight="bold", fontsize=8.0)
    tidy(ax, "AUC (95% CI)")
    ax.tick_params(axis="y", length=0)
    save(fig, "figure2_discrimination")


# ---------------------------------------------------------------- Figure 3
def fig3_sensitivity():
    d = pd.read_csv(RES / "pathway_auc.csv")
    order = ["none", "S", "A", "A+G+E+S"]
    xlab = ["No\nadjustment", "Demographic\n(sex)", "Acquisition\n(5 attributes)",
            "All 29 proxies\n(over-adjusted)"]
    fig, ax = plt.subplots(figsize=(150 * MM, 82 * MM))
    x = np.arange(len(order))

    ax.axvspan(2.55, 3.6, color="#f2f2f4", zorder=0)
    ax.text(3.08, 0.783, "includes body size,\na candidate mediator", ha="center", va="top",
            fontsize=6.8, color=MUTED)

    ends = []
    for m in ENCODERS:
        s = d[(d.pathway == "image") & (d.model_display == m)].set_index("nuisance_set")
        yv = [s.loc[k, "auc_oof_mean"] for k in order]
        el = [s.loc[k, "auc_oof_mean"] - s.loc[k, "auc_oof_ci_lo"] for k in order]
        eh = [s.loc[k, "auc_oof_ci_hi"] - s.loc[k, "auc_oof_mean"] for k in order]
        ax.errorbar(x, yv, yerr=[el, eh], color=ENC_COLOR[m], linewidth=2.0, marker="o",
                    markersize=4.5, capsize=2.2, elinewidth=0.8, zorder=3,
                    markeredgecolor="#ffffff", markeredgewidth=0.7)
        ends.append([s.loc["A", "auc_oof_mean"], m])

    ends.sort()
    GAP = 0.018
    for i in range(1, len(ends)):
        if ends[i][0] - ends[i - 1][0] < GAP:
            ends[i][0] = ends[i - 1][0] + GAP
    for ylab, m in ends:
        s = d[(d.pathway == "image") & (d.model_display == m)].set_index("nuisance_set")
        yv = s.loc["A", "auc_oof_mean"]
        ax.plot([2.05, 2.14], [yv, ylab], color=ENC_COLOR[m], linewidth=0.7, zorder=3)
        ax.text(2.18, ylab, f"{ENC_SHORT[m]}  {yv:.3f}", color=ENC_COLOR[m], fontsize=7.0,
                va="center", fontweight="bold")

    ax.axhline(0.5, color=CHANCE, linewidth=1.1, zorder=2)
    ax.text(-0.40, 0.505, "chance", color=CHANCE, fontsize=7.0, va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels(xlab)
    ax.set_xlim(-0.46, 3.6)
    ax.set_ylim(0.49, 0.80)
    ax.set_ylabel("Hernia AUC (95% CI)")
    ax.set_xlabel("Variables removed from the embedding within each training fold")
    tidy(ax, grid_axis="y")
    save(fig, "figure3_sensitivity")


# ---------------------------------------------------------------- Figure 4
def fig4_device():
    s = pd.read_csv(RES / "device_stratified.csv")
    order = ["全队列(参照)", "DX 模态全体",
             "CR 模态全体", "3001x3001 单机(方形探测器)"]
    C = cohort_counts()
    label = {order[0]: f"Pooled cohort\n(n = {C['n']})",
             order[1]: f"Digital radiography only\n(n = {C['n_dx']})",
             order[2]: f"Computed radiography only\n(n = {C['n_cr']})",
             order[3]: f"Fixed-matrix stratum\n(n = {C['n_sq']}; {C['n_sq_h']} / {C['n_sq_c']})"}
    fig, ax = plt.subplots(figsize=(150 * MM, 86 * MM))
    y0 = 0.0
    ticks, ticklabels = [], []
    for st in order:
        sub = s[s.stratum == st]
        ticks.append(y0 - 1.5)
        ticklabels.append(label[st])
        for j, m in enumerate(ENCODERS):
            r = sub[sub.model == m]
            if r.empty:
                continue
            r = r.iloc[0]
            ci_bar(ax, y0 - j, r.auc, r.ci_lo, r.ci_hi, ENC_COLOR[m], height=0.62, fs=6.6)
        y0 -= len(ENCODERS) + 1.4
    ax.axvline(0.5, color=CHANCE, linewidth=1.1, zorder=6)
    ax.set_yticks(ticks)
    ax.set_yticklabels(ticklabels, fontsize=7.4)
    ax.set_ylim(y0 + 0.6, 1.0)
    ax.set_xlim(0.34, 0.90)
    ax.set_xticks([0.4, 0.5, 0.6, 0.7, 0.8])
    tidy(ax, "AUC (95% CI)")
    ax.tick_params(axis="y", length=0)
    ax.legend(handles=[Line2D([], [], marker="s", linestyle="", markersize=6,
                              color=ENC_COLOR[m], label=ENC_SHORT[m]) for m in ENCODERS],
              loc="lower right", bbox_to_anchor=(1.0, 1.005), ncol=4, frameon=False,
              handletextpad=0.4, columnspacing=1.1, fontsize=7.0)
    save(fig, "figure4_device")


# ---------------------------------------------------------------- Supplement
def figS_recoverability():
    d = pd.read_csv(RES / "recoverability_summary.csv")
    NICE = {"modality_cr": "CR vs DR acquisition mode*",
            "body_bbox_area_frac": "Body bounding-box area",
            "body_bbox_height_frac": "Body bounding-box height",
            "body_bbox_width_frac": "Body bounding-box width",
            "body_mask_area_frac": "Body mask area",
            "shoulder_width_frac": "Shoulder width",
            "thorax_width_frac": "Thoracic width",
            "image_height_px": "Image height (pixels)",
            "thorax_mean_intensity": "Mean thoracic intensity",
            "image_width_px": "Image width (pixels)",
            "thorax_p10_intensity": "Thoracic 10th-percentile intensity",
            "sex_male": "Sex*",
            "exam_days_since_min": "Examination date"}
    best = (d[d.task.isin(NICE)].sort_values("primary_value", ascending=False)
            .groupby("task", as_index=False).first())
    best["label"] = best.task.map(NICE)
    best = best.sort_values("primary_value")
    fig, ax = plt.subplots(figsize=(120 * MM, 76 * MM))
    y = np.arange(len(best))
    ax.barh(y, best.primary_value, height=0.62, color=BLUE, zorder=3, linewidth=0)
    for yi, (_, r) in zip(y, best.iterrows()):
        sub = d[d.task == r.task]
        ax.scatter(sub.primary_value, [yi] * len(sub), s=9, color="#ffffff",
                   edgecolor=INK, linewidth=0.6, zorder=5)
        ax.text(r.primary_value + 0.012, yi, f"{r.primary_value:.3f}", va="center",
                fontsize=7.0, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(best.label)
    ax.set_xlim(0, 1.10)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_xlabel("Recoverability from the frozen embedding\n"
                  "(AUC for binary targets*, out-of-fold $R^2$ otherwise)")
    tidy(ax)
    ax.tick_params(axis="y", length=0)
    ax.legend(handles=[
        Line2D([], [], marker="s", linestyle="", markersize=6, color=BLUE,
               label="best of 4 encoders"),
        Line2D([], [], marker="o", linestyle="", markersize=4, markerfacecolor="#ffffff",
               markeredgecolor=INK, color=INK, label="individual encoders")],
        loc="lower right", bbox_to_anchor=(1.0, 1.005), ncol=2, frameon=False,
        handletextpad=0.4, columnspacing=1.4)
    save(fig, "figureS1_recoverability")


def figS_controls():
    fig, axes = plt.subplots(1, 2, figsize=(150 * MM, 78 * MM),
                             gridspec_kw={"width_ratios": [1.0, 1.0], "wspace": 0.62,
                                          "top": 0.80, "bottom": 0.26,
                                          "left": 0.16, "right": 0.985})
    ax = axes[0]
    m = pd.read_csv(RES / "masking_check.csv")
    y = np.arange(len(ENCODERS))[::-1]
    for off, cond, color, lab in [(0.19, "unmasked", BLUE, "as delivered"),
                                  (-0.19, "annotation-masked", ORANGE, "annotations masked")]:
        vals = [m[(m.condition == cond) & (m.model_display == e)].auc_oof.iloc[0] for e in ENCODERS]
        ax.barh(y + off, [v - 0.5 for v in vals], left=0.5, height=0.34, color=color,
                zorder=3, linewidth=0, label=lab)
        for yi, v in zip(y + off, vals):
            ax.text(v + 0.004, yi, f"{v:.3f}", va="center", fontsize=6.6, color=INK)
    ax.axvline(0.5, color=CHANCE, linewidth=1.1, zorder=5)
    ax.set_yticks(y)
    ax.set_yticklabels([ENC_SHORT[e] for e in ENCODERS], fontsize=7.0)
    ax.set_xlim(0.47, 0.715)
    ax.set_xticks([0.5, 0.6, 0.7])
    ax.set_title("(a)  Burned-in annotation masking\n      (independent 609-image cohort)",
                 loc="left", fontweight="bold", fontsize=7.8)
    ax.set_xlabel("AUC")
    tidy(ax)
    ax.tick_params(axis="y", length=0)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, -0.22), fontsize=6.8,
              handlelength=1.2, handletextpad=0.5)

    ax = axes[1]
    v = pd.read_csv(RES / "vlm_summary.csv")
    models = sorted(v.model_display.unique())
    y = np.arange(len(models))[::-1]
    internal = [v[(v.model_display == mm) & (v.condition == "ft")
                  & (v.test_set == "foye_hernia")].balanced_accuracy.iloc[0] for mm in models]
    external = [v[(v.model_display == mm) & (v.condition == "ft")
                  & (v.test_set == "foye_center2")].balanced_accuracy.iloc[0] for mm in models]
    ax.barh(y, [x - 0.5 for x in internal], left=0.5, height=0.46, color=BLUE,
            zorder=3, linewidth=0, label="fine-tuned, internal test")
    for yi, x in zip(y, internal):
        ax.text(x + 0.005, yi, f"{x:.3f}", va="center", fontsize=6.6, color=INK)
    ax.scatter(external, y, s=26, marker="D", color=ORANGE, zorder=6, edgecolor="#ffffff",
               linewidth=0.7,
               label="fine-tuned, external center\n(all 0.500: single-class output)")
    ax.axvline(0.5, color=CHANCE, linewidth=1.1, zorder=5)
    ax.set_yticks(y)
    ax.set_yticklabels(models, fontsize=7.0)
    ax.set_xlim(0.47, 0.715)
    ax.set_xticks([0.5, 0.6, 0.7])
    ax.set_title("(b)  Vision-language models\n      after QLoRA fine-tuning",
                 loc="left", fontweight="bold", fontsize=7.8)
    ax.set_xlabel("Balanced accuracy")
    tidy(ax)
    ax.tick_params(axis="y", length=0)
    h, l = ax.get_legend_handles_labels()
    ax.legend(h[::-1], l[::-1], frameon=False, loc="upper left", bbox_to_anchor=(0.0, -0.22),
              fontsize=6.8, handlelength=1.2, handletextpad=0.5)
    save(fig, "figureS2_controls")


if __name__ == "__main__":
    fig1_design()
    fig2_discrimination()
    fig3_sensitivity()
    fig4_device()
    figS_recoverability()
    figS_controls()
    print("[done]", FIG)
