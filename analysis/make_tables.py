#!/usr/bin/env python3
"""Emit complete LaTeX tabulars from the result files, so no number is typed.

Each fragment owns \\begin{tabular} ... \\end{tabular}: TeX's alignment scanner
cannot look across an \\input boundary for \\noalign or \\omit, so \\midrule,
\\bottomrule and a leading \\multicolumn all break if only rows are inputted.
Captions and labels stay in the manuscript.
"""

from pathlib import Path

import pandas as pd

RES = Path("/scratch/hl106/80_workspace/foye/paper/results")
FOYE = Path("/scratch/hl106/80_workspace/foye")
TAB = Path("/scratch/hl106/80_workspace/foye/paper/tables")

ENCODERS = ["BiomedCLIP", "TorchXRayVision", "RAD-DINO", "ImageNet-DenseNet121"]
STRATA = ["全队列(参照)", "DX 模态全体", "CR 模态全体", "3001x3001 单机(方形探测器)"]


def w(name, body, colspec, header):
    TAB.mkdir(parents=True, exist_ok=True)
    head = "\n".join(header) if isinstance(header, (list, tuple)) else header
    (TAB / f"{name}.tex").write_text(
        "\\begin{tabular}{" + colspec + "}\n\\toprule\n"
        + head.rstrip("\n") + "\n\\midrule\n"
        + body.rstrip("\n") + "\n\\bottomrule\n\\end{tabular}\n")
    print("[tab]", name)


def f3(x):
    return f"{x:.3f}"


def ci(mean, lo, hi):
    return f"{mean:.3f} ({lo:.3f}--{hi:.3f})"


def sgn(x, nd=3):
    return ("$+$" if x >= 0 else "$-$") + f"{abs(x):.{nd}f}"


# ---------------------------------------------------------------- Table 1
def table1_cohort():
    u = pd.read_csv(RES / "univariate_proxy.csv")
    NICE = {
        "modality_cr": ("CR acquisition mode, n (\\%)", "pct"),
        "image_width_px": ("Image width, px", "int"),
        "image_height_px": ("Image height, px", "int"),
        "thorax_p10_intensity": ("Thoracic 10th-percentile intensity", "f1"),
        "thorax_mean_intensity": ("Mean thoracic intensity", "f1"),
        "sex_male": ("Male sex, n (\\%)", "pct"),
        "body_bbox_area_frac": ("Body bounding-box area, fraction", "f3"),
        "shoulder_width_frac": ("Shoulder width, fraction", "f3"),
        "thorax_width_frac": ("Thoracic width, fraction", "f3"),
        "exam_days_since_min": ("Days since first study in cohort", "f0"),
    }
    m = pd.read_csv("/scratch/hl106/foye/cohort_1960/proxy_metadata.csv")
    n_h, n_n = int((m.label == 1).sum()), int((m.label == 0).sum())
    lines = []
    for key, (label, fmt) in NICE.items():
        r = u[u.feature == key].iloc[0]

        def show(mean, sd, n):
            if fmt == "pct":
                return f"{int(round(mean * n))} ({mean * 100:.1f})"
            if fmt in ("int", "f0"):
                return f"{mean:.0f} $\\pm$ {sd:.0f}"
            if fmt == "f1":
                return f"{mean:.1f} $\\pm$ {sd:.1f}"
            return f"{mean:.3f} $\\pm$ {sd:.3f}"

        lines.append(f"{label} & {show(r.hernia_mean, r.hernia_sd, n_h)} & "
                     f"{show(r.nonhernia_mean, r.nonhernia_sd, n_n)} & "
                     f"{sgn(r.smd, 2)} & {ci(r.auc, r.auc_ci_lo, r.auc_ci_hi)} \\\\")
    w("table1_cohort", "\n".join(lines), "lccc c",
      [r"Variable & Hernia & Non-hernia & SMD & AUC (95\% CI) \\",
       rf"& (n$=${n_h}) & (n$=${n_n}) & & \\"])


# ---------------------------------------------------------------- Table 2
def table2_discrimination():
    d = pd.read_csv(RES / "pathway_auc.csv")
    e = pd.read_csv(RES / "external_validation.csv").set_index("model_display")
    lines = []
    for m in ENCODERS:
        r = d[(d.pathway == "image") & (d.nuisance_set == "none") & (d.model_display == m)].iloc[0]
        x = e.loc[m]
        lines.append(f"{m} & {f3(r.auc_mean)} $\\pm$ {f3(r.auc_sd)} & "
                     f"{ci(r.auc_oof_mean, r.auc_oof_ci_lo, r.auc_oof_ci_hi)} & "
                     f"{f3(r.bacc_mean)} & {ci(x.auc, x.ci_lo, x.ci_hi)} \\\\")
    n_all = len(pd.read_csv("/scratch/hl106/foye/cohort_1960/proxy_metadata.csv"))
    w("table2_discrimination", "\n".join(lines), "lcccc",
      [rf"& \multicolumn{{3}}{{c}}{{Primary cohort (n$=${n_all})}} & External (n$=$100) \\",
       r"\cmidrule(lr){2-4}\cmidrule(lr){5-5}",
       r"Encoder & Per-fold AUC & AUC (95\% CI) & Balanced acc. & AUC (95\% CI) \\"])


# ---------------------------------------------------------------- Table 3
def table3_sensitivity():
    d = pd.read_csv(RES / "pathway_auc.csv")
    c = pd.read_csv(RES / "comparisons_c.csv")
    order = [("none", "No adjustment"), ("S", "Demographic"),
             ("A", "Acquisition"), ("A+G+E+S", "All 29 proxies")]
    lines = []
    for m in ENCODERS:
        s = d[(d.pathway == "image") & (d.model_display == m)].set_index("nuisance_set")
        cells = " & ".join(f3(s.loc[k, "auc_oof_mean"]) for k, _ in order)
        row = c[c.comparison == f"{m}: raw - acquisition-adjusted"].iloc[0]
        lines.append(f"{m} & {cells} & {sgn(-row.delta_auc)} \\\\")
    w("table3_sensitivity", "\n".join(lines), "lccccc",
      [r"& No & Demographic & Acquisition & All 29 proxies & $\Delta$ from \\",
       r"Encoder & adjustment & (sex) & (5 attributes) & (over-adjusted) & acquisition \\"])


def table3b_comparisons():
    c = pd.read_csv(RES / "comparisons_c.csv")
    keep = c[c.comparison.str.contains("acquisition-only proxy")]
    lines = [f"{r.comparison.replace(' - acquisition-only proxy', '')} & {sgn(r.delta_auc)} & "
             f"({sgn(r.ci_lo)} to {sgn(r.ci_hi)}) & "
             f"{'$<$0.001' if r.p_bootstrap < 0.001 else f'{r.p_bootstrap:.3f}'} \\\\"
             for _, r in keep.iterrows()]
    w("table3b_comparisons", "\n".join(lines), "lccc",
      [r"Acquisition-adjusted encoder, minus an & & & \\",
       r"acquisition-only model that sees no pixels & $\Delta$AUC & 95\% CI & $p$ \\"])


# ---------------------------------------------------------------- Table 4
def table4_device():
    s = pd.read_csv(RES / "device_stratified.csv")
    LAB = {STRATA[0]: "Pooled cohort",
           STRATA[1]: "Digital radiography",
           STRATA[2]: "Computed radiography",
           STRATA[3]: "Fixed-matrix stratum"}
    lines = []
    for st in STRATA:
        sub = s[s.stratum == st]
        n = int(sub.n.iloc[0])
        nh = int(sub.n_hernia.iloc[0])
        cells = " & ".join(
            f"{sub[sub.model == m].auc.iloc[0]:.3f} ({sub[sub.model == m].ci_lo.iloc[0]:.2f}--"
            f"{sub[sub.model == m].ci_hi.iloc[0]:.2f})" for m in ENCODERS)
        lines.append(f"{LAB[st]} & {n} ({nh}/{n - nh}) & {cells} \\\\")
    w("table4_device", "\n".join(lines), "llcccc",
      [r"& n & \multicolumn{4}{c}{AUC (95\% CI)} \\", r"\cmidrule(lr){3-6}",
       r"Stratum & (hernia/ctrl) & BiomedCLIP & TorchXRayVis. & RAD-DINO & ImageNet-DN121 \\"])


# ---------------------------------------------------------------- supplement
def tableS_recoverability():
    d = pd.read_csv(RES / "recoverability_summary.csv")
    keys = ["biomedclip", "torchxrayvision", "rad-dino", "imagenet"]
    lines = []
    for task, g in sorted(d.groupby("task"), key=lambda kv: -kv[1].primary_value.max()):
        metric = "AUC" if g.task_type.iloc[0] == "classification" else "$R^2$"
        cells = []
        for k in keys:
            v = g[g.model == k].primary_value.iloc[0]
            cells.append(f3(v) if v >= 0 else sgn(v))
        safe = task.replace("_", r"\_")
        lines.append(f"\\texttt{{{safe}}} & {metric} & " + " & ".join(cells) + r" \\")
    w("tableS1_recoverability", "\n".join(lines), "llcccc",
      r"Target variable & Metric & BiomedCLIP & TorchXRayVision & RAD-DINO & ImageNet-DN121 \\")


def tableS_masking():
    m = pd.read_csv(RES / "masking_check.csv")
    lines = []
    for e in ENCODERS:
        a = m[(m.condition == "unmasked") & (m.model_display == e)].iloc[0]
        b = m[(m.condition == "annotation-masked") & (m.model_display == e)].iloc[0]
        lines.append(f"{e} & {ci(a.auc_oof, a.auc_ci_lo, a.auc_ci_hi)} & "
                     f"{ci(b.auc_oof, b.auc_ci_lo, b.auc_ci_hi)} & "
                     f"{sgn(b.auc_oof - a.auc_oof)} \\\\")
    w("tableS2_masking", "\n".join(lines), "lccc",
      r"Encoder & As delivered, AUC (95\% CI) & Annotations masked, AUC (95\% CI) & $\Delta$ \\")


def tableS_vlm():
    v = pd.read_csv(RES / "vlm_summary.csv")
    models = ["MedGemma-4B-IT", "HuatuoGPT-V-7B", "InternVL", "Qwen2.5-VL",
              "LLaVA-OneVision", "SmolVLM"]
    lines = []
    for m in models:
        cells = []
        for cond in ("zero_shot", "ft"):
            for ts in ("foye_hernia", "foye_center2"):
                r = v[(v.model_display == m) & (v.condition == cond) & (v.test_set == ts)].iloc[0]
                mark = "$^{\\dagger}$" if r.single_class_output else ""
                cells += [f"{f3(r.balanced_accuracy)}{mark}", f3(r.auc)]
        lines.append(f"{m} & " + " & ".join(cells) + r" \\")
    w("tableS3_vlm", "\n".join(lines), "lcccccccc",
      [r"& \multicolumn{4}{c}{Zero-shot} & \multicolumn{4}{c}{QLoRA fine-tuned} \\",
       r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
       r"& \multicolumn{2}{c}{Internal} & \multicolumn{2}{c}{External} &",
       r"  \multicolumn{2}{c}{Internal} & \multicolumn{2}{c}{External} \\",
       r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
       r"Model & BAcc & AUC & BAcc & AUC & BAcc & AUC & BAcc & AUC \\"])


def tableS_proxy_models():
    d = pd.read_csv(RES / "pathway_auc.csv")
    LAB = {"acquisition": "Acquisition attributes", "body_geometry": "Body geometry",
           "thorax_intensity_lucency": "Thoracic exposure", "sex": "Sex",
           "all_proxy_factors": "All proxy variables"}
    lines = []
    for k in ["acquisition", "thorax_intensity_lucency", "body_geometry", "sex",
              "all_proxy_factors"]:
        r = d[(d.pathway == "proxy") & (d.model_display == k)].iloc[0]
        lines.append(f"{LAB[k]} & {int(r.n_nuisance)} & "
                     f"{ci(r.auc_oof_mean, r.auc_oof_ci_lo, r.auc_oof_ci_hi)} & "
                     f"{f3(r.bacc_mean)} \\\\")
    w("tableS4_proxy_models", "\n".join(lines), "lccc",
      r"Variable set (no pixel data) & n & AUC (95\% CI) & Balanced acc. \\")


if __name__ == "__main__":
    table1_cohort()
    table2_discrimination()
    table3_sensitivity()
    table3b_comparisons()
    table4_device()
    tableS_recoverability()
    tableS_masking()
    tableS_vlm()
    tableS_proxy_models()
    print("[done]", TAB)
