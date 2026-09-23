#!/usr/bin/env python3
"""Tables for the analyses added in the revision (2026-09).

  tables/table5_matched.tex        main text: matched comparisons
  tables/tableS8_missingness.tex   who has a readable age
  tables/tableS9_nonlinear.tex     linear versus spline adjustment

All numbers are read from results/revision_analyses.json and
results/pathway_auc.csv; nothing is typed in.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

BASE = Path("/scratch/hl106/80_workspace/foye/paper")
RES = BASE / "results"
TAB = BASE / "tables"
ENC = ["BiomedCLIP", "TorchXRayVision", "RAD-DINO", "ImageNet-DenseNet121"]
NICE = {
    "image_height_px": "Image height", "image_width_px": "Image width",
    "image_aspect": "Image aspect ratio", "modality_cr": "CR acquisition mode",
    "mid_body_area_share": "Middle body area share",
    "thorax_p10_intensity": "Thoracic 10th-percentile intensity",
    "thorax_mean_intensity": "Mean thoracic intensity",
    "exam_days_since_min": "Days since first study", "sex_male": "Male sex",
    "label_hernia": "Hernia group",
}


def ci(a, lo, hi):
    return f"{a:.3f} ({lo:.3f}--{hi:.3f})"


def write(name, body, cols, head):
    lines = [f"\\begin{{tabular}}{{{cols}}}", "\\toprule", *head, "\\midrule", body,
             "\\bottomrule", "\\end{tabular}"]
    (TAB / f"{name}.tex").write_text("\n".join(lines) + "\n")
    print("[tab]", name)


def by_model(rows, key):
    return {r["model"]: r for r in rows}


def main():
    R = json.loads((RES / "revision_analyses.json").read_text())

    # ---------------- Table 5: matched comparisons ----------------
    a2, a4, a4s = R["A2_age_matched"], R["A4_age_acquisition_matched"], R["A4b_age_acquisition_sex_matched"]
    rows = []

    def row(label, pairs, u6, d, key="full_cohort_score_on_matched", cikey="ci"):
        m = {r["model"]: r for r in d}
        cells = " & ".join(ci(m[e][key], *m[e][cikey]) for e in ENC)
        return f"{label} & {pairs} ({u6}) & {cells} \\\\"

    rows.append(row("Age ($\\pm$3 months)", a2["pairs"], a2["pairs_case_under_6y"], a2["encoders"]))
    rows.append(row("\\quad case under 6 years", a2["pairs_case_under_6y"], a2["pairs_case_under_6y"],
                    a2["encoders"], "full_cohort_score_on_matched_under6", "ci_under6"))
    rows.append(row("\\quad probe refitted within pairs", a2["pairs"], a2["pairs_case_under_6y"],
                    a2["encoders"], "probe_within_matched", "ci_within"))
    rows.append(row("Age and acquisition", a4["pairs"], a4["pairs_case_under_6y"], a4["encoders"]))
    rows.append(row("Age, acquisition and sex", a4s["pairs"], a4s["pairs_case_under_6y"], a4s["encoders"]))
    write("table5_matched", "\n".join(rows), "lccccc",
          ["Matched on & Pairs (case $<$6 y) & BiomedCLIP & TorchXRayVision & RAD-DINO & ImageNet-DN121 \\\\"])

    # ---------------- Table S8: age missingness ----------------
    a1 = R["A1_age_missingness"]
    L = [f"Readable, hernia group & {a1['readable_hernia'][0]} / {a1['readable_hernia'][1]} "
         f"({100 * a1['rate_hernia']:.1f}\\%) \\\\",
         f"Readable, control group & {a1['readable_control'][0]} / {a1['readable_control'][1]} "
         f"({100 * a1['rate_control']:.1f}\\%) \\\\",
         f"Readable, CR images & {100 * a1['readable_rate_CR']:.1f}\\% \\\\",
         f"Readable, DX images & {100 * a1['readable_rate_DX']:.1f}\\% \\\\"]
    for y, v in a1["readable_rate_by_year"].items():
        L.append(f"Readable, examinations in {y} & {100 * v:.1f}\\% \\\\")
    L.append(f"Unreadable: no overlay text found & {a1['unreadable_no_text_found']} \\\\")
    L.append(f"Unreadable: text found but not read & {a1['unreadable_text_not_read']} \\\\")
    L.append("\\midrule")
    L.append("\\multicolumn{2}{l}{\\emph{Largest differences, readable minus unreadable (SMD)}} \\\\")
    for r in a1["smd"][:6]:
        L.append(f"{NICE.get(r['variable'], r['variable'].replace('_', ' '))} & "
                 f"{r['smd_readable_minus_unreadable']:+.2f} \\\\")
    lab = next(r for r in a1["smd"] if r["variable"] == "label_hernia")
    L.append(f"Hernia group & {lab['smd_readable_minus_unreadable']:+.2f} \\\\")
    L.append("\\midrule")
    L.append("\\multicolumn{2}{l}{\\emph{Predicting readability (AUC, 95\\% CI)}} \\\\")
    p = a1["predict_readable_auc"]
    L.append(f"From the 29 image-derived variables & {ci(p['proxies_29']['auc'], *p['proxies_29']['ci'])} \\\\")
    L.append(f"From group membership alone & {ci(p['label_only']['auc'], *p['label_only']['ci'])} \\\\")
    L.append("\\midrule")
    L.append("\\multicolumn{2}{l}{\\emph{Encoder AUC, readable / unreadable subset}} \\\\")
    for r in a1["encoder_auc_by_subset"]:
        L.append(f"{r['model']} & {r['readable']:.3f} / {r['unreadable']:.3f} \\\\")
    write("tableS8_missingness", "\n".join(L), "lc", ["Quantity & Value \\\\"])

    # ---------------- Table S9: non-linear adjustment ----------------
    d = pd.read_csv(RES / "pathway_auc.csv")
    a5 = R["A5_nonlinear"]
    age = {r["model"]: r for r in a5["age_subset"]}
    full = {(r["model"], r["layer"]): r for r in a5["full_cohort"]}
    L = []
    for e in ENC:
        lin = {k: d[(d.pathway == "image") & (d.model_display == e) & (d.nuisance_set == k)].iloc[0]
               for k in ["A", "A+G"]}
        L.append(f"{e} & {age[e]['age_linear_log']:.3f} & {age[e]['age_spline']:.3f} & "
                 f"{lin['A'].auc_oof_mean:.3f} & {full[(e, 'A')]['spline']:.3f} & "
                 f"{lin['A+G'].auc_oof_mean:.3f} & {full[(e, 'A+G')]['spline']:.3f} \\\\")
    write("tableS9_nonlinear", "\n".join(L), "lcccccc",
          ["& \\multicolumn{2}{c}{Age (readable subset)} & \\multicolumn{2}{c}{Acquisition (5)} "
           "& \\multicolumn{2}{c}{Acquisition + body geometry} \\\\",
           "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}",
           "Encoder & Linear & Spline & Linear & Spline & Linear & Spline \\\\"])


if __name__ == "__main__":
    main()
