#!/usr/bin/env python3
"""Cross-check the numbers asserted in manuscript.tex against the result files.

Fails loudly if a claim in the prose no longer matches what the analysis
produced. Run after any re-analysis.
"""

import json
import sys
from pathlib import Path

import pandas as pd

BASE = Path("/scratch/hl106/80_workspace/foye/paper")
RES = BASE / "results"
FOYE = Path("/scratch/hl106/80_workspace/foye")

# Two scopes. `tex` is the whole submission: manuscript, Additional file 1 and
# every generated table, and it answers "is this number reported anywhere".
# `prose` is the two hand-written files only. A number that the prose quotes has
# to be checked against `prose`, because a correct table can otherwise hide a
# wrong sentence: that is how a mis-rounded interval and a mis-attributed
# balanced accuracy both survived an earlier run of this script.
manuscript_tex = (BASE / "manuscript.tex").read_text()
supplementary_tex = (BASE / "supplementary.tex").read_text()
prose = manuscript_tex + "\n" + supplementary_tex
tex = "\n".join([manuscript_tex, supplementary_tex]
                + [f.read_text() for f in sorted((BASE / "tables").glob("*.tex"))])
d = pd.read_csv(RES / "pathway_auc.csv")
c = pd.read_csv(RES / "comparisons_c.csv")
uni = pd.read_csv(RES / "univariate_proxy.csv")
mask = pd.read_csv(RES / "masking_check.csv")
vlm = pd.read_csv(RES / "vlm_summary.csv")
ext = pd.read_csv(RES / "external_validation.csv")
dev = pd.read_csv(RES / "device_stratified.csv")
rec = pd.read_csv(RES / "recoverability_summary.csv")

ENC = ["BiomedCLIP", "TorchXRayVision", "RAD-DINO", "ImageNet-DenseNet121"]
fails = []


def img(m, ns="none", col="auc_oof_mean"):
    return d[(d.pathway == "image") & (d.model_display == m) & (d.nuisance_set == ns)][col].iloc[0]


def check(label, v, nd=3, where=None):
    """`where=prose` for a number the running text quotes, so that a correct
    table cannot cover a wrong sentence."""
    hay = tex if where is None else where
    if f"{v:.{nd}f}" not in hay:
        scope = "the manuscript" if where is None else "the running text"
        fails.append(f"{label}: {v:.{nd}f} not in {scope}")


# primary discrimination, all encoders, with CI bounds; the Results quote these
for m in ENC:
    for col in ("auc_oof_mean", "auc_oof_ci_lo", "auc_oof_ci_hi"):
        check(f"primary {m} {col}", img(m, "none", col), where=prose)

# external validation, also quoted in the Results
for _, r in ext.iterrows():
    for v, lab in ((r.auc, "auc"), (r.ci_lo, "lo"), (r.ci_hi, "hi")):
        check(f"external {r.model_display} {lab}", v, where=prose)

# sensitivity layers
for m in ENC:
    for ns in ("S", "A", "A+G+E+S"):
        check(f"{m} adjusted {ns}", img(m, ns))
# the prose gives these as ranges, so the ends must appear
for kind in ("acquisition-adjusted", "demographic-adjusted"):
    v = c[c.comparison.str.contains(f"raw - {kind}")].delta_auc
    check(f"delta raw - {kind} min", v.min())
    check(f"delta raw - {kind} max", v.max())

# acquisition-only proxy and its comparison
px = d[(d.pathway == "proxy") & (d.model_display == "acquisition")].iloc[0]
for v, lab in ((px.auc_oof_mean, "mean"), (px.auc_oof_ci_lo, "lo"), (px.auc_oof_ci_hi, "hi")):
    check(f"acquisition-only proxy {lab}", v)
px29 = d[(d.pathway == "proxy") & (d.model_display == "all_proxy_factors")].iloc[0]
for v, lab in ((px29.auc_oof_mean, "mean"), (px29.auc_oof_ci_lo, "lo"), (px29.auc_oof_ci_hi, "hi")):
    check(f"29-proxy {lab}", v)

# device stratification: the single-unit numbers must all appear
single = dev[dev.stratum.str.startswith("3001")]
for _, r in single.iterrows():
    for v, lab in ((r.auc, "auc"), (r.ci_lo, "lo"), (r.ci_hi, "hi")):
        check(f"device single {r.model} {lab}", v)

# univariate group differences
for feat in ["image_width_px", "thorax_p10_intensity"]:
    check(f"univariate {feat}", uni[uni.feature == feat].auc.iloc[0])

# masking control
for m in ENC:
    for cond in ["unmasked", "annotation-masked"]:
        check(f"mask {m} {cond}",
              mask[(mask.condition == cond) & (mask.model_display == m)].auc_oof.iloc[0])

# recoverability claims quoted in the text
best = rec.groupby("task").primary_value.max()
for t in ["body_bbox_area_frac", "modality_cr", "image_height_px",
          "thorax_mean_intensity", "exam_days_since_min", "central_lucency_frac"]:
    check(f"recoverability {t}", best[t])

# VLM ranges
ftr = vlm[(vlm.condition == "ft") & (vlm.test_set == "foye_hernia")]
zs = vlm[(vlm.condition == "zero_shot") & (vlm.test_set == "foye_hernia")]
e2 = vlm[(vlm.condition == "ft") & (vlm.test_set == "foye_center2")]
for lab, v in [("zs min", zs.balanced_accuracy.min()), ("zs max", zs.balanced_accuracy.max()),
               ("ft bacc min", ftr.balanced_accuracy.min()),
               ("ft bacc max", ftr.balanced_accuracy.max()),
               ("ft auc min", ftr.auc.min()), ("ft auc max", ftr.auc.max())]:
    check(f"vlm {lab}", v)
# the external claim: the numbers quoted must be the fine-tuned ones, not the
# zero-shot ones, and the single-answer behaviour must still hold
if not (e2.single_class_output.all() and (e2.balanced_accuracy == 0.5).all()):
    fails.append("VLM external single-class claim no longer true")
check("vlm external ft auc min", e2.auc.min())
check("vlm external ft auc max", e2.auc.max())

# re-fitting only the decision threshold on external images
thr = json.loads((RES / "vlm_threshold.json").read_text())["models"]
best = max(thr, key=lambda r: r["balanced_accuracy_threshold_fitted_on_half"])
check("threshold refit, best model",
      best["balanced_accuracy_threshold_fitted_on_half"], where=prose)
for r in thr:
    if r["auc"] < 0.55:
        check(f"threshold refit, {r['model']} (near chance)",
              r["balanced_accuracy_threshold_fitted_on_half"])
if not all(abs(r["balanced_accuracy_at_model_threshold"] - 0.5) < 1e-9 for r in thr):
    fails.append("a fine-tuned model no longer sits at 0.500 on the external cohort")

# the vision-language split predates the unit audit: report what that cost
sp = json.loads((RES / "vlm_split_audit.json").read_text())
for lab, v in [("training images later excluded", sp["splits"]["train"]["excluded_by_the_audit"]),
               ("calibration images later excluded", sp["splits"]["calib"]["excluded_by_the_audit"]),
               ("test images later excluded", sp["splits"]["test"]["excluded_by_the_audit"]),
               ("children in the reported test set", sp["reported_internal_test"]["children"])]:
    if str(v) not in tex:
        fails.append(f"VLM split audit: {lab} ({v}) not in manuscript.tex")
if sp["reported_internal_test"]["children_also_in_training"] or \
        sp["reported_internal_test"]["children_also_in_calibration"]:
    fails.append("a child of the reported VLM test set also appears in training")
# the guard looks at the prose only: the generated tables legitimately hold
# these values in their zero-shot columns
prose = (BASE / "manuscript.tex").read_text() + (BASE / "supplementary.tex").read_text()
zs2 = vlm[(vlm.condition == "zero_shot") & (vlm.test_set == "foye_center2")]
for v, lab in ((zs2.auc.min(), "zero-shot external auc min"),
               (zs2.balanced_accuracy.min(), "zero-shot external bacc min")):
    if f"{v:.3f}" in prose:
        fails.append(f"{lab} ({v:.3f}) appears in the text; check it is not "
                     "being quoted as a fine-tuned result")

# structural facts asserted in Methods
for n in ["2089", "69", "2020", "2009", "49", "1960", "981", "979", "1800",
          "157", "115", "42", "590", "1370", "1607", "609"]:
    if n not in tex:
        fails.append(f"count {n} missing from manuscript.tex")


# ---- de-identification audit and the age recovered from the overlay ----

rep = json.loads((RES / "remask_report.json").read_text())["primary"]
tl, tr = rep["rect"]["top_left"], rep["rect"]["top_right"]
for label, v in [("rebuilt mask top-left width", 100 * tl["w"]),
                 ("rebuilt mask top-left height", 100 * tl["h"]),
                 ("rebuilt mask top-right width", 100 * tr["w"]),
                 ("rebuilt mask top-right height", 100 * tr["h"])]:
    check(label, v, nd=1)
if str(rep["audit"]["extra_boxes_images"]) not in tex:
    fails.append("count of images with text outside the fixed rectangles "
                 f"({rep['audit']['extra_boxes_images']}) not in manuscript.tex")

flag = pd.read_csv(RES / "mask_failure_flag.csv")
for label, v in [("residual-text rate, hernia",
                  100 * flag[flag.label == 1].residual_text.mean()),
                 ("residual-text rate, control",
                  100 * flag[flag.label == 0].residual_text.mean())]:
    check(label, v, nd=1)

ag = json.loads((RES / "age_analysis_remasked.json").read_text())
check("age alone AUC", ag["age_alone"]["auc"], where=prose)
check("age alone CI lo", ag["age_alone"]["ci"][0], where=prose)
check("age alone CI hi", ag["age_alone"]["ci"][1], where=prose)
for label, v in [("median age hernia", ag["age_alone"]["median_months_hernia"]),
                 ("median age control", ag["age_alone"]["median_months_control"])]:
    if str(int(v)) not in tex:
        fails.append(f"{label}: {int(v)} not in manuscript.tex")
if str(ag["n_age"]) not in tex:
    fails.append(f"n with readable age ({ag['n_age']}) not in manuscript.tex")
for r in ag["encoders_age_subcohort"]:
    check(f"{r['model']} age-subcohort raw", r["auc_raw"])
    check(f"{r['model']} age-adjusted", r["auc_age_adjusted"])
    check(f"{r['model']} age-adjusted CI lo", r["ci_age_adjusted"][0])
    check(f"{r['model']} age-adjusted CI hi", r["ci_age_adjusted"][1])
    check(f"{r['model']} age cost", r["delta"])

# ---- the age-matched confirmatory cohort the Discussion specifies ----
plan = json.loads((RES / "matched_cohort_plan.json").read_text())
for b in plan["bands"]:
    if b["shortfall"] and str(b["shortfall"]) not in tex:
        fails.append(f"shortfall for band {b['band']} ({b['shortfall']}) not in manuscript.tex")
for key in ("total_shortfall_readable", "total_shortfall_scaled"):
    if str(plan[key]) not in tex:
        fails.append(f"{key} ({plan[key]}) not in manuscript.tex")
for m in plan["matching"]:
    if m["caliper_months"] in (3, 12):
        if str(m["pairs"]) not in tex:
            fails.append(f"matched pairs at +/-{m['caliper_months']} mo "
                         f"({m['pairs']}) not in manuscript.tex")
        check(f"|SMD| at +/-{m['caliper_months']} mo", m["abs_smd"])
        if m["caliper_months"] == 3 and str(m["pairs_under_6y"]) not in tex:
            fails.append(f"matched pairs with the case under 6 y "
                         f"({m['pairs_under_6y']}) not in manuscript.tex")
for p in plan["sample_size"]:
    if p["auc"] in (0.55, 0.58, 0.60):
        for k in ("n80", "n90"):
            if str(p[k]) not in tex:
                fails.append(f"sample size {k} at AUC {p['auc']} ({p[k]}) "
                             "not in manuscript.tex")

# the primary build is the rebuilt cohort; the prose states how far the
# originally masked build differs from it
pub = pd.read_csv(BASE / "results_published" / "pathway_auc.csv")
diff = (d.set_index(["model_display", "nuisance_set"]).auc_oof_mean
        - pub.set_index(["model_display", "nuisance_set"]).auc_oof_mean).abs().dropna()
per_model = diff.groupby(level=0).max()
check("largest rebuilt-vs-original difference", per_model.max())
check("largest difference among the other three",
      per_model.drop(per_model.idxmax()).max())

# ---- analyses added for the revision ----
rev = json.loads((RES / "revision_analyses.json").read_text())
a1 = rev["A1_age_missingness"]
for label, v in [("readable rate hernia", 100 * a1["rate_hernia"]),
                 ("readable rate control", 100 * a1["rate_control"]),
                 ("readable rate CR", 100 * a1["readable_rate_CR"]),
                 ("readable rate DX", 100 * a1["readable_rate_DX"])]:
    check(label, v, nd=1)
for label, v in [("predict readable from variables",
                  a1["predict_readable_auc"]["proxies_29"]["auc"]),
                 ("predict readable from group",
                  a1["predict_readable_auc"]["label_only"]["auc"])]:
    check(label, v)
for n in (a1["unreadable_no_text_found"], a1["unreadable_text_not_read"]):
    if str(n) not in tex:
        fails.append(f"count of unreadable ages ({n}) missing from manuscript.tex")

for key in ("A2_age_matched", "A4_age_acquisition_matched",
            "A4b_age_acquisition_sex_matched"):
    blk = rev[key]
    if str(blk["pairs"]) not in tex:
        fails.append(f"{key}: pair count {blk['pairs']} missing from manuscript.tex")
    for r in blk["encoders"]:
        check(f"{key} {r['model']}", r["full_cohort_score_on_matched"])
        check(f"{key} {r['model']} CI lo", r["ci"][0], nd=2)
        check(f"{key} {r['model']} CI hi", r["ci"][1], nd=2)
        # the probe retrained inside the matched set answers the other question.
        # Table 5 prints its intervals to two decimals so that the table fits
        # inside the text block, so they are checked at that precision.
        check(f"{key} {r['model']} refit", r["probe_within_matched"])
        check(f"{key} {r['model']} refit CI lo", r["ci_within"][0], nd=2)
        check(f"{key} {r['model']} refit CI hi", r["ci_within"][1], nd=2)

a5 = rev["A5_nonlinear"]
for r in a5["age_subset"]:
    check(f"age spline {r['model']}", r["age_spline"])
for r in a5["full_cohort"]:
    if r["layer"] == "A":
        check(f"acquisition spline {r['model']}", r["spline"])

a6 = rev["A6_single_detector"]
check("smallest detectable AUC in the single unit",
      a6["min_detectable_auc_80pct_power"])
for label, v in [("rho age vs image height, DX",
                  rev["image_size_vs_age"]["DX"]["rho_age_image_height"]),
                 ("rho age vs image width, DX",
                  rev["image_size_vs_age"]["DX"]["rho_age_image_width"])]:
    check(label, v, nd=2)

# the vision-language models ran on the earlier build; the manuscript must not
# claim otherwise. `foye_vqa_items.csv` is the record of what they were given.
vlm_items = pd.read_csv("/scratch/hl106/80_workspace/foye/vlm_qlora/data/foye_vqa_items.csv")
vlm_roots = set(vlm_items[vlm_items.dataset == "foye_hernia"].image_path
                .str.rsplit("/", n=2).str[0])
if vlm_roots == {"/scratch/hl106/foye/ChestCR_prepared"}:
    flat = " ".join(manuscript_tex.split())      # the source is hard-wrapped
    if "fine-tuned and evaluated before the rebuild" not in flat:
        fails.append("the vision-language models ran on the originally masked images, "
                     "and the manuscript no longer says so")
elif vlm_roots != {"/scratch/hl106/foye/ChestCR_remasked"}:
    fails.append(f"unexpected image root for the vision-language models: {vlm_roots}")

# placeholders must not reach the submission
for marker in ("XXGITHUBURLXX", "reviewnote", "CONFIRM", "TODO"):
    if marker in tex:
        fails.append(f"placeholder {marker} still in the manuscript")

print(f"checked manuscript + supplementary + {len(list((BASE / 'tables').glob('*.tex')))} tables against results")
if fails:
    print(f"\n{len(fails)} MISMATCHES:")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("all checked numbers match the result files")
