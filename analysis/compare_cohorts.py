#!/usr/bin/env python3
"""Headline numbers on the 2009-image cohort (as submitted) next to the
1960-image cohort (revision). Writes results/cohort_comparison.md.

Old numbers are read from the archive made before the rerun,
../trash/before_clean_cohort_20260921/, new ones from results/.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

BASE = Path("/scratch/hl106/80_workspace/foye/paper")
OLD = BASE.parent / "trash" / "before_clean_cohort_20260921"
NEW = BASE
ENC = ["BiomedCLIP", "TorchXRayVision", "RAD-DINO", "ImageNet-DenseNet121"]
LAYERS = [("none", "Unadjusted"), ("S", "Sex"), ("A", "Acquisition (5)"),
          ("A+G", "+ body geometry (15)"), ("A+G+E", "+ thoracic exposure (8)"),
          ("A+G+E+S", "All 29 proxies")]


def ci(a, lo, hi):
    return f"{a:.3f} ({lo:.3f}–{hi:.3f})"


def pathway(root, build):
    d = pd.read_csv(root / build / "pathway_auc.csv")
    return d


def main():
    L = ["# 2009 张（投稿版） vs 1960 张（修订版）\n",
         "括号内为 95% CI。原遮挡版 = results/，重遮挡版 = results_remasked/。\n"]

    for build, title in [("results", "原遮挡版"), ("results_remasked", "重遮挡版")]:
        o, n = pathway(OLD, build), pathway(NEW, build)
        L.append(f"\n## 编码器判别力，逐层校正（{title}）\n")
        L.append("| 编码器 | 校正 | 2009 | 1960 | 差 |\n|---|---|---|---|---|")
        for e in ENC:
            for key, lab in LAYERS:
                ro = o[(o.pathway == "image") & (o.model_display == e) & (o.nuisance_set == key)]
                rn = n[(n.pathway == "image") & (n.model_display == e) & (n.nuisance_set == key)]
                if ro.empty or rn.empty:
                    continue
                ro, rn = ro.iloc[0], rn.iloc[0]
                L.append(f"| {e} | {lab} | {ci(ro.auc_oof_mean, ro.auc_oof_ci_lo, ro.auc_oof_ci_hi)} | "
                         f"{ci(rn.auc_oof_mean, rn.auc_oof_ci_lo, rn.auc_oof_ci_hi)} | "
                         f"{rn.auc_oof_mean - ro.auc_oof_mean:+.3f} |")

    o, n = pathway(OLD, "results"), pathway(NEW, "results")
    L.append("\n## 不看图的代理模型\n\n| 变量组 | 2009 | 1960 | 差 |\n|---|---|---|---|")
    for key in sorted(set(n[n.pathway != "image"].model)):
        ro, rn = o[o.model == key], n[n.model == key]
        if ro.empty or rn.empty:
            continue
        ro, rn = ro.iloc[0], rn.iloc[0]
        L.append(f"| {key} | {ci(ro.auc_oof_mean, ro.auc_oof_ci_lo, ro.auc_oof_ci_hi)} | "
                 f"{ci(rn.auc_oof_mean, rn.auc_oof_ci_lo, rn.auc_oof_ci_hi)} | "
                 f"{rn.auc_oof_mean - ro.auc_oof_mean:+.3f} |")

    co = pd.read_csv(OLD / "results" / "comparisons_c.csv")
    cn = pd.read_csv(NEW / "results" / "comparisons_c.csv")
    m = co.merge(cn, on="comparison", suffixes=("_o", "_n"))
    L.append("\n## 配对比较（ΔAUC，p）\n\n| 比较 | 2009 | 1960 |\n|---|---|---|")
    for r in m.itertuples():
        L.append(f"| {r.comparison} | {r.delta_auc_o:+.3f} ({r.ci_lo_o:+.3f}, {r.ci_hi_o:+.3f}), p={r.p_bootstrap_o:.3f} | "
                 f"{r.delta_auc_n:+.3f} ({r.ci_lo_n:+.3f}, {r.ci_hi_n:+.3f}), p={r.p_bootstrap_n:.3f} |")

    L.append("\n## 年龄（能读出年龄的子集）\n")
    for tag in ["published", "remasked"]:
        a = json.loads((OLD / "results" / f"age_analysis_{tag}.json").read_text())
        b = json.loads((NEW / "results" / f"age_analysis_{tag}.json").read_text())
        L.append(f"\n**{tag}**：可读 {a['n_age']} → {b['n_age']}；年龄单独 "
                 f"{ci(a['age_alone']['auc'], *a['age_alone']['ci'])} → {ci(b['age_alone']['auc'], *b['age_alone']['ci'])}；"
                 f"中位月龄 疝 {a['age_alone']['median_months_hernia']:.0f} / 对照 {a['age_alone']['median_months_control']:.0f} → "
                 f"{b['age_alone']['median_months_hernia']:.0f} / {b['age_alone']['median_months_control']:.0f}\n")
        L.append("| 编码器 | 未校正 2009 → 1960 | 扣年龄后 2009 → 1960 | 代价 2009 → 1960 |\n|---|---|---|---|")
        for ra, rb in zip(a["encoders_age_subcohort"], b["encoders_age_subcohort"]):
            L.append(f"| {ra['model']} | {ra['auc_raw']:.3f} → {rb['auc_raw']:.3f} | "
                     f"{ci(ra['auc_age_adjusted'], *ra['ci_age_adjusted'])} → {ci(rb['auc_age_adjusted'], *rb['ci_age_adjusted'])} | "
                     f"{ra['delta']:.3f} → {rb['delta']:.3f} |")

    do = pd.read_csv(OLD / "results" / "device_stratified.csv")
    dn = pd.read_csv(NEW / "results" / "device_stratified.csv")
    m = do.merge(dn, on=["stratum", "model"], suffixes=("_o", "_n"))
    L.append("\n## 设备分层\n\n| 层 | 编码器 | 2009 (n) | 1960 (n, 疝) |\n|---|---|---|---|")
    for r in m.itertuples():
        L.append(f"| {r.stratum} | {r.model} | {ci(r.auc_o, r.ci_lo_o, r.ci_hi_o)} ({r.n_o}) | "
                 f"{ci(r.auc_n, r.ci_lo_n, r.ci_hi_n)} ({r.n_n}, {r.n_hernia_n}) |")

    eo = pd.read_csv(OLD / "results" / "external_validation.csv")
    en = pd.read_csv(NEW / "results" / "external_validation.csv")
    m = eo.merge(en, on="model_display", suffixes=("_o", "_n"))
    L.append("\n## 外部队列（100 张；训练集 2009 → 1960）\n\n| 编码器 | 2009 | 1960 |\n|---|---|---|")
    for r in m.itertuples():
        L.append(f"| {r.model_display} | {ci(r.auc_o, r.ci_lo_o, r.ci_hi_o)} | {ci(r.auc_n, r.ci_lo_n, r.ci_hi_n)} |")

    out = NEW / "results" / "cohort_comparison.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
