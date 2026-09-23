#!/usr/bin/env python3
"""
Manuscript-grade re-analysis for the pediatric-chest-radiograph hernia
shortcut audit.

Everything the paper reports about the image pathway, the proxy pathway and
the residualisation cascade is produced here, with repeated stratified CV
(5 repeats x 5 folds) and bootstrap confidence intervals on pooled
out-of-fold predictions.

Inputs are read-only:
  /scratch/hl106/foye/cohort_1960/proxy_metadata.csv
  /scratch/hl106/foye/cohort_1960/emb_remasked/embeddings_*.npz

Outputs: paper/results/*.json, *.csv
"""

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# The 1960-image cohort written by build_clean_cohort.py (revision, 2026-09).
# The 2009-image originals are primary_2k_radiographic_proxy_metadata.csv and
# /scratch/hl106/foye/results_2k; see audit_units.py for what was dropped and why.
PROXY_CSV = "/scratch/hl106/foye/cohort_1960/proxy_metadata.csv"
# The rebuilt (fully masked) images are the primary build; the originally
# masked build is kept as a check, in results_published/.
EMB_DIR = "/scratch/hl106/foye/cohort_1960/emb_remasked"
OUT_DIR = "/scratch/hl106/80_workspace/foye/paper/results"

N_SPLITS = 5
N_REPEATS = 5
SEED = 42
N_BOOT = 2000

MODEL_DISPLAY = {
    "biomedclip": "BiomedCLIP",
    "torchxrayvision": "TorchXRayVision",
    "rad-dino": "RAD-DINO",
    "imagenet": "ImageNet-DenseNet121",
}
MODEL_ORDER = ["biomedclip", "torchxrayvision", "rad-dino", "imagenet"]

ACQUISITION = [
    "modality_cr",
    "exam_days_since_min",
    "image_width_px",
    "image_height_px",
    "image_aspect",
]
GEOMETRY = [
    "body_bbox_width_frac",
    "body_bbox_height_frac",
    "body_bbox_area_frac",
    "body_bbox_aspect",
    "body_mask_area_frac",
    "upper_body_area_share",
    "mid_body_area_share",
    "lower_body_area_share",
    "upper_body_width_frac",
    "mid_body_width_frac",
    "lower_body_width_frac",
    "shoulder_width_frac",
    "thorax_width_frac",
    "shoulder_to_lower_width_ratio",
    "thorax_to_lower_width_ratio",
]
EXPOSURE = [
    "thorax_mean_intensity",
    "thorax_p10_intensity",
    "thorax_p90_intensity",
    "central_lucency_frac",
    "left_lucency_frac",
    "right_lucency_frac",
    "lucency_lr_asymmetry",
    "thorax_body_mask_frac",
]
SEX = ["sex_male"]

NUISANCE_SETS = {
    "none": [],
    "S": SEX,
    "A": ACQUISITION,
    "A+G": ACQUISITION + GEOMETRY,
    "A+G+E": ACQUISITION + GEOMETRY + EXPOSURE,
    "A+G+E+S": ACQUISITION + GEOMETRY + EXPOSURE + SEX,
}
NUISANCE_LABEL = {
    "none": "Raw embedding (no adjustment)",
    "S": "Sex only (1) - the conventional demographic check",
    "A": "Acquisition (5)",
    "A+G": "Acquisition + body geometry (20)",
    "A+G+E": "Acquisition + geometry + exposure (28)",
    "A+G+E+S": "All proxies incl. sex (29)",
}

FACTOR_SETS = {
    "sex": SEX,
    "acquisition": ACQUISITION,
    "body_geometry": GEOMETRY,
    "thorax_intensity_lucency": EXPOSURE,
    "all_proxy_factors": ACQUISITION + GEOMETRY + EXPOSURE + SEX,
}


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def load_embeddings(d):
    out = {}
    for path in sorted(Path(d).glob("embeddings_*.npz")):
        key = path.stem.replace("embeddings_", "")
        out[key] = np.load(path)["features"].astype(np.float32)
    return out


def boot_auc_ci(y, s, n_boot=N_BOOT, seed=SEED):
    """Stratified bootstrap percentile CI for a single AUC."""
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate(
            [rng.choice(pos, pos.size, replace=True), rng.choice(neg, neg.size, replace=True)]
        )
        vals[b] = roc_auc_score(y[idx], s[idx])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def boot_auc_diff_ci(y, s_a, s_b, n_boot=N_BOOT, seed=SEED):
    """Paired bootstrap for AUC(a) - AUC(b) on the same subjects."""
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate(
            [rng.choice(pos, pos.size, replace=True), rng.choice(neg, neg.size, replace=True)]
        )
        vals[b] = roc_auc_score(y[idx], s_a[idx]) - roc_auc_score(y[idx], s_b[idx])
    lo, hi = float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
    p_two = 2.0 * min((vals <= 0).mean(), (vals >= 0).mean())
    return lo, hi, float(min(p_two, 1.0))


def fold_residualize(X_tr, X_te, Z_tr, Z_te, alpha=10.0):
    imp = SimpleImputer(strategy="median")
    zs, xs = StandardScaler(), StandardScaler()
    Ztr = zs.fit_transform(imp.fit_transform(Z_tr))
    Zte = zs.transform(imp.transform(Z_te))
    Xtr = xs.fit_transform(X_tr)
    Xte = xs.transform(X_te)
    ridge = Ridge(alpha=alpha).fit(Ztr, Xtr)
    Rtr = Xtr - ridge.predict(Ztr)
    Rte = Xte - ridge.predict(Zte)
    ss_res = float(np.sum(Rte ** 2))
    ss_tot = float(np.sum((Xte - Xtr.mean(axis=0, keepdims=True)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return Rtr, Rte, r2


def cv_scores(X, y, Z=None, tabular=False):
    """Repeated stratified CV. Returns per-fold AUCs and per-repeat pooled OOF scores."""
    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    aucs, baccs, r2s = [], [], []
    oof = np.full((N_REPEATS, len(y)), np.nan)
    for i, (tr, te) in enumerate(rskf.split(X, y)):
        rep = i // N_SPLITS
        if tabular:
            clf = make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(),
                LogisticRegression(max_iter=5000, C=1.0, random_state=SEED),
            )
            clf.fit(X[tr], y[tr])
            s = clf.predict_proba(X[te])[:, 1]
        else:
            if Z is not None and Z.shape[1] > 0:
                Xtr, Xte, r2 = fold_residualize(X[tr], X[te], Z[tr], Z[te])
                r2s.append(r2)
            else:
                sc = StandardScaler()
                Xtr = sc.fit_transform(X[tr])
                Xte = sc.transform(X[te])
            clf = LogisticRegression(max_iter=5000, C=1.0, random_state=SEED)
            clf.fit(Xtr, y[tr])
            s = clf.predict_proba(Xte)[:, 1]
        oof[rep, te] = s
        aucs.append(roc_auc_score(y[te], s))
        baccs.append(balanced_accuracy_score(y[te], (s >= 0.5).astype(int)))
    return {
        "auc_folds": aucs,
        "auc_mean": float(np.mean(aucs)),
        "auc_sd": float(np.std(aucs, ddof=1)),
        "bacc_mean": float(np.mean(baccs)),
        "bacc_sd": float(np.std(baccs, ddof=1)),
        "nuisance_r2_mean": float(np.nanmean(r2s)) if r2s else None,
        "nuisance_r2_sd": float(np.nanstd(r2s, ddof=1)) if len(r2s) > 1 else None,
        "oof": oof,
    }


def bagged(oof):
    """One score per subject: the out-of-fold score averaged over repeats.

    Every downstream estimate (the primary AUC, its bootstrap CI, and the
    paired comparisons) is computed from this single vector, so the numbers
    in different tables refer to the same quantity.
    """
    return np.nanmean(oof, axis=0)


def summarise(res, y, seed_off=0):
    """AUC of the repeat-averaged out-of-fold score, with a bootstrap CI."""
    s = bagged(res["oof"])
    lo, hi = boot_auc_ci(y, s, seed=SEED + seed_off)
    return {
        "auc_mean": res["auc_mean"],
        "auc_sd": res["auc_sd"],
        "bacc_mean": res["bacc_mean"],
        "bacc_sd": res["bacc_sd"],
        "auc_oof_mean": float(roc_auc_score(y, s)),
        "auc_oof_ci_lo": lo,
        "auc_oof_ci_hi": hi,
        "nuisance_r2_mean": res["nuisance_r2_mean"],
        "nuisance_r2_sd": res["nuisance_r2_sd"],
    }


def main():
    global EMB_DIR, OUT_DIR
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--emb-dir", default=EMB_DIR,
                    help="directory of embeddings_<model>.npz")
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()
    EMB_DIR, OUT_DIR = args.emb_dir, args.out_dir

    ensure_dir(OUT_DIR)
    meta = pd.read_csv(PROXY_CSV)
    y = meta["label"].astype(int).to_numpy()
    emb = load_embeddings(EMB_DIR)
    n = len(y)
    print(f"[data] n={n} labels={dict(Counter(y.tolist()))}", flush=True)

    store_oof = {}
    rows = []

    # ---------- 1. image pathway x residualisation cascade ----------
    for ns_key, cols in NUISANCE_SETS.items():
        Z = (
            meta[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
            if cols
            else None
        )
        for k, mk in enumerate(MODEL_ORDER):
            X = emb[mk]
            assert len(X) == n, f"{mk}: {len(X)} != {n}"
            print(f"[image] {mk} | nuisance={ns_key}", flush=True)
            res = cv_scores(X, y, Z)
            s = summarise(res, y, seed_off=k)
            store_oof[(mk, ns_key)] = res["oof"]
            rows.append(
                dict(
                    pathway="image",
                    model=mk,
                    model_display=MODEL_DISPLAY[mk],
                    nuisance_set=ns_key,
                    nuisance_label=NUISANCE_LABEL[ns_key],
                    n_nuisance=len(cols),
                    **s,
                )
            )

    # ---------- 2. proxy-only pathway (image-free) ----------
    for fs_key, cols in FACTOR_SETS.items():
        Z = meta[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
        print(f"[proxy] {fs_key} ({len(cols)} features)", flush=True)
        res = cv_scores(Z, y, tabular=True)
        s = summarise(res, y, seed_off=7)
        store_oof[("proxy_" + fs_key, "none")] = res["oof"]
        rows.append(
            dict(
                pathway="proxy",
                model="proxy_" + fs_key,
                model_display=fs_key,
                nuisance_set="none",
                nuisance_label="Image-free proxy model",
                n_nuisance=len(cols),
                **s,
            )
        )

    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "pathway_auc.csv"), index=False)
    with open(os.path.join(OUT_DIR, "pathway_auc.json"), "w") as f:
        json.dump(rows, f, indent=2)

    # ---------- 3. paired comparison: full proxy model vs each raw image model ----------
    comp = []
    proxy_oof = bagged(store_oof[("proxy_all_proxy_factors", "none")])
    for mk in MODEL_ORDER:
        img_oof = bagged(store_oof[(mk, "none")])
        d = roc_auc_score(y, proxy_oof) - roc_auc_score(y, img_oof)
        lo, hi, p = boot_auc_diff_ci(y, proxy_oof, img_oof)
        comp.append(
            dict(
                comparison=f"proxy_all(29) - {MODEL_DISPLAY[mk]}",
                delta_auc=float(d),
                ci_lo=lo,
                ci_hi=hi,
                p_bootstrap=p,
            )
        )
    # raw vs fully residualised, per model
    for mk in MODEL_ORDER:
        raw = bagged(store_oof[(mk, "none")])
        adj = bagged(store_oof[(mk, "A+G+E+S")])
        d = roc_auc_score(y, raw) - roc_auc_score(y, adj)
        lo, hi, p = boot_auc_diff_ci(y, raw, adj)
        comp.append(
            dict(
                comparison=f"{MODEL_DISPLAY[mk]} raw - residualised(29)",
                delta_auc=float(d),
                ci_lo=lo,
                ci_hi=hi,
                p_bootstrap=p,
            )
        )
    pd.DataFrame(comp).to_csv(os.path.join(OUT_DIR, "paired_comparisons.csv"), index=False)

    # ---------- 4. univariate proxy associations with bootstrap CI ----------
    uni = []
    all_feats = ACQUISITION + GEOMETRY + EXPOSURE + SEX
    for j, feat in enumerate(all_feats):
        v = pd.to_numeric(meta[feat], errors="coerce").to_numpy(dtype=float)
        ok = ~np.isnan(v)
        yy, vv = y[ok], v[ok]
        a = roc_auc_score(yy, vv)
        direction = "higher in hernia" if a >= 0.5 else "higher in non-hernia"
        # direction-free AUC so every row reads as discrimination strength
        s = vv if a >= 0.5 else -vv
        a_df = roc_auc_score(yy, s)
        lo, hi = boot_auc_ci(yy, s, n_boot=1000, seed=SEED + j)
        h_mean, h_sd = float(np.mean(vv[yy == 1])), float(np.std(vv[yy == 1], ddof=1))
        n_mean, n_sd = float(np.mean(vv[yy == 0])), float(np.std(vv[yy == 0], ddof=1))
        pooled_sd = np.sqrt((h_sd ** 2 + n_sd ** 2) / 2)
        uni.append(
            dict(
                feature=feat,
                n=int(ok.sum()),
                hernia_mean=h_mean,
                hernia_sd=h_sd,
                nonhernia_mean=n_mean,
                nonhernia_sd=n_sd,
                smd=float((h_mean - n_mean) / pooled_sd) if pooled_sd > 0 else np.nan,
                auc=float(a_df),
                auc_ci_lo=lo,
                auc_ci_hi=hi,
                direction=direction,
            )
        )
    pd.DataFrame(uni).sort_values("auc", ascending=False).to_csv(
        os.path.join(OUT_DIR, "univariate_proxy.csv"), index=False
    )

    np.savez_compressed(
        os.path.join(OUT_DIR, "oof_scores.npz"),
        y=y,
        **{f"{k[0]}|{k[1]}": v for k, v in store_oof.items()},
    )

    with open(os.path.join(OUT_DIR, "analysis_manifest.json"), "w") as f:
        json.dump(
            {
                "proxy_csv": PROXY_CSV,
                "embeddings_dir": EMB_DIR,
                "n": int(n),
                "label_counts": {str(k): int(v) for k, v in Counter(y.tolist()).items()},
                "cv": f"RepeatedStratifiedKFold({N_SPLITS} folds x {N_REPEATS} repeats), seed={SEED}",
                "bootstrap": f"{N_BOOT} stratified resamples, percentile 95% CI, on the repeat-averaged out-of-fold score",
                "probe": "LogisticRegression(C=1.0, max_iter=5000) on standardised features",
                "residualisation": "Per fold: Ridge(alpha=10) maps standardised nuisance -> standardised embedding, fit on training fold only; residual = embedding - prediction",
                "nuisance_sets": NUISANCE_SETS,
                "factor_sets": FACTOR_SETS,
                "compute_note": "CPU-only sklearn run.",
            },
            f,
            indent=2,
        )
    print("[done]", OUT_DIR, flush=True)


if __name__ == "__main__":
    main()
