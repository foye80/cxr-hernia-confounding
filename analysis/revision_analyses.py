#!/usr/bin/env python3
"""Analyses added for the revision (2026-09), all on the 1960-image cohort and
the rebuilt (fully masked) images, the build the manuscript reports.

  A1  age missingness: who has a readable age, and does it matter
      (editor 4, reviewer 1.3)
  A2  age-matched comparison inside the existing data
      (reviewer 2.3, editor's closing paragraph)
  A4  matched on age and acquisition at once (editor's closing paragraph)
  A5  non-linear adjustment for age and acquisition (reviewer 1.10)
  A6  precision of the single-detector analysis (editor 6, reviewer 1.4)

Output: results/revision_analyses.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, RepeatedStratifiedKFold
from sklearn.preprocessing import SplineTransformer, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plan_matched_controls import hanley_var  # noqa: E402
from run_paper_analysis import (ACQUISITION, EXPOSURE, GEOMETRY,  # noqa: E402
                                PROXY_CSV, SEX)

BASE = Path("/scratch/hl106/80_workspace/foye/paper")
RES = BASE / "results"
EMB = Path("/scratch/hl106/foye/cohort_1960/emb_remasked")
MODELS = ["biomedclip", "torchxrayvision", "rad-dino", "imagenet"]
DISPLAY = {"biomedclip": "BiomedCLIP", "torchxrayvision": "TorchXRayVision",
           "rad-dino": "RAD-DINO", "imagenet": "ImageNet-DenseNet121"}
SEED, N_BOOT = 42, 2000


# ---------------------------------------------------------------- helpers
def smd(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    s = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return float((a.mean() - b.mean()) / s) if s > 0 else 0.0


def boot_ci(y, s, seed=SEED):
    """Stratified bootstrap percentile CI of an AUC."""
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    v = np.empty(N_BOOT)
    for b in range(N_BOOT):
        i = np.concatenate([rng.choice(pos, pos.size), rng.choice(neg, neg.size)])
        v[b] = roc_auc_score(y[i], s[i])
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def pair_boot_ci(case_idx, ctrl_idx, s, seed=SEED):
    """Bootstrap over matched pairs, keeping each pair together."""
    rng = np.random.default_rng(seed)
    n = len(case_idx)
    v = np.empty(N_BOOT)
    for b in range(N_BOOT):
        k = rng.integers(0, n, n)
        i = np.concatenate([case_idx[k], ctrl_idx[k]])
        y = np.r_[np.ones(n), np.zeros(n)]
        v[b] = roc_auc_score(y, s[i])
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def probe_oof(X, y, splits, nuisance=None, expand=None):
    """Logistic probe over the given (train, test) splits, optionally after
    residualising X on a nuisance matrix inside each training fold. `expand`
    turns the raw nuisance columns into a basis (fitted on the training fold
    only). Returns the repeat-averaged out-of-fold score."""
    acc = np.full((len(splits), len(y)), np.nan)
    for r, (tr, te) in enumerate(splits):
        Xtr, Xte = X[tr], X[te]
        if nuisance is not None:
            imp = SimpleImputer(strategy="median").fit(nuisance[tr])
            Ztr, Zte = imp.transform(nuisance[tr]), imp.transform(nuisance[te])
            if expand is not None:
                ex = expand().fit(Ztr)
                Ztr, Zte = ex.transform(Ztr), ex.transform(Zte)
            zs = StandardScaler().fit(Ztr)
            Ztr, Zte = zs.transform(Ztr), zs.transform(Zte)
            xs = StandardScaler().fit(Xtr)
            Xtr, Xte = xs.transform(Xtr), xs.transform(Xte)
            ridge = Ridge(alpha=10.0).fit(Ztr, Xtr)
            Xtr, Xte = Xtr - ridge.predict(Ztr), Xte - ridge.predict(Zte)
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(C=1.0, max_iter=5000, random_state=SEED)
        clf.fit(sc.transform(Xtr), y[tr])
        acc[r, te] = clf.predict_proba(sc.transform(Xte))[:, 1]
    # splits come as n_repeats blocks of 5 folds; average over repeats
    n_rep = len(splits) // 5
    per_rep = np.array([np.nanmean(acc[k * 5:(k + 1) * 5], axis=0) for k in range(n_rep)])
    return np.nanmean(per_rep, axis=0)


def rskf_splits(y):
    return list(RepeatedStratifiedKFold(n_splits=5, n_repeats=5,
                                        random_state=SEED).split(np.zeros(len(y)), y))


def spline_basis():
    return SplineTransformer(n_knots=5, degree=3, include_bias=False)


def greedy_pairs(case_idx, ctrl_idx, dist, caliper):
    """1:1 nearest-neighbour matching without replacement. `dist(i, J)` gives
    the distance from case i to the candidate controls J. Cases are visited
    in the order given."""
    used = np.zeros(len(ctrl_idx), bool)
    pairs = []
    for i in case_idx:
        d = dist(i, ctrl_idx)
        d = np.where(used, np.inf, d)
        j = int(np.argmin(d))
        if d[j] <= caliper:
            used[j] = True
            pairs.append((i, ctrl_idx[j]))
    return np.array(pairs, dtype=int).reshape(-1, 2)


def refit_within_pairs(X, pairs, seed=SEED):
    """Train and test a probe inside a matched set, with folds formed by pair,
    and return one out-of-fold score per matched child. This answers a
    different question from evaluating the primary model on the same children:
    it asks whether any difference is learnable at equal age (and acquisition,
    and sex), not whether the model we already fitted still separates them."""
    idx = np.concatenate([pairs[:, 0], pairs[:, 1]])
    y = np.r_[np.ones(len(pairs)), np.zeros(len(pairs))].astype(int)
    groups = np.r_[np.arange(len(pairs)), np.arange(len(pairs))]
    splits = []
    for rep_i in range(5):
        perm = np.random.default_rng(seed + rep_i).permutation(len(pairs))
        splits += list(GroupKFold(n_splits=5).split(np.zeros(len(idx)), y, perm[groups]))
    return idx, y, probe_oof(X[idx], y, splits)


# ---------------------------------------------------------------- main
def main():
    meta = pd.read_csv(PROXY_CSV)
    y = meta.label.astype(int).to_numpy()
    n = len(y)
    age_t = pd.read_csv(RES / "burned_in_age.csv")
    assert len(age_t) == n and (age_t.label.to_numpy() == y).all()
    age = age_t.age_months.to_numpy(float)
    have = np.isfinite(age)
    oof = np.load(RES / "oof_scores.npz")
    score = {m: np.nanmean(oof[f"{m}|none"], axis=0) for m in MODELS}
    emb = {m: np.load(EMB / f"embeddings_{m}.npz")["features"].astype(np.float32) for m in MODELS}
    for m in MODELS:
        assert len(emb[m]) == n
    proxies = ACQUISITION + GEOMETRY + EXPOSURE + SEX
    P = meta[proxies].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    out = {"n": int(n), "build": "rebuilt (fully masked)"}

    # ============ A1 age missingness ============
    print("\n== A1 age missingness")
    no_text = age_t.age_text.isna().to_numpy()
    a1 = {
        "n_readable": int(have.sum()),
        "readable_hernia": [int(have[y == 1].sum()), int((y == 1).sum())],
        "readable_control": [int(have[y == 0].sum()), int((y == 0).sum())],
        "unreadable_no_text_found": int((~have & no_text).sum()),
        "unreadable_text_not_read": int((~have & ~no_text).sum()),
    }
    a1["rate_hernia"] = a1["readable_hernia"][0] / a1["readable_hernia"][1]
    a1["rate_control"] = a1["readable_control"][0] / a1["readable_control"][1]
    print(f"readable: hernia {a1['rate_hernia']:.3f}, control {a1['rate_control']:.3f}; "
          f"unreadable: no text {a1['unreadable_no_text_found']}, text not read {a1['unreadable_text_not_read']}")
    # readable vs unreadable, variable by variable
    rows = []
    for j, c in enumerate(proxies):
        rows.append({"variable": c, "smd_readable_minus_unreadable": smd(P[have, j], P[~have, j])})
    rows.append({"variable": "label_hernia", "smd_readable_minus_unreadable": smd(y[have], y[~have])})
    rows.sort(key=lambda r: -abs(r["smd_readable_minus_unreadable"]))
    a1["smd"] = rows
    a1["n_abs_smd_over_0.1"] = int(sum(abs(r["smd_readable_minus_unreadable"]) > 0.1 for r in rows))
    a1["max_abs_smd"] = rows[0]
    print("largest |SMD|:", [(r["variable"], round(r["smd_readable_minus_unreadable"], 3)) for r in rows[:6]])
    # by modality and year
    mod = meta.modality_cr.to_numpy()
    a1["readable_rate_CR"] = float(have[mod == 1].mean())
    a1["readable_rate_DX"] = float(have[mod == 0].mean())
    yr = meta.exam_year.to_numpy()
    a1["readable_rate_by_year"] = {str(int(k)): float(have[yr == k].mean()) for k in np.unique(yr)}
    print(f"readable rate CR {a1['readable_rate_CR']:.3f}, DX {a1['readable_rate_DX']:.3f}; by year {a1['readable_rate_by_year']}")
    # can readability be predicted?
    hv = have.astype(int)
    sp = rskf_splits(hv)
    feats = {"proxies_29": P, "label_only": y[:, None].astype(float),
             "proxies_29_plus_label": np.column_stack([P, y])}
    a1["predict_readable_auc"] = {}
    for k, Z in feats.items():
        acc = np.full((len(sp), n), np.nan)
        for r, (tr, te) in enumerate(sp):
            imp = SimpleImputer(strategy="median").fit(Z[tr])
            sc = StandardScaler().fit(imp.transform(Z[tr]))
            clf = LogisticRegression(max_iter=5000).fit(sc.transform(imp.transform(Z[tr])), hv[tr])
            acc[r, te] = clf.predict_proba(sc.transform(imp.transform(Z[te])))[:, 1]
        s = np.nanmean(np.array([np.nanmean(acc[i * 5:(i + 1) * 5], 0) for i in range(5)]), 0)
        a1["predict_readable_auc"][k] = {"auc": float(roc_auc_score(hv, s)), "ci": boot_ci(hv, s)}
    print("predict readable:", {k: round(v["auc"], 3) for k, v in a1["predict_readable_auc"].items()})
    # encoder discrimination in each subset
    a1["encoder_auc_by_subset"] = []
    for m in MODELS:
        s = score[m]
        r = {"model": DISPLAY[m],
             "readable": float(roc_auc_score(y[have], s[have])), "readable_ci": boot_ci(y[have], s[have]),
             "unreadable": float(roc_auc_score(y[~have], s[~have])), "unreadable_ci": boot_ci(y[~have], s[~have])}
        a1["encoder_auc_by_subset"].append(r)
        print(f"  {r['model']:22s} readable {r['readable']:.3f}  unreadable {r['unreadable']:.3f}")
    out["A1_age_missingness"] = a1

    # ============ A2 age-matched ============
    print("\n== A2 age-matched (+/-3 months)")
    cases = np.flatnonzero(have & (y == 1))
    ctrls = np.flatnonzero(have & (y == 0))
    cases = cases[np.argsort(age[cases], kind="stable")]
    ctrls = ctrls[np.argsort(age[ctrls], kind="stable")]
    pr = greedy_pairs(cases, ctrls, lambda i, J: np.abs(age[J] - age[i]), 3)
    ci_, co_ = pr[:, 0], pr[:, 1]
    a2 = {"pairs": int(len(pr)), "pairs_case_under_6y": int((age[ci_] < 72).sum()),
          "balance": {"age_smd": smd(age[ci_], age[co_]),
                      "sex_smd": smd(meta.sex_male.to_numpy()[ci_], meta.sex_male.to_numpy()[co_]),
                      "modality_cr_smd": smd(mod[ci_], mod[co_]),
                      "image_width_smd": smd(meta.image_width_px.to_numpy()[ci_], meta.image_width_px.to_numpy()[co_]),
                      "image_height_smd": smd(meta.image_height_px.to_numpy()[ci_], meta.image_height_px.to_numpy()[co_]),
                      "exam_days_smd": smd(meta.exam_days_since_min.to_numpy()[ci_], meta.exam_days_since_min.to_numpy()[co_])},
          "encoders": []}
    print(f"pairs {a2['pairs']} ({a2['pairs_case_under_6y']} with case <6 y); balance {({k: round(v, 3) for k, v in a2['balance'].items()})}")
    idx = np.concatenate([ci_, co_])
    ym = np.r_[np.ones(len(pr)), np.zeros(len(pr))].astype(int)
    groups = np.r_[np.arange(len(pr)), np.arange(len(pr))]
    u6 = age[ci_] < 72
    for m in MODELS:
        s = score[m]
        # (i) the full-cohort out-of-fold score, evaluated on the matched children
        auc_i = float(roc_auc_score(ym, s[idx]))
        ci_i = pair_boot_ci(ci_, co_, s)
        # the same, on pairs whose case is under 6 years
        auc_u6 = float(roc_auc_score(np.r_[np.ones(u6.sum()), np.zeros(u6.sum())],
                                     s[np.concatenate([ci_[u6], co_[u6]])]))
        ci_u6 = pair_boot_ci(ci_[u6], co_[u6], s)
        # (ii) a probe trained and tested inside the matched set, folds by pair
        idx2, ym2, s_in = refit_within_pairs(emb[m], pr)
        auc_ii = float(roc_auc_score(ym2, s_in))
        full = np.zeros(n); full[idx2] = s_in
        ci_ii = pair_boot_ci(ci_, co_, full)
        r = {"model": DISPLAY[m],
             "full_cohort_score_on_matched": auc_i, "ci": ci_i,
             "full_cohort_score_on_matched_under6": auc_u6, "ci_under6": ci_u6,
             "probe_within_matched": auc_ii, "ci_within": ci_ii}
        a2["encoders"].append(r)
        print(f"  {r['model']:22s} full-cohort score on pairs {auc_i:.3f} {np.round(ci_i,3)} | "
              f"<6y {auc_u6:.3f} {np.round(ci_u6,3)} | within-matched probe {auc_ii:.3f} {np.round(ci_ii,3)}")
    out["A2_age_matched"] = a2

    # ============ A4 age + acquisition ============
    print("\n== A4 age + acquisition matched")
    W = meta.image_width_px.to_numpy(float)
    H = meta.image_height_px.to_numpy(float)

    def dist4(i, J):
        same_mod = mod[J] == mod[i]
        da = np.abs(age[J] - age[i]) / 6.0
        dw = np.abs(np.log(W[J] / W[i])) / 0.10
        dh = np.abs(np.log(H[J] / H[i])) / 0.10
        d = np.maximum.reduce([da, dw, dh])     # every criterion within its caliper
        return np.where(same_mod, d, np.inf)

    pr4 = greedy_pairs(cases, ctrls, dist4, 1.0)
    a4 = {"criteria": "same CR/DX mode; age within 6 months; image width and height each within 10%",
          "pairs": int(len(pr4))}
    if len(pr4):
        c4, k4 = pr4[:, 0], pr4[:, 1]
        a4["pairs_case_under_6y"] = int((age[c4] < 72).sum())
        a4["balance"] = {v: smd(meta[v].to_numpy(float)[c4], meta[v].to_numpy(float)[k4])
                         for v in ["modality_cr", "image_width_px", "image_height_px", "image_aspect",
                                   "exam_days_since_min", "sex_male"] + GEOMETRY[:3]}
        a4["balance"]["age_months"] = smd(age[c4], age[k4])
        a4["encoders"] = []
        for m in MODELS:
            s = score[m]
            auc = float(roc_auc_score(np.r_[np.ones(len(pr4)), np.zeros(len(pr4))], s[np.r_[c4, k4]]))
            idx4, ym4, s_in4 = refit_within_pairs(emb[m], pr4)
            full4 = np.zeros(n); full4[idx4] = s_in4
            a4["encoders"].append({"model": DISPLAY[m], "full_cohort_score_on_matched": auc,
                                   "ci": pair_boot_ci(c4, k4, s),
                                   "probe_within_matched": float(roc_auc_score(ym4, s_in4)),
                                   "ci_within": pair_boot_ci(c4, k4, full4)})
        print(f"pairs {a4['pairs']} ({a4['pairs_case_under_6y']} case <6 y); "
              f"balance {({k: round(v, 3) for k, v in a4['balance'].items()})}")
        for r in a4["encoders"]:
            print(f"  {r['model']:22s} {r['full_cohort_score_on_matched']:.3f} {np.round(r['ci'],3)}")
    out["A4_age_acquisition_matched"] = a4

    # A4b: the same, with sex matched as well (the A2 and A4 pairs leave sex
    # unbalanced, SMD about -0.43, because the controls are almost all boys)
    sex = meta.sex_male.to_numpy()

    def dist4s(i, J):
        return np.where(sex[J] == sex[i], dist4(i, J), np.inf)

    pr4s = greedy_pairs(cases, ctrls, dist4s, 1.0)
    c4s, k4s = pr4s[:, 0], pr4s[:, 1]
    a4s = {"criteria": a4["criteria"] + "; same sex", "pairs": int(len(pr4s)),
           "pairs_case_under_6y": int((age[c4s] < 72).sum()),
           "balance": {"age_months": smd(age[c4s], age[k4s]), "sex_male": smd(sex[c4s], sex[k4s])},
           "encoders": []}
    for m in MODELS:
        s = score[m]
        auc = float(roc_auc_score(np.r_[np.ones(len(pr4s)), np.zeros(len(pr4s))], s[np.r_[c4s, k4s]]))
        idx4s, ym4s, s_in4s = refit_within_pairs(emb[m], pr4s)
        full4s = np.zeros(n); full4s[idx4s] = s_in4s
        a4s["encoders"].append({"model": DISPLAY[m], "full_cohort_score_on_matched": auc,
                                "ci": pair_boot_ci(c4s, k4s, s),
                                "probe_within_matched": float(roc_auc_score(ym4s, s_in4s)),
                                "ci_within": pair_boot_ci(c4s, k4s, full4s)})
    print(f"with sex: pairs {a4s['pairs']}")
    for r in a4s["encoders"]:
        print(f"  {r['model']:22s} {r['full_cohort_score_on_matched']:.3f} {np.round(r['ci'],3)}")
    out["A4b_age_acquisition_sex_matched"] = a4s

    # why acquisition variables stand in for the child: image size tracks age
    from scipy.stats import spearmanr
    rho = {}
    for label, sel in [("DX", mod == 0), ("CR", mod == 1)]:
        s_ = have & sel
        rho[label] = {"n": int(s_.sum()),
                      "rho_age_image_height": float(spearmanr(age[s_], H[s_]).correlation),
                      "rho_age_image_width": float(spearmanr(age[s_], W[s_]).correlation)}
    out["image_size_vs_age"] = rho
    print("image size vs age:", rho)

    # ============ A5 non-linear adjustment ============
    print("\n== A5 non-linear adjustment")
    a5 = {"age_subset": [], "full_cohort": []}
    # (a) age, on the readable subset: the linear basis age_analysis.py uses vs a cubic spline
    ya = y[have]
    sp_a = rskf_splits(ya)
    Zage = age[have][:, None]
    lin_age = lambda: _LinLog()  # noqa: E731
    for m in MODELS:
        X = emb[m][have]
        s_lin = probe_oof(X, ya, sp_a, Zage, expand=lin_age)
        s_spl = probe_oof(X, ya, sp_a, Zage, expand=spline_basis)
        r = {"model": DISPLAY[m],
             "age_linear_log": float(roc_auc_score(ya, s_lin)), "ci_linear": boot_ci(ya, s_lin),
             "age_spline": float(roc_auc_score(ya, s_spl)), "ci_spline": boot_ci(ya, s_spl)}
        a5["age_subset"].append(r)
        print(f"  age   {r['model']:22s} linear+log {r['age_linear_log']:.3f}  spline {r['age_spline']:.3f}")
    # (b) acquisition, and acquisition + body geometry, on the whole cohort
    sp_f = rskf_splits(y)
    acq_cont = [c for c in ACQUISITION if c != "modality_cr"]
    for layer, cols in [("A", ACQUISITION), ("A+G", ACQUISITION + GEOMETRY)]:
        Z = meta[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        cont = [i for i, c in enumerate(cols) if c != "modality_cr"]
        binc = [i for i, c in enumerate(cols) if c == "modality_cr"]
        expand = lambda cont=cont, binc=binc: _SplineSome(cont, binc)  # noqa: E731
        for m in MODELS:
            s_spl = probe_oof(emb[m], y, sp_f, Z, expand=expand)
            r = {"model": DISPLAY[m], "layer": layer,
                 "spline": float(roc_auc_score(y, s_spl)), "ci_spline": boot_ci(y, s_spl)}
            a5["full_cohort"].append(r)
            print(f"  {layer:4s}  {r['model']:22s} spline {r['spline']:.3f}")
    out["A5_nonlinear"] = a5

    # ============ A6 single detector precision ============
    print("\n== A6 single-detector precision")
    sq = (meta.image_width_px.astype(int).astype(str) + "x" +
          meta.image_height_px.astype(int).astype(str)).to_numpy() == "3001x3001"
    n1, n0 = int((y[sq] == 1).sum()), int((y[sq] == 0).sum())
    za, zb = norm.ppf(0.975), norm.ppf(0.80)
    se0 = np.sqrt((n1 + n0 + 1) / (12.0 * n1 * n0))
    mda = None
    for a in np.arange(0.501, 0.999, 0.001):
        if a - 0.5 >= za * se0 + zb * np.sqrt(hanley_var(a, n1, n0)):
            mda = float(round(a, 3))
            break
    out["A6_single_detector"] = {"n_hernia": n1, "n_control": n0,
                                 "min_detectable_auc_80pct_power": mda,
                                 "null_se": float(se0)}
    print(f"n1={n1} n0={n0}: smallest AUC detectable vs 0.5 at 80% power = {mda}")

    (RES / "revision_analyses.json").write_text(json.dumps(out, indent=1, default=float))
    print("\nwrote", RES / "revision_analyses.json")


class _LinLog:
    """[age, log1p(age)], the basis age_analysis.py uses."""
    def fit(self, Z):
        return self

    def transform(self, Z):
        return np.column_stack([Z, np.log1p(np.clip(Z, 0, None))])


class _SplineSome:
    """Cubic spline on the continuous columns, binary columns passed through."""
    def __init__(self, cont, binc):
        self.cont, self.binc = cont, binc

    def fit(self, Z):
        self.sp = spline_basis().fit(Z[:, self.cont])
        return self

    def transform(self, Z):
        return np.column_stack([self.sp.transform(Z[:, self.cont]), Z[:, self.binc]])


if __name__ == "__main__":
    main()
