#!/usr/bin/env python3
"""Provenance check (ARCHIVED: reads the trashed June run): re-run the ORIGINAL June-13 protocol (single stratified
5-fold, no repeats) from the raw embeddings and proxy table, and compare with
residualized_hernia_probe_summary.json as it was written on 2026-06-13.

If these agree, the paper's pipeline is reading the same data and computing the
same thing; the paper's slightly different numbers come only from the change of
estimator (5x5 repeats + bagged out-of-fold score), not from different data.
"""
import json
import numpy as np, pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import sys
sys.path.insert(0, "analysis")
from run_paper_analysis import (ACQUISITION, GEOMETRY, EXPOSURE, SEX,
                                fold_residualize, load_embeddings, PROXY_CSV, EMB_DIR)

ALL29 = ACQUISITION + GEOMETRY + EXPOSURE + SEX
meta = pd.read_csv(PROXY_CSV)
y = meta["label"].astype(int).to_numpy()
emb = load_embeddings(EMB_DIR)
Z = meta[ALL29].apply(pd.to_numeric, errors="coerce").to_numpy(float)

def june_probe(X, y, Z=None, tabular=False):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    a = []
    for tr, te in skf.split(X, y):
        if tabular:
            c = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                              LogisticRegression(max_iter=5000, C=1.0, random_state=42))
            c.fit(X[tr], y[tr]); s = c.predict_proba(X[te])[:, 1]
        else:
            if Z is not None:
                Xtr, Xte, _ = fold_residualize(X[tr], X[te], Z[tr], Z[te])
            else:
                sc = StandardScaler(); Xtr = sc.fit_transform(X[tr]); Xte = sc.transform(X[te])
            c = LogisticRegression(max_iter=5000, C=1.0, random_state=42).fit(Xtr, y[tr])
            s = c.predict_proba(Xte)[:, 1]
        a.append(roc_auc_score(y[te], s))
    return float(np.mean(a))

june = {(x["model"], x["nuisance_set"]): x["auc_mean"] for x in json.load(
    open("/scratch/hl106/80_workspace/trash/foye_residualized_hernia_probe_20260813/residualized_hernia_probe_summary.json"))}

checks = [("biomedclip", "none_raw_embedding", june_probe(emb["biomedclip"], y)),
          ("torchxrayvision", "none_raw_embedding", june_probe(emb["torchxrayvision"], y)),
          ("rad-dino", "none_raw_embedding", june_probe(emb["rad-dino"], y)),
          ("imagenet", "none_raw_embedding", june_probe(emb["imagenet"], y)),
          ("proxy_only", "acquisition_geometry_exposure_sex", june_probe(Z, y, tabular=True)),
          ("biomedclip", "acquisition_geometry_exposure_sex", june_probe(emb["biomedclip"], y, Z))]

print(f"{'model':16s} {'nuisance':34s} {'2026-06-13 文件':>15s} {'今天重跑':>10s} {'差':>8s}")
ok = True
for m, ns, mine in checks:
    ref = june[(m, ns)]
    d = abs(ref - mine)
    ok &= d < 1e-6
    print(f"{m:16s} {ns:34s} {ref:15.6f} {mine:10.6f} {d:8.1e}")
print("\n完全一致" if ok else "\n不一致")
