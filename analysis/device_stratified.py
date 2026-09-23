#!/usr/bin/env python3
"""Device-stratified probe: the decisive control.

Residualising on 5 crude file attributes is a weak proxy for the imaging device.
The stronger test is to hold the device fixed and ask whether the label is still
predictable. Two stratifications are available without DICOM headers:
  (a) the 3001x3001 square-detector group (one uncropped CR unit), and
  (b) the coarse CR / DX modality split.
"""
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
import sys; sys.path.insert(0, "analysis")
from run_paper_analysis import load_embeddings, PROXY_CSV, EMB_DIR, boot_auc_ci, MODEL_DISPLAY

meta = pd.read_csv(PROXY_CSV)
meta["sig"] = meta.image_width_px.astype(int).astype(str) + "x" + meta.image_height_px.astype(int).astype(str)
emb = load_embeddings(EMB_DIR)
y_all = meta.label.astype(int).to_numpy()

def probe(X, y, n_rep=5):
    if min(np.bincount(y)) < 10:
        return None
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_rep, random_state=42)
    oof = np.full((n_rep, len(y)), np.nan)
    for i, (tr, te) in enumerate(rskf.split(X, y)):
        sc = StandardScaler(); Xtr = sc.fit_transform(X[tr]); Xte = sc.transform(X[te])
        c = LogisticRegression(max_iter=5000, C=1.0, random_state=42).fit(Xtr, y[tr])
        oof[i // 5, te] = c.predict_proba(Xte)[:, 1]
    s = np.nanmean(oof, axis=0)
    lo, hi = boot_auc_ci(y, s, n_boot=2000)
    return roc_auc_score(y, s), lo, hi

strata = [("3001x3001 单机(方形探测器)", meta.sig == "3001x3001"),
          ("CR 模态全体", meta.modality_cr == 1),
          ("DX 模态全体", meta.modality_cr == 0),
          ("全队列(参照)", pd.Series(True, index=meta.index))]

rows = []
for name, mask in strata:
    idx = np.flatnonzero(mask.to_numpy())
    y = y_all[idx]
    print(f"\n{name}: n={len(idx)}  疝={int(y.sum())} 非疝={int((y==0).sum())}")
    for k in ["biomedclip", "torchxrayvision", "rad-dino", "imagenet"]:
        r = probe(emb[k][idx], y)
        if r is None:
            print(f"  {MODEL_DISPLAY[k]:22s} 样本不足"); continue
        a, lo, hi = r
        flag = "显著" if lo > 0.5 else "不显著(CI 含 0.5)"
        print(f"  {MODEL_DISPLAY[k]:22s} {a:.3f} [{lo:.3f}, {hi:.3f}]  {flag}")
        rows.append(dict(stratum=name, n=len(idx), n_hernia=int(y.sum()),
                         model=MODEL_DISPLAY[k], auc=a, ci_lo=lo, ci_hi=hi))
pd.DataFrame(rows).to_csv("results/device_stratified.csv", index=False)
print("\n-> results/device_stratified.csv")
