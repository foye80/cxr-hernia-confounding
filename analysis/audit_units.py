#!/usr/bin/env python3
"""Unit-of-analysis audit for the revision (editor comment 2, reviewer 1.15)
and the laterality recount (editor comment 9, reviewer 3.4-3.6).

Questions answered:
  * how many distinct children the 2009 analysed radiographs come from;
  * whether any child contributes more than one radiograph, or appears in
    both groups;
  * which clinical records are ambiguous (one image linked to two records,
    byte-identical files, conflicting fields);
  * laterality and comorbidity counts on the records that actually enter
    the analysis.

The child identifier is the 8-digit radiology number that prefixes every
image file name. The `patient_id` field of the clinical table is an
admission number, so it can differ between two admissions of one child.

Only counts are written. No identifier, file name or date leaves this script.

Output: results/patient_units.json, results/clean_cohort_mask.npz
(a boolean mask aligned to the analysis order, plus the exclusion reason).
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

FOYE = Path("/scratch/hl106/foye")
META = FOYE / "ChestCR_prepared" / "metadata.json"
RAW = FOYE / "ChestCR"
PREP = FOYE / "ChestCR_prepared"
PROXY_CSV = Path("/scratch/hl106/80_workspace/foye/radiographic_proxy_probe/"
                 "primary_2k_radiographic_proxy_metadata.csv")
RES = Path("/scratch/hl106/80_workspace/foye/paper/results")

# folder name on disk -> label (1 = hernia)
FOLDER_LABEL = {"0": 1, "1": 0}


def items_of(txt):
    """Split a discharge diagnosis into its numbered items."""
    parts = re.split(r"[,，;；]\s*(?=\d+\.)", str(txt))
    return [re.sub(r"^\s*\d+\.", "", p).strip() for p in parts if p.strip()]
def top_paren(it):
    """split an item into text outside parentheses and top-level parenthetical groups"""
    out, groups, depth, buf = [], [], 0, ""
    for ch in it:
        if ch in "（(":
            if depth == 0: groups.append(""); 
            else: groups[-1] += ch
            depth += 1; continue
        if ch in "）)":
            depth -= 1
            if depth > 0: groups[-1] += ch
            continue
        if depth == 0: out.append(ch)
        else: groups[-1] += ch
    return "".join(out), groups
def parse_side(txt):
    """Side of the inguinal hernia repaired at this admission.

    Reads only the items that name an inguinal hernia, so a left-sided
    cryptorchidism next to a right-sided hernia no longer makes the hernia
    left (the original parser in prepare_dataset.py searched the whole
    string). An item whose own qualifier is exactly \"术后\" records an earlier
    repair, not this one. Returns bilateral / left / right / unspecified,
    postop_only (every hernia item is an earlier repair) or no_hernia_item.
    """
    cur, hernia_items, postop_only = set(), 0, 0
    for it in items_of(txt):
        main, groups = top_paren(it)
        if not ("腹股沟" in main and "疝" in main): continue
        hernia_items += 1
        if "术后" in main or any(g.strip() == "术后" for g in groups):
            postop_only += 1; continue
        # side marker sits in the item head, e.g. 左（侧） / 右侧 / 双侧
        head = main
        if "双侧" in head: cur |= {"L", "R"}
        else:
            if "左" in head: cur.add("L")
            if "右" in head: cur.add("R")
            if not ("左" in head or "右" in head): cur.add("?")
    if hernia_items == 0: return "no_hernia_item"
    if not cur: return "postop_only"
    s = cur - {"?"}
    return {frozenset({"L","R"}): "bilateral", frozenset({"L"}): "left",
            frozenset({"R"}): "right"}.get(frozenset(s), "unspecified")


def child_of(image_id: str) -> str:
    return image_id.split("_")[0]


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    out: dict = {}

    # ---------------- clinical records ----------------
    meta = pd.DataFrame(json.load(open(META)))
    meta["label"] = meta["label"].astype(int)
    meta["child"] = meta["image_id"].map(child_of)
    meta["side_parsed"] = meta["diagnosis_raw"].map(parse_side)
    # D: the label does not match the diagnosis text of this admission
    meta["label_doubt"] = (((meta.label == 1) & meta.side_parsed.isin(["postop_only", "no_hernia_item"]))
                           | ((meta.label == 0) & ~meta.side_parsed.isin(["postop_only", "no_hernia_item"])))
    out["records"] = {
        "n": int(len(meta)),
        "by_label": {str(k): int(v) for k, v in meta.label.value_counts().items()},
        "unique_patient_id": int(meta.patient_id.nunique()),
        "unique_image_id": int(meta.image_id.nunique()),
        "unique_child": int(meta.child.nunique()),
    }
    rec_per_img = meta.groupby("image_id").size()
    shared = rec_per_img[rec_per_img > 1].index
    sh = meta[meta.image_id.isin(shared)]
    agree = sh.groupby("image_id").agg(
        labels=("label", "nunique"), sexes=("sex", "nunique"),
        sides=("side", lambda s: s.fillna("NA").nunique()))
    out["records"]["image_id_with_multiple_records"] = {
        "n_image_ids": int(len(shared)),
        "n_records": int(len(sh)),
        "record_count_distribution": {str(k): int(v) for k, v in
                                      rec_per_img[rec_per_img > 1].value_counts().items()},
        "label_conflict": int((agree.labels > 1).sum()),
        "sex_conflict": int((agree.sexes > 1).sum()),
        "side_conflict": int((agree.sides > 1).sum()),
    }

    # ---------------- files on disk ----------------
    files = {}
    for folder, lab in FOLDER_LABEL.items():
        raw = sorted(p for p in (RAW / folder).glob("*.jpg"))
        prep = sorted(p for p in (PREP / folder).glob("*.jpg"))
        files[folder] = {"raw": raw, "prep": prep}
    out["files"] = {
        f"folder_{k}_label_{FOLDER_LABEL[k]}": {"raw_jpg": len(v["raw"]),
                                                "prepared_jpg": len(v["prep"])}
        for k, v in files.items()}

    # ---------------- analysis cohort ----------------
    proxy = pd.read_csv(PROXY_CSV)
    assert len(proxy) == 2009, len(proxy)
    proxy["child"] = proxy["image_id"].map(child_of)
    out["analysis_cohort"] = {
        "n": int(len(proxy)),
        "by_label": {str(k): int(v) for k, v in proxy.label.value_counts().items()},
        "unique_image_id": int(proxy.image_id.nunique()),
        "unique_child": int(proxy.child.nunique()),
        "metadata_rows_distribution": {str(k): int(v) for k, v in
                                       proxy.metadata_rows.value_counts().items()},
        "metadata_conflict_true": int(proxy.metadata_conflict.astype(str)
                                      .str.lower().eq("true").sum()),
    }

    # children with more than one analysed radiograph
    per_child = proxy.groupby("child").agg(n=("image_id", "size"),
                                           labels=("label", "nunique"))
    multi = per_child[per_child.n > 1]
    out["analysis_cohort"]["children_with_multiple_images"] = {
        "n_children": int(len(multi)),
        "n_images": int(multi.n.sum()),
        "images_per_child_distribution": {str(k): int(v) for k, v in
                                          multi.n.value_counts().items()},
        "children_in_both_groups": int((multi.labels > 1).sum()),
    }

    # byte-identical analysed images (on the prepared build)
    path_of = {}
    for folder, lab in FOLDER_LABEL.items():
        for p in files[folder]["prep"]:
            path_of[(p.stem, lab)] = p
    hashes = []
    for r in proxy.itertuples():
        p = path_of.get((r.image_id, int(r.label)))
        hashes.append(md5(p) if p is not None else None)
    proxy["md5"] = hashes
    out["analysis_cohort"]["missing_prepared_file"] = int(proxy.md5.isna().sum())
    dup = proxy[proxy.md5.notna()].groupby("md5").agg(
        n=("image_id", "size"), labels=("label", "nunique"))
    dup = dup[dup.n > 1]
    out["analysis_cohort"]["byte_identical_groups"] = {
        "n_groups": int(len(dup)),
        "n_images": int(dup.n.sum()),
        "across_both_groups": int((dup.labels > 1).sum()),
    }

    # ---------------- conservative clean cohort ----------------
    # Drop every analysed image whose unit is in any doubt:
    #   A  its image_id is linked to more than one clinical record
    #   B  its child contributes more than one analysed image
    #   C  it is byte-identical to another analysed image
    #   D  its label disagrees with the diagnosis text of the admission
    #      (hernia group, but every hernia item is an earlier repair; or
    #      control group, but the diagnosis names a current inguinal hernia)
    reason = np.array([""] * len(proxy), dtype=object)
    a = proxy.image_id.isin(shared).to_numpy()
    b = proxy.child.isin(multi.index).to_numpy()
    c = proxy.md5.isin(dup.index).to_numpy()
    doubt_imgs = set(meta.image_id[meta.label_doubt])
    dd = proxy.image_id.isin(doubt_imgs).to_numpy()
    for flag, tag in [(a, "A"), (b, "B"), (c, "C"), (dd, "D")]:
        reason[flag] = [r + tag for r in reason[flag]]
    keep = reason == ""
    out["clean_cohort"] = {
        "rule": "drop if image_id has >1 clinical record (A), child has >1 "
                "analysed image (B), image is byte-identical to another (C), or the label "
                "disagrees with the diagnosis text of that admission (D)",
        "n_dropped": int((~keep).sum()),
        "dropped_by_reason": {k: int(v) for k, v in Counter(reason[~keep]).items()},
        "dropped_by_label": {str(k): int(v) for k, v in
                             proxy.label[~keep].value_counts().items()},
        "n_kept": int(keep.sum()),
        "kept_by_label": {str(k): int(v) for k, v in
                          proxy.label[keep].value_counts().items()},
        "kept_unique_child": int(proxy.child[keep].nunique()),
    }

    # ---------------- laterality (T2) ----------------
    def lat(df):
        d = df[df.label == 1]
        side = d.side_parsed.value_counts()
        old = d.side.fillna("missing").value_counts()
        n_bi = int(side.get("bilateral", 0))
        n_r = int(side.get("right", 0))
        n_l = int(side.get("left", 0))
        return {
            "n_children": int(len(d)),
            "side": {str(k): int(v) for k, v in side.items()},
            "side_original_parser": {str(k): int(v) for k, v in old.items()},
            "sum_bilateral_right_left": n_bi + n_r + n_l,
            "repaired_sides": 2 * n_bi + n_r + n_l,
            "with_hydrocele": int(d.with_hydrocele.astype(str).str.lower().eq("true").sum()),
            "with_cryptorchidism": int(d.with_cryptorchidism.astype(str).str.lower().eq("true").sum()),
            "hernia_type": {str(k): int(v) for k, v in d.hernia_type.value_counts().items()},
        }

    # records as a whole (what the manuscript's counts may have used)
    out["laterality_all_records"] = lat(meta)
    # one record per analysed image: the analysed image's own record(s)
    m_an = meta[meta.image_id.isin(set(proxy.image_id))]
    out["laterality_records_of_analysed_images"] = lat(m_an)
    m_clean = meta[meta.image_id.isin(set(proxy.image_id[keep]))]
    out["laterality_clean_cohort"] = lat(m_clean)
    # raw diagnosis strings behind each parsed side, to audit the parser
    d = meta[meta.label == 1]
    out["label_doubt_records"] = {
        "hernia_postop_only": int(((meta.label == 1) & (meta.side_parsed == "postop_only")).sum()),
        "hernia_no_hernia_item": int(((meta.label == 1) & (meta.side_parsed == "no_hernia_item")).sum()),
        "control_with_current_inguinal_hernia": int(((meta.label == 0) & meta.label_doubt).sum()),
    }
    out["side_reclassified"] = {f"{a}->{b}": int(n) for (a, b), n in
        meta[meta.label == 1].groupby(["side", "side_parsed"]).size().items() if a != b}
    out["side_parser_audit"] = {
        str(side): {
            "has_双侧": int(g.diagnosis_raw.str.contains("双侧").sum()),
            "has_左": int(g.diagnosis_raw.str.contains("左").sum()),
            "has_右": int(g.diagnosis_raw.str.contains("右").sum()),
            "n": int(len(g)),
        } for side, g in d.groupby(d.side.fillna("missing"))}

    RES.mkdir(exist_ok=True)
    (RES / "patient_units.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    np.savez(RES / "clean_cohort_mask.npz", keep=keep, reason=reason.astype(str))
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
