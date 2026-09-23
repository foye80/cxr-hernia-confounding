#!/usr/bin/env python3
"""Build refs.bib from authoritative metadata (Crossref for journal DOIs,
DataCite for arXiv DOIs), verifying that each DOI actually resolves to the
paper we intend to cite.

Every entry is keyed by DOI, and the fetched title is checked against an
expected-title fragment. Anything that fails verification is reported and
written to refs_FAILED.txt instead of silently entering the bibliography.
"""

import json
import re
import subprocess
import sys
import time
from pathlib import Path

OUT = Path("/scratch/hl106/80_workspace/foye/paper/refs.bib")
FAIL = Path("/scratch/hl106/80_workspace/foye/paper/refs_FAILED.txt")
MAILTO = "yh163@rice.edu"

# key: (doi, distinctive title fragment that the resolved record must contain)
REFS = {
    "geirhos2020shortcut": ("10.1038/s42256-020-00257-z", "shortcut learning in deep neural networks"),
    "zech2018variable": ("10.1371/journal.pmed.1002683", "variable generalization performance"),
    "degrave2021ai": ("10.1038/s42256-021-00338-7", "selects shortcuts over signal"),
    "badgeley2019deep": ("10.1038/s41746-019-0105-1", "deep learning predicts hip fracture"),
    "gichoya2022ai": ("10.1016/S2589-7500(22)00063-2", "ai recognition of patient race"),
    "glocker2023algorithmic": ("10.1016/j.ebiom.2023.104467", "algorithmic encoding of protected characteristics"),
    "oakdenrayner2020hidden": ("10.1145/3368555.3384468", "hidden stratification"),
    "roberts2021common": ("10.1038/s42256-021-00307-0", "common pitfalls and recommendations"),
    "banerjee2023shortcuts": ("10.1016/j.jacr.2023.06.025", "causing bias in radiology artificial intelligence"),
    "collins2024tripod": ("10.1136/bmj-2023-078378", "tripod+ai statement"),
    "mongan2020claim": ("10.1148/ryai.2020200029", "checklist for artificial intelligence in medical imaging"),
    "zhao2020training": ("10.1038/s41467-020-19784-9", "confounder-free deep learning"),
    "obermeyer2019dissecting": ("10.1126/science.aax2342", "dissecting racial bias"),
    "wynants2020prediction": ("10.1136/bmj.m1328", "prediction models for diagnosis and prognosis of covid-19"),
    "varoquaux2022machine": ("10.1038/s41746-022-00592-y", "methodological failures and recommendations"),
    "kaufman2012leakage": ("10.1145/2382577.2382579", "leakage in data mining"),
    "burcharth2013inguinal": ("10.1371/journal.pone.0054367", "nationwide prevalence of groin hernia repair"),
    "olesen2019inguinal": ("10.1007/s10029-019-01877-0", "risk of incarceration in children with inguinal hernia"),
    "delong1988comparing": ("10.2307/2531595", "comparing the areas under two or more correlated"),
    "willemink2020preparing": ("10.1148/radiol.2020192224", "preparing medical imaging data for machine learning"),
    "zhang2025biomedclip": ("10.1056/AIoa2400640", "multimodal biomedical foundation model"),
    "perezgarcia2025raddino": ("10.1038/s42256-024-00965-w", "scalable medical image encoders"),
    "deng2009imagenet": ("10.1109/CVPR.2009.5206848", "imagenet"),
    "huang2017densenet": ("10.1109/CVPR.2017.243", "densely connected convolutional networks"),
    "chen2024internvl": ("10.1109/CVPR52733.2024.02283", "scaling up vision foundation models"),
    "jabbour2020deep": ("10.48550/arXiv.2009.10132", "exploiting and preventing shortcuts"),
    "glocker2023risk": ("10.1148/ryai.230060", "risk of bias in chest radiography"),
    "raghu2021deep": ("10.1016/j.jcmg.2021.01.008", "biological age from chest radiographs"),
    "ieki2022deep": ("10.1038/s43856-022-00220-6", "age estimation from chest x-rays"),
    "poplin2018prediction": ("10.1038/s41551-018-0195-0", "cardiovascular risk factors from retinal fundus"),
    # arXiv-only records (DataCite)
    "hu2022lora": ("10.48550/arXiv.2106.09685", "lora"),
    "dettmers2023qlora": ("10.48550/arXiv.2305.14314", "qlora"),
    "oquab2024dinov2": ("10.48550/arXiv.2304.07193", "dinov2"),
    "cohen2022torchxrayvision": ("10.48550/arXiv.2111.00595", "torchxrayvision"),
    "bai2025qwen25vl": ("10.48550/arXiv.2502.13923", "qwen2.5-vl"),
    "li2025llavaonevision": ("10.48550/arXiv.2408.03326", "llava-onevision"),
    "marafioti2025smolvlm": ("10.48550/arXiv.2504.05299", "smolvlm"),
    "sellergren2025medgemma": ("10.48550/arXiv.2507.05201", "medgemma"),
    "chen2024huatuogpt": ("10.48550/arXiv.2406.19280", "huatuogpt-vision"),
}


def curl(url, accept=None):
    cmd = ["curl", "-sL", "--max-time", "30", "-A", f"paper-bib-builder (mailto:{MAILTO})"]
    if accept:
        cmd += ["-H", f"Accept: {accept}"]
    cmd.append(url)
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout


def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


UNI = {
    "\u201c": "``", "\u201d": "''", "\u2018": "`", "\u2019": "'",
    "\u2013": "--", "\u2014": "---", "\u2212": "-", "\u00a0": " ",
    "\u2032": "'", "\u00d7": r"$\\times$", "\u2264": r"$\\leq$",
    "\u2265": r"$\\geq$", "\u03b1": r"$\\alpha$", "\u03b2": r"$\\beta$",
}


def esc(s):
    """LaTeX-escape and fold characters the ec-lmr fonts cannot represent."""
    s = s or ""
    for k, v in UNI.items():
        s = s.replace(k, v)
    return s.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")


def fetch_crossref(doi):
    d = json.loads(curl(f"https://api.crossref.org/works/{doi}"))["message"]
    return {
        "authors": [(a.get("family", ""), a.get("given", "")) for a in d.get("author", [])
                    if a.get("family")],
        "title": (d.get("title") or [""])[0],
        "venue": (d.get("container-title") or [""])[0],
        "year": (d.get("issued", {}).get("date-parts", [[None]])[0] or [None])[0],
        "volume": d.get("volume"), "number": d.get("issue"), "pages": d.get("page"),
        "doi": d.get("DOI"), "type": d.get("type", ""),
    }


def fetch_datacite(doi):
    d = json.loads(curl(f"https://api.datacite.org/dois/{doi}"))["data"]["attributes"]
    auth = []
    for a in d.get("creators", []):
        fam, giv = a.get("familyName"), a.get("givenName")
        if not fam and a.get("name"):
            parts = a["name"].split(",")
            fam = parts[0].strip()
            giv = parts[1].strip() if len(parts) > 1 else ""
        if fam:
            auth.append((fam, giv or ""))
    return {
        "authors": auth,
        "title": (d.get("titles") or [{}])[0].get("title", ""),
        "venue": "arXiv", "year": d.get("publicationYear"),
        "volume": None, "number": None, "pages": None,
        "doi": d.get("doi"), "type": "preprint",
    }


def to_bib(key, m):
    authors = " and ".join(f"{f}, {g}".strip(", ") for f, g in m["authors"])
    typ = "article" if m["venue"] and m["type"] != "preprint" else (
        "misc" if m["type"] == "preprint" else "inproceedings")
    fields = [f"  author = {{{authors}}}", f"  title = {{{{{esc(m['title'])}}}}}"]
    if m["venue"]:
        fields.append(f"  {'journal' if typ != 'inproceedings' else 'booktitle'} = "
                      f"{{{esc(m['venue'])}}}")
    if m["year"]:
        fields.append(f"  year = {{{m['year']}}}")
    for k, f in (("volume", "volume"), ("number", "number"), ("pages", "pages")):
        if m.get(k):
            fields.append(f"  {f} = {{{m[k]}}}")
    if m["doi"]:
        fields.append(f"  doi = {{{m['doi']}}}")
    return f"@{typ}{{{key},\n" + ",\n".join(fields) + "\n}\n"


def main():
    good, bad = [], []
    for key, (doi, frag) in REFS.items():
        try:
            m = fetch_datacite(doi) if doi.startswith("10.48550") else fetch_crossref(doi)
        except Exception as e:
            bad.append((key, doi, f"fetch error: {e}"))
            print(f"  FAIL {key:26s} fetch error")
            continue
        if norm(frag) not in norm(m["title"]):
            bad.append((key, doi, f"title mismatch: got {m['title']!r}"))
            print(f"  FAIL {key:26s} MISMATCH -> {m['title'][:60]!r}")
            continue
        first = m["authors"][0][0] if m["authors"] else "?"
        print(f"  ok   {key:26s} {first:16s} {str(m['year']):5s} {m['venue'][:34]}")
        good.append(to_bib(key, m))
        time.sleep(0.25)

    OUT.write_text("".join(good))
    print(f"\n[bib] {len(good)} verified entries -> {OUT}")
    if bad:
        FAIL.write_text("\n".join(f"{k}\t{d}\t{r}" for k, d, r in bad) + "\n")
        print(f"[bib] {len(bad)} FAILED -> {FAIL}")
        sys.exit(1)


if __name__ == "__main__":
    main()
