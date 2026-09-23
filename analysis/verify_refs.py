#!/usr/bin/env python3
"""Read-only check of refs.bib against Crossref and DataCite.

build_bib.py *generates* the bibliography and checks a title fragment on the
way. This checks what is actually in the file right now, field by field, and
writes nothing. Run it after any manual edit to refs.bib.

  /scratch/hl106/conda_envs/zx_xrv/bin/python analysis/verify_refs.py
"""
import json
import re
import subprocess
import sys
import time
from pathlib import Path

PAPER = Path("/scratch/hl106/80_workspace/foye/paper")


def parse_bib(text):
    """Split refs.bib into {key: {field: value}}. Values keep their braces off."""
    entries = {}
    for m in re.finditer(r"@(\w+)\{([^,]+),(.*?)\n\}", text, re.S):
        kind, key, body = m.group(1), m.group(2).strip(), m.group(3)
        fields = {}
        for fm in re.finditer(r"(\w+)\s*=\s*\{(.*?)\}\s*,?\s*(?=\n\s*\w+\s*=|\s*$)", body, re.S):
            fields[fm.group(1).lower()] = re.sub(r"\s+", " ", fm.group(2)).strip()
        fields["_type"] = kind.lower()
        entries[key] = fields
    return entries


def norm(s):
    s = (s or "").replace("{", "").replace("}", "")
    s = s.replace("\\&", "&").replace("\\%", "%").replace("\\_", "_")
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def curl(url):
    r = subprocess.run(["curl", "-sL", "--max-time", "30", "-A", "refs-verifier"],
                       capture_output=True, text=True)
    r = subprocess.run(["curl", "-sL", "--max-time", "30", "-A", "refs-verifier", url],
                       capture_output=True, text=True)
    return r.stdout


def fetch(doi):
    """Return normalized metadata, or None if the DOI does not resolve."""
    if doi.lower().startswith("10.48550"):
        raw = curl(f"https://api.datacite.org/dois/{doi}")
        try:
            d = json.loads(raw)["data"]["attributes"]
        except Exception:
            return None
        titles = d.get("titles") or [{}]
        creators = []
        for a in d.get("creators", []):
            fam = a.get("familyName")
            if not fam and a.get("name"):
                fam = a["name"].split(",")[0].strip()
            if fam:
                creators.append(fam)
        return {"src": "datacite", "title": titles[0].get("title", ""),
                "year": d.get("publicationYear"), "venue": d.get("publisher") or "",
                "volume": None, "pages": None, "authors": creators}
    raw = curl(f"https://api.crossref.org/works/{doi}")
    try:
        d = json.loads(raw)["message"]
    except Exception:
        return None
    return {"src": "crossref", "title": (d.get("title") or [""])[0],
            "year": (d.get("issued", {}).get("date-parts", [[None]])[0] or [None])[0],
            "venue": (d.get("container-title") or [""])[0],
            "volume": d.get("volume"), "pages": d.get("page"),
            "authors": [a["family"] for a in d.get("author", []) if a.get("family")]}


def main():
    bib = parse_bib((PAPER / "refs.bib").read_text())
    tex = (PAPER / "manuscript.tex").read_text() + (PAPER / "supplementary.tex").read_text()
    cited = set()
    for m in re.findall(r"\\cite\{([^}]*)\}", tex):
        cited |= {k.strip() for k in m.split(",")}

    bad, checked = {}, 0
    for key in sorted(bib):
        e = bib[key]
        mark = " " if key in cited else "."      # '.' = present but never cited
        doi = e.get("doi")
        if not doi:
            bad[key] = ["no DOI in refs.bib"]
            print(f"{mark}FAIL {key}: no DOI")
            continue
        meta = fetch(doi)
        time.sleep(0.2)
        checked += 1
        if meta is None:
            bad[key] = [f"DOI does not resolve: {doi}"]
            print(f"{mark}FAIL {key}: DOI does not resolve ({doi})")
            continue

        problems = []
        bt, rt = norm(e.get("title")), norm(meta["title"])
        if bt and rt and bt not in rt and rt not in bt:
            problems.append(f"title: bib '{e.get('title')}' vs registry '{meta['title']}'")
        if e.get("year") and meta["year"] and str(e["year"]) != str(meta["year"]):
            problems.append(f"year: bib {e['year']} vs registry {meta['year']}")
        bv, rv = norm(e.get("journal") or e.get("booktitle")), norm(meta["venue"])
        if bv and rv and bv not in rv and rv not in bv:
            problems.append(f"venue: bib '{e.get('journal') or e.get('booktitle')}'"
                            f" vs registry '{meta['venue']}'")
        if e.get("volume") and meta["volume"] and norm(e["volume"]) != norm(meta["volume"]):
            problems.append(f"volume: bib {e['volume']} vs registry {meta['volume']}")
        if e.get("pages") and meta["pages"] and norm(e["pages"]) != norm(meta["pages"]):
            problems.append(f"pages: bib {e['pages']} vs registry {meta['pages']}")
        if meta["authors"]:
            first = norm(meta["authors"][0])
            if first and first not in norm(e.get("author")):
                problems.append(f"first author '{meta['authors'][0]}' not in bib author list")

        if problems:
            bad[key] = problems
            print(f"{mark}FAIL {key}  [{meta['src']}]")
            for p in problems:
                print(f"       {p}")
        else:
            print(f"{mark}ok   {key}  [{meta['src']}]  {meta['year']}  {meta['venue'][:44]}")

    print(f"\n{checked} DOIs resolved and compared; {len(bib)} entries in refs.bib; "
          f"{len(cited)} keys cited")
    uncited = sorted(set(bib) - cited)
    if uncited:
        print("present but never cited (BibTeX drops these): " + ", ".join(uncited))
    missing = sorted(cited - set(bib))
    if missing:
        print("CITED BUT ABSENT FROM refs.bib: " + ", ".join(missing))
    if bad:
        print(f"\n{len(bad)} entries need attention")
        return 1
    print("\nevery cited reference matches its registry record")
    return 0


if __name__ == "__main__":
    sys.exit(main())
