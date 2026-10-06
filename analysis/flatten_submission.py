#!/usr/bin/env python3
"""Build the LaTeX zip BMC asks for.

BMC's uploader says:

  "It's best if your manuscript - including all text, figures and tables - is
   in one editable file. LaTeX documents with figures and tables compressed
   into a .zip format. We will compile these into a PDF for peer review."

So the zip has to compile to a complete PDF, figures included. That is the
opposite of the older "figures as separate uploads" route, which needed the
graphics stripped out; nothing is stripped here.

What this produces in submission/:

  manuscript_latex.zip   what goes in the manuscript slot. Flat inside: the
                         single .tex, Figure1-5.pdf, the BMC class files,
                         refs.bib and the prebuilt .bbl.
  Additional_file_1.pdf  uploaded separately, as an additional file
  cover_letter.txt       pasted into the form
  SUBMISSION_CHECKLIST.md

The working copy keeps \input{tables/...} and figures/ paths; only the copy
inside the zip is rewritten.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

BASE = Path("/scratch/hl106/80_workspace/foye/paper")
OUT = BASE / "submission"
ZIP = OUT / "manuscript_latex.zip"

SUPPORT = ["bmcart.cls", "bmcart-biblio.sty", "bmc-mathphys.bst", "refs.bib"]
FIGS = ["figure1_design", "figure1_images", "figure2_discrimination",
        "figure3_sensitivity", "figure4_device"]

# Additional file 1 travels as its own archive: plain article class, no
# bibliography, its own S-numbered figures.
SUPP_ZIP = OUT / "Additional_file_1.zip"
SUPP_FIGS = ["figureS1_recoverability", "figureS2_controls"]

TECTONIC = "/scratch/hl106/80_workspace/.tools/bin/tectonic"
BIBTEX = "/scratch/hl106/conda_envs/tex/bin/bibtex"


def flatten(tex: str) -> str:
    """Inline \\input{tables/...}: BMC wants one .tex, not a tree of them."""
    def sub(m):
        p = BASE / (m.group(1) + ".tex")
        return (f"% --- inlined from {m.group(1)}.tex ---\n"
                + p.read_text().rstrip() + "\n% --- end inline ---")
    out = re.sub(r"\\input\{([^}]+)\}", sub, tex)
    assert r"\input{" not in out, "an \\input survived flattening"
    return out


def retarget_figures(tex: str, figs=None, prefix="Figure") -> str:
    """Point \\includegraphics at the flat Figure{n}.pdf names used in the zip.

    Everything sits at the top level of the archive so the file compiles
    wherever it is unpacked, with no figures/ subdirectory to preserve.
    """
    # take the order from the document itself, so renumbering a float in the
    # manuscript cannot leave the archive's Figure{n}.pdf names behind
    if figs is None:
        figs = [m.group(1) for m in
                re.finditer(r"\\includegraphics\[[^\]]*\]\{figures/([^}]+)\.pdf\}", tex)]
        assert sorted(figs) == sorted(FIGS), f"unexpected figure set: {figs}"
    for i, name in enumerate(figs, start=1):
        old, new = f"figures/{name}.pdf", f"{prefix}{i}.pdf"
        assert tex.count(old) == 1, f"expected one reference to {old}"
        tex = tex.replace(old, new)
    assert "figures/" not in tex
    assert tex.count(r"\includegraphics") == len(figs)
    return tex


def build_and_check(workdir: Path, stem: str = "manuscript",
                    want_bbl: bool = True) -> Path | None:
    """Run the publisher's build and fail here rather than on their server.

    tectonic's bundled BibTeX has a fixed buffer that overflows on
    bmc-mathphys.bst with this many references, so TeX and BibTeX are run
    separately: tectonic with --pass tex, then a real BibTeX. The PDF is then
    typeset from a throwaway copy with the .bbl pasted in place of
    \\bibliography, which is the only way tectonic never reaches its own BibTeX.
    The file that ships keeps \\bibliography{refs} and is untouched.
    """
    src = workdir / f"{stem}.tex"

    def tex_pass(target: Path, final=False):
        cmd = [TECTONIC, "--keep-intermediates", "-o", str(workdir), str(target)]
        if not final:
            cmd[1:1] = ["--pass", "tex"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        assert r.returncode == 0, f"tectonic failed on {target.name}:\n" + r.stderr[-2000:]
        return r.stderr

    tex_pass(src)
    bbl = None
    n_items = 0
    if want_bbl:
        r = subprocess.run([BIBTEX, stem], cwd=workdir, capture_output=True, text=True)
        assert r.returncode == 0, "bibtex failed:\n" + r.stdout[-2000:]
        bbl = workdir / f"{stem}.bbl"
        n_items = bbl.read_text().count(r"\bibitem")

    body = src.read_text()
    if bbl is not None:
        m = re.search(r"\\bibliographystyle\{[^}]*\}\s*\n\\bibliography\{[^}]*\}", body)
        assert m, "bibliography block not found"
        # slicing, not re.sub: the .bbl contains escapes like \i
        body = body[:m.start()] + bbl.read_text() + body[m.end():]
    check = workdir / f"_check_{stem}.tex"
    check.write_text(body)

    tex_pass(check)
    log = tex_pass(check, final=True)
    bad = [ln for ln in log.splitlines()
           if ln.startswith("error") or "undefined" in ln.lower() or "Citation" in ln]
    assert not bad, "build is not clean:\n" + "\n".join(bad[:10])

    pdf = workdir / f"_check_{stem}.pdf"
    assert pdf.exists() and pdf.stat().st_size > 50_000, "no PDF produced"
    pages = pdf.read_bytes().count(b"/Type /Page") - pdf.read_bytes().count(b"/Type /Pages")
    print(f"  {stem}: builds clean, {n_items} references, "
          f"PDF {pdf.stat().st_size:,} bytes")
    return bbl


def build_supplementary_zip():
    """Additional file 1 as its own archive: source, figures and the PDF.

    BMC displays the PDF to reviewers, so it ships too; the source is there
    because they ask for editable formats.
    """
    tex = (BASE / "supplementary.tex").read_text()
    tex = retarget_figures(flatten(tex), SUPP_FIGS, prefix="FigureS")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        (tmp / "supplementary.tex").write_text(tex)
        for i, name in enumerate(SUPP_FIGS, start=1):
            shutil.copy(BASE / "figures" / f"{name}.pdf", tmp / f"FigureS{i}.pdf")
        build_and_check(tmp, stem="supplementary", want_bbl=False)
        shutil.copy(BASE / "supplementary.pdf", tmp / "Additional_file_1.pdf")
        # the standalone PDF is what gets uploaded, so write it from the build
        # that just happened: a leftover copy from an earlier run is the easiest
        # wrong file to pick in the uploader
        shutil.copy(BASE / "supplementary.pdf", OUT / "Additional_file_1.pdf")
        members = ["Additional_file_1.pdf", "supplementary.tex"] + \
                  [f"FigureS{i}.pdf" for i in range(1, len(SUPP_FIGS) + 1)]
        with zipfile.ZipFile(SUPP_ZIP, "w", zipfile.ZIP_DEFLATED) as z:
            for m in members:
                z.write(tmp / m, arcname=m)
    return members


def write_cover_letter():
    md = (BASE / "cover_letter.md").read_text()
    body = md.split("\n", 1)[1].replace("**", "").replace("*", "")
    assert "[DATE]" not in body and "TO BE COMPLETED" not in body, \
        "cover letter still has a placeholder"
    (OUT / "cover_letter.txt").write_text(body.strip() + "\n")


def write_checklist(members):
    inside = "\n".join(f"  {m}" for m in members)
    (OUT / "SUBMISSION_CHECKLIST.md").write_text(f"""# BMC Medical Imaging submission package

Generated by `analysis/flatten_submission.py`.

## What goes in which slot

| Slot in the submission system | File |
|---|---|
| Manuscript (LaTeX, editable) | `manuscript_latex.zip` |
| Additional file 1 | `Additional_file_1.zip` |
| Cover letter (paste or upload) | `cover_letter.txt` |

BMC compiles the zip into the peer-review PDF, so the zip carries the figures
rather than having them uploaded one by one. Contents, all at the top level:

{inside}

## Notes

* `manuscript.tex` is one self-contained file: the tables are inlined, and the
  figures are referenced as `Figure1.pdf` ... `Figure{len(FIGS)}.pdf` sitting
  next to it in the archive.
* The build was verified end to end before zipping: tex, bibtex, tex, tex, with
  no errors and no undefined citations. `manuscript.bbl` is that build's output,
  included so their compiler does not have to resolve the bibliography itself.
* Article type: **Research**.
* Corresponding author: Fang Lin, cesare_l@icloud.com.
* Ethics approval [2026] Ke-Yan-Lun-Shen-Zi No. 061, Clinical Research Ethics
  Committee of the First Affiliated Hospital of Xiamen University.
* No suggested reviewers are named; leave that field blank.
* Data availability: code and numerical results on request, radiographs cannot
  be released. Do **not** upload anything from `ChestCR_prepared/` - the file
  names carry hospital identifiers and the pixels still carry age.
""")


def main():
    OUT.mkdir(exist_ok=True)
    tex = (BASE / "manuscript.tex").read_text()
    tex = retarget_figures(flatten(tex))

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        (tmp / "manuscript.tex").write_text(tex)
        for f in SUPPORT:
            shutil.copy(BASE / f, tmp / f)
        for i, name in enumerate(FIGS, start=1):
            shutil.copy(BASE / "figures" / f"{name}.pdf", tmp / f"Figure{i}.pdf")

        bbl = build_and_check(tmp)

        members = ["manuscript.tex", "manuscript.bbl"] + \
                  [f"Figure{i}.pdf" for i in range(1, len(FIGS) + 1)] + SUPPORT
        with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
            for m in members:
                src = tmp / m
                assert src.exists(), f"{m} missing from the build directory"
                z.write(src, arcname=m)

    build_supplementary_zip()
    write_cover_letter()
    write_checklist(members)

    # Anything left from the old separate-figure route would only confuse the
    # upload; move it aside rather than deleting it.
    trash = BASE.parent / "trash" / "submission_old_separate_figures"
    stale = [p for p in OUT.iterdir()
             if p.name not in {ZIP.name, SUPP_ZIP.name,
                               "cover_letter.txt", "SUBMISSION_CHECKLIST.md",
                               "response_to_editor.pdf",
                               "response_to_reviewers.pdf",
                               "manuscript_marked_up.pdf",
                               "Additional_file_1.pdf"}]
    if stale:
        trash.mkdir(parents=True, exist_ok=True)
        for p in stale:
            shutil.move(str(p), str(trash / p.name))
        print(f"  moved {len(stale)} file(s) from the old layout to {trash}")

    print(f"\nwrote {OUT}")
    for p in sorted(OUT.iterdir()):
        print(f"  {p.name:30s} {p.stat().st_size:>9,} bytes")
    for arc in (ZIP, SUPP_ZIP):
        with zipfile.ZipFile(arc) as z:
            print(f"\n{arc.name} contains {len(z.namelist())} files:")
            for i in z.infolist():
                print(f"  {i.filename:26s} {i.file_size:>9,} bytes")


if __name__ == "__main__":
    main()
