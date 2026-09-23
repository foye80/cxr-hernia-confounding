#!/usr/bin/env bash
# Build manuscript.pdf locally.
#
# Two local tools, each missing a piece:
#   * tectonic compiles bmcart.cls fine but its bundled BibTeX has a fixed
#     buffer that overflows on bmc-mathphys.bst with this many references
#     (panics at engine_bibtex/src/global.rs, "range end index 400022").
#   * the conda TeX Live at conda_envs/tex has a real BibTeX, but no LaTeX
#     format file and no access to tectonic's font cache, so it cannot typeset.
#
# So: tectonic runs TeX, the real BibTeX builds the .bbl, and tectonic then
# typesets a copy with the .bbl pasted in place of \bibliography, which means it
# never invokes its own BibTeX. manuscript.tex itself is left untouched and
# stays the thing that goes to Overleaf and to BMC, where a normal
# latex+bibtex+latex+latex run works.
#
# Usage: ./build_pdf.sh
set -euo pipefail
cd "$(dirname "$0")"
export PATH=/scratch/hl106/conda_envs/tex/bin:$PATH

tectonic --pass tex --keep-intermediates -o . manuscript.tex >/dev/null 2>&1
bibtex manuscript >/dev/null

python3 - <<'PY'
import re
from pathlib import Path
tex = Path("manuscript.tex").read_text()
bbl = Path("manuscript.bbl").read_text()
m = re.search(r"\\bibliographystyle\{[^}]*\}\s*\n\\bibliography\{[^}]*\}", tex)
assert m, "bibliography block not found"
out = tex[:m.start()] + bbl + tex[m.end():]   # not re.sub: the .bbl has \i etc.
Path(".manuscript_local.tex").write_text(out)
PY

tectonic .manuscript_local.tex >/dev/null 2>&1
tectonic .manuscript_local.tex 2>&1 | grep -E "^error|Writing" || true
mv .manuscript_local.pdf manuscript.pdf
echo "manuscript.pdf rebuilt"
