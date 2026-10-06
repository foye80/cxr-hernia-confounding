# Can chest radiographs detect inguinal hernia in children? A study of confounding

Analysis code and numerical results for the study of the same name
(BMC Medical Imaging).

The study asks whether frozen chest-radiograph foundation models separate
children admitted for indirect inguinal hernia repair from other pediatric
surgical patients, and how much of that separation is confounding. The short
answer: all of it that we can measure. Matching the two groups on age,
acquisition and sex removes the separation.

## What is here

```
analysis/     every script that produced a number in the paper
results/      the numerical results those scripts wrote
tables/       the LaTeX tables generated from results/
build_pdf.sh  local PDF build for the manuscript
```

## What is not here, and why

The radiographs are not included. Their file names are hospital identifiers and
examination dates, the pixels carried a burned-in text block with the patient's
name and age, and the institution restricts sharing of the images. De-identified
images may be requested from the corresponding author with the approval of the
Clinical Research Ethics Committee of the First Affiliated Hospital of Xiamen
University.

For the same reason the following intermediate files are not included, because
each row names an image file: the per-image age table read from the overlay, the
per-image residual-text flags, and the de-identification audit report. Every
aggregate computed from them is in `results/`.

The frozen embeddings (four encoders x 1960 images) are not included either;
they are large and are reproducible from the images with
`analysis/extract_remasked.py`.

## Pipeline

Paths at the top of each script point at the local data directories and have to
be changed. Everything except the embedding extraction runs on CPU.

1. `analysis/remask_build.py` rebuilds the de-identified cohort from the
   originals, with one fixed mask rectangle for every image, and audits its own
   output. `analysis/burned_in_text.py` and `analysis/audit_text.py` hold the
   text detector.
2. `analysis/ocr_age.py` reads the age line from the overlay with a template
   matcher (`--templates` builds the glyph clusters first).
3. `analysis/extract_remasked.py` extracts the four frozen encoders (GPU).
4. `analysis/audit_units.py` checks that each radiograph belongs to one child,
   one record and one group, and recounts laterality from the discharge
   diagnosis text. `analysis/build_clean_cohort.py` applies the resulting mask
   to every row-aligned input.
5. `analysis/run_paper_analysis.py` runs the probes and the adjustment layers;
   `analysis/comparisons_c.py` the paired comparisons;
   `analysis/age_analysis.py` the age analysis;
   `analysis/device_stratified.py` the device strata;
   `analysis/external_validation.py` the external cohort;
   `analysis/recoverability_clean.py` how far each variable can be recovered
   from an embedding;
   `analysis/revision_analyses.py` the matched comparisons, the missingness
   analysis, the spline adjustment and the power calculation;
   `analysis/plan_matched_controls.py` the feasibility of a future age-matched
   cohort.
6. `analysis/make_tables.py`, `analysis/make_revision_tables.py`,
   `analysis/make_figures.py` and `analysis/make_figure_images.py` write the
   tables and figures.
7. `analysis/verify_manuscript.py` compares the values quoted in the manuscript
   and in the tables with `results/`, and exits non-zero on a mismatch. It
   checks that a quoted value exists in the result files, and, for the numbers
   the running text reports, that it is quoted in the running text and not only
   in a table. It does not parse sentences, so it cannot tell that a correctly
   copied number has been attached to the wrong experiment; two such errors were
   found by hand and each one added a specific guard to the script.

`rerun_clean.sh` and `rerun_primary.sh` run steps 5 and 6 in the order used for
the published results.

## Environment

Python 3.10, numpy, pandas, scipy, scikit-learn 1.6.1, matplotlib, opencv,
Pillow. Embedding extraction additionally needs torch, torchvision,
torchxrayvision, open\_clip and transformers. The vision--language-model
fine-tuning used peft and bitsandbytes; its settings are listed in the
Supplementary Methods of the paper.

## Reading the results

`results/KEY_NUMBERS.md` and `results/cohort_comparison.md` summarize the
headline numbers. `results/pathway_auc.csv` holds the probe results for every
encoder and adjustment layer, `results/revision_analyses.json` the matched
comparisons, `results/patient_units.json` the unit audit, and
`results/matched_cohort_plan.json` the design of a future age-matched cohort.

## Citation

Please cite the article. If you use the code, cite this repository as well:
DOI [10.5281/zenodo.22916390](https://doi.org/10.5281/zenodo.22916390). That is
the concept DOI and always resolves to the current version; each release also
has its own version DOI.

## License

MIT (see LICENSE).
