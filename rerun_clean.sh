#!/usr/bin/env bash
# Re-run every analysis on the 1960-image cohort (revision, 2026-09).
# Inputs are built by analysis/audit_units.py + analysis/build_clean_cohort.py.
# The 2009-image results are archived in ../trash/before_clean_cohort_20260921/.
set -euo pipefail
cd /scratch/hl106/80_workspace/foye/paper
P=/scratch/hl106/conda_envs/zx_xrv/bin/python
C=/scratch/hl106/foye/cohort_1960
echo "[start] $(date)"

# the two builds are independent: run them side by side
$P analysis/run_paper_analysis.py --emb-dir $C/emb_published --out-dir results \
    > logs/clean_pathway_published.log 2>&1 &
p1=$!
$P analysis/run_paper_analysis.py --emb-dir $C/emb_remasked --out-dir results_remasked \
    > logs/clean_pathway_remasked.log 2>&1 &
p2=$!
$P analysis/age_analysis.py --emb-dir $C/emb_published --tag published \
    > logs/clean_age_published.log 2>&1 &
p3=$!
$P analysis/age_analysis.py --emb-dir $C/emb_remasked --tag remasked \
    > logs/clean_age_remasked.log 2>&1 &
p4=$!
$P analysis/device_stratified.py > logs/clean_device.log 2>&1 &
p5=$!
$P analysis/plan_matched_controls.py > logs/clean_matching_plan.log 2>&1 &
p6=$!
HF_HOME=/scratch/hl106/huggingface /scratch/hl106/conda_envs/medshortcut/bin/python \
    analysis/external_validation.py > logs/clean_external.log 2>&1 &
p7=$!
for p in $p1 $p2 $p3 $p4 $p5 $p6 $p7; do wait $p; done
echo "[pathways, age, device, matching plan, external done] $(date)"

# needs results/oof_scores.npz from the published build
$P analysis/comparisons_c.py > logs/clean_comparisons.log 2>&1
echo "[done] $(date)"
