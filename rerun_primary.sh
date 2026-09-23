#!/usr/bin/env bash
# Revision, 2026-09: the rebuilt (fully masked) images are the primary build.
# results/           <- rebuilt images (primary)
# results_published/ <- originally masked images (check that the mask does not matter)
set -euo pipefail
cd /scratch/hl106/80_workspace/foye/paper
P=/scratch/hl106/conda_envs/zx_xrv/bin/python
C=/scratch/hl106/foye/cohort_1960
mkdir -p results_published logs
echo "[start] $(date)"
$P analysis/run_paper_analysis.py --emb-dir $C/emb_remasked --out-dir results > logs/p_pathway_primary.log 2>&1 &
a=$!
$P analysis/run_paper_analysis.py --emb-dir $C/emb_published --out-dir results_published > logs/p_pathway_published.log 2>&1 &
b=$!
$P analysis/device_stratified.py > logs/p_device.log 2>&1 &
c=$!
HF_HOME=/scratch/hl106/huggingface /scratch/hl106/conda_envs/medshortcut/bin/python analysis/external_validation.py > logs/p_external.log 2>&1 &
d=$!
$P analysis/recoverability_clean.py > logs/p_recoverability.log 2>&1 &
e=$!
for p in $a $b $c $d $e; do wait $p; done
echo "[pathways, device, external, recoverability done] $(date)"
# both need results/oof_scores.npz from the primary build
$P analysis/comparisons_c.py > logs/p_comparisons.log 2>&1
$P -u analysis/revision_analyses.py > logs/p_revision_analyses.log 2>&1
$P analysis/make_tables.py > logs/p_tables.log 2>&1
$P analysis/make_figures.py > logs/p_figures.log 2>&1
$P analysis/make_figure_images.py > logs/p_fig1.log 2>&1
echo "[done] $(date)"
