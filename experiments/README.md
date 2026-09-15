# Experiments (2026-09-08 → 2026-09-15)

Experiment code and small result records for the work that followed the SE(3)
vector-safe track. The generator was **frozen** at the best I3 baseline
(`armb_i3_run/checkpoints/25000.pt`) for all of this work; no generator retrain
was authorized.

Large binaries (checkpoints, embeddings, feature dumps, sequence files) are
**not committed**; they are registered by SHA-256 in
[`../evidence/large-artifacts-registry.json`](../evidence/large-artifacts-registry.json).
Narrative write-ups live in [`../docs/experiments/`](../docs/experiments).

## Directory map

| Directory | Track | What it holds |
| --- | --- | --- |
| `p3-screening/` | C (unified screening) | ScreeningHead + geometry-probe code, no-leak split, control metrics |
| `p4-pool-screening/` | C (rule-BIF re-rank) | Frozen-pool ranking framework + anti-gaming reports |
| `trackA-conservative-ft/` | A (generation) | 530k-regression diagnosis, conservative head-only fine-tune runners/configs |
| `cd100-diagnosis/` | A (generation) | CrossDocked-100 diagnosis pilot orchestration + aggregated summary |
| `trackB-ablation/` | B (DF ablation) | fair-comparison ablation record (single-pocket, candidate-selection only) |

## Headline results (all honest, several negative)

- **C-line ScreeningHead — weak pair-specificity (near-negative).** On the
  823-complex cluster-disjoint test set, full shared representation reaches
  test Pearson **0.648**, barely above `ligand_only` **0.628** and
  `pocket_only` **0.631**; `pocket_shuffle` **0.611**; `label_permutation`
  collapses to **0.117** (so the head does learn *something*, but the added
  pair information is marginal). Pooled EF1% ≈ 2.0, BEDROC20 ≈ 0.40 over
  top-25% pk actives. Records: `p3-screening/records/screeninghead_metrics.json`.
- **C-line geometry causal control (2026-09-15) — NO-GO.** Real paired atomic
  geometry does not beat the ligand marginal: full **0.320** < ligand_only
  **0.510**; full−permuted **+0.095** (plausibly pocket-marginal, not genuine
  pair interaction). Permutation integrity fully checked (identity/cross-split/
  same-cluster all 0). Decision: do **not** start BIF+DF joint fine-tuning and
  do **not** build a frozen interaction head under the current geometry
  representation. Records: `p3-screening/records/geometry_probe/causal_control_v2_metrics.json`;
  write-up: `../docs/experiments/cline-bif-geometry-signal-negative.md`.
- **Track A 530k continuation — regressed, stopped.** Paired 12-pocket ×
  3-seed pilot (72/72 accepted) shows 530k worse than DF-500k: Vina mean
  **+0.19 kcal/mol** [+0.03, +0.35], direct PB pass **−0.055**. Do not reuse
  530k as a start point. Records: `cd100-diagnosis/records/diagnosis_pilot_v2_summary.json`;
  write-up: `../docs/experiments/530k-regression-diagnosis-report.md`.
- **Track A conservative fine-tune — framework ready, not yet accepted.**
  Head-only (A) / head+position (B) runners with reset-optimizer semantics and
  dry-run gates are in place; Plan A formal 5k checkpoint exists but has not
  passed the paired acceptance gate. Review:
  `../docs/experiments/conservative-finetune-config-review.md`.
- **P4a rule-BIF re-rank — size artifact, negative after debias.** Raw-Vina
  "enrichment" is a heavy-atom artifact; with size-neutral ligand-efficiency
  oracle the enrichment vanishes. Framework reusable. Records under
  `p4-pool-screening/records/`.

## Reproduction notes

- Shared-representation extraction requires the frozen i3 model with
  `set_science_vector_origin('zero')`, kekulized ligands, and the transform
  order documented in `p3-screening/code/probe_emb3.py`.
- `refined_split.json` (committed) + `refined_labels.json` / `refined_seqs.fasta`
  / `refined_clusters.json` (registered, not committed) reproduce the no-leak
  split (seed 20260908, train 4049 / val 444 / test 823, all 5 eval pockets in test).
