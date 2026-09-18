# DFE-Unified — narrative reconstruction (2026-09-16)

## Why the original narrative had to change

The original sell-order was C > B > A, where **C = one shared representation doing
generation + affinity screening + interpretability**. C is now falsified, and A
(generation SOTA) is not currently reachable by further fine-tuning.

Measured evidence, all from this project:

| Claim tested | Result | Verdict |
|---|---|---|
| Continue DF pretraining 500k→530k | CrossDocked-100 macro Vina −6.70 vs −6.95; 12-pocket×3-seed pilot confirms degradation, worst on high-affinity pockets | dead |
| Conservative FT (freeze encoder+DF, heads only, lr 1e-5, 5k) | directional check net ≈ −0.17 kcal/mol worse | dead |
| Shared frozen encoder → absolute affinity (P3) | full 0.648 vs ligand-only 0.628 | no pair signal |
| Real atom-level pair geometry → absolute affinity (causal control v2) | full test Pearson 0.320 **below** ligand-only 0.510; full−permuted +0.096 attributable to pocket marginal | **hard veto** |

Root cause is not only our representation: PDBBind affinity labels are dominated by
the ligand marginal (LP-PDBBind / PDBbind-CleanSplit / GEMS show a model can score
well after removing one binding partner). Therefore *any* absolute-affinity
regression head trained on PDBBind inherits that bias, regardless of how good the
shared representation is.

**What died: "shared representation → absolute affinity screening".**
**What did not die: the idea of one model that both generates and judges its own
protein–ligand interaction geometry.**

## The reconstruction

Old: generation + affinity screening + interpretability from one shared embedding.

New: **generation + interaction-plausibility self-assessment, where that assessment
is the RL reward/critic.**

```
DF generator  ──generates pose──►  interaction critic
      ▲                                   │
      └──────── GRPO reward ◄──────────────┘
   (steric clash / H-bond geometry / buried unsatisfied donors)
```

Why this is defensible where C was not:

1. **Genuinely pair-specific.** Whether a ligand atom overlaps a pocket atom is a
   property of *that ligand in that pocket*. Swap the pocket and the label changes.
   It cannot be reduced to a ligand-only marginal by construction — though this must
   still be proven empirically, see the gate below.
2. **Does not touch PDBBind affinity labels.** Supervision comes from physical
   geometry, so the ligand-bias root cause is bypassed entirely.
3. **It fixes a real, previously unmeasured defect.** Our reported
   `PoseBusters(config='mol')` pass rate (60.8%) checks the ligand conformer only —
   no protein at all. A 4-pocket / 160-molecule probe of DF-500k output found
   **41.2% of generated molecules sterically clash with the pocket** (worst overlap
   1.23 Å, closest contact 1.59 Å) while still passing `config='mol'`. That failure
   mode was invisible in every metric we have published so far.
4. **It unifies with the RL track instead of competing with it.** The critic is not a
   second sell-point bolted on; it is the internal reward component for
   docking-in-the-loop GRPO, alongside Vina.

## Mandatory causal gate before building the critic

The previous line died because `ligand_only` beat `full`. The same trap is possible
here: large molecules clash more often, so "clash" could be a size artifact.

Critically, feeding ligand–pocket pair distances as input to predict clash is
**circular** — the label is derived from those distances. The decisive question is
the opposite one:

> Can clash be predicted from ligand-only information alone?

- If **yes** (high AUC from ligand descriptors) → clash is a proxy for "molecule is
  big", the critic would merely teach the model to shrink molecules. Reject.
- If **no** → the label genuinely requires the pocket, and a pair-aware critic has a
  reason to exist.

Gate (all must hold):

```
ligand_only AUC            < 0.65     (clash not explainable by ligand alone)
pocket_only AUC            < 0.65     (nor by pocket alone)
|corr(clash_rate, heavy_atoms)| < 0.5 (not a size artifact)
clash-free fraction in 0.05–0.95      (both classes present)
```

### Gate result: FAILED, twice. The critic is cancelled.

Measured on all 93 pockets / 10032 DF-500k molecules.

**Locked gate (molecule-level holdout):**

| criterion | measured | required | |
|---|---:|---|:--:|
| ligand_only AUC | 0.569 | < 0.65 | pass |
| pocket_only AUC | 0.671 | < 0.65 | **fail** |
| \|corr(clash, heavy_atoms)\| | 0.077 | < 0.5 | pass |
| clash-free fraction | 0.507 | 0.05–0.95 | pass |

Criterion 2 as locked was mis-specified: pocket difficulty is a legitimate prior, not
cheating, so the question should have been whether pair-aware features add increment
*over* that prior. The result is reported under the original gate first
(`GATE_PASS=false`), and the redesign argued separately — the threshold was not
restated to change the verdict.

**Revised gate — pair increment, pocket-grouped holdout (no pocket in both splits),
3 seeds:**

| feature block | AUC | ±sd |
|---|---:|---:|
| ligand_only | 0.520 | 0.038 |
| pocket_only | 0.641 | 0.015 |
| lig+pocket | **0.636** | 0.030 |
| coarse_pair_only | 0.538 | 0.016 |
| lig+pocket+coarse_pair | **0.620** | 0.040 |
| + borderline burial8 | 0.622 | 0.023 |

Required `delta >= +0.10` over the strongest marginal baseline. Measured
**`delta = -0.0166`** — coarse pair placement adds nothing and is marginally harmful.
`PAIR_GATE_PASS=false`.

Coarse pair features were restricted to aggregate placement (ligand centroid to pocket
centroid distance, projections on pocket principal axes, fraction of ligand atoms
outside the pocket 90th-percentile envelope, Rg ratio) and deliberately exclude
`min_dist` / `worst_overlap` / `n_clash`, which define the label. An 8 A burial count is
borderline (far beyond vdW contact ~3.5 A) and is therefore reported as a separate
variant rather than folded into the main number.

Switching to pocket-grouped holdout also corrected an earlier methodological flaw:
molecule-level splitting let the model memorise per-pocket difficulty, inflating
pocket_only to 0.671; grouped, it falls to 0.641 and ligand_only falls to 0.520
(≈ random).

**Interpretation.** Clash is determined neither by the ligand, nor by the pocket, nor by
coarse relative placement — only by per-atom relative geometry, which *is* the label's
definition. So clash is not a predictable property to be learned; it is a **directly
computable check** (milliseconds per molecule, no training, no generalisation risk).

**Consequence.** No interaction critic, and therefore no second head: the reconstructed
unified story does not survive its own gate, and the unified line is closed. The reward
signal itself stands, and must enter GRPO as a **hard geometric gate computed directly**,
alongside Vina.

Artifacts: `/workspace/ayb/experiments/dfe-unified-cd100/interaction_critic_gate/`
(`gate_result.json`, `pair_increment_gate.json`, `per_mol.jsonl`, `pair_coarse.jsonl`).

## Position on A (generation SOTA)

DF-500k is the best checkpoint we have: CrossDocked-100 pocket-macro Vina **−6.950**,
QED 0.564, direct PB 60.8%, heavy atoms 15.97, connectivity 100%, over 93/93
evaluated unique pocket directories, 10032 molecules.

Published references (Vina Dock, protocol **not** re-verified locally):
Pocket2Mol −7.15, reference ligands −7.45, TargetDiff −7.80, DecompDiff −8.39,
MolCRAFT ≈ −8.5. So DF-500k trails by ~0.2 to ~1.5 kcal/mol.

Two honest caveats that must accompany any A-line claim:

- **93/100, not 100/100.** Seven manifest entries map onto duplicate target
  directories; a strict 100-entry aggregate still needs those filled.
- **No no-DF matched baseline exists.** There is no capacity-matched no-DF
  Pocket2Mol checkpoint on the server, so "DF beats baseline" has never been tested
  on CrossDocked. Any future RL gain must be measured against the same DF-500k start
  under the same protocol, or the source of the gain is unattributable.

## Current plan

The unified line is closed — both in its original form (shared representation →
absolute affinity) and in its reconstructed form (interaction critic). Two tracks remain.

1. **Primary — docking-in-the-loop GRPO** from DF-500k as policy base. Multi-objective
   reward (Vina + QED + SA + strain) with KL to the reference policy to prevent reward
   hacking, and a **directly computed pocket-clash gate** — not a learned critic — as a
   hard term. Per-target pilot first, measured against the same DF-500k start under the
   identical sampling protocol.
2. **Backstop — methodology / negative-results paper.** Now materially stronger than when
   first proposed, because it carries four measured findings rather than one narrative:
   - **49.3% of DF-500k molecules sterically clash with the pocket while passing
     `PoseBusters(config='mol')`** (93 pockets, 10032 molecules). The field's standard
     validity metric, as commonly configured, does not look at the protein at all.
   - PDBBind absolute-affinity regression is ligand-marginal dominated: pair geometry
     scores *below* ligand-only (0.320 vs 0.510) under cluster-disjoint splits.
   - Clash is not learnable from marginals or coarse placement (delta −0.017); it is a
     computable check, not a predictable property.
   - Likelihood-continuation and conservative fine-tuning both degrade docking while
     supervised loss improves — supervised convergence is not a proxy for Vina.

Explicitly abandoned: extending likelihood pretraining, conservative FT hyperparameter
search, shared-encoder absolute-affinity screening, a learned interaction-plausibility
critic, and pure architecture/prior changes without an optimisation loop.
