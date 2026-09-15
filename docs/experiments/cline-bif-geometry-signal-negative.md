# BIF geometry signal probe — NEGATIVE result (2026-09-15)

## Question
Does real ligand-pocket paired atomic geometry carry affinity signal beyond ligand-only / pocket-only marginals? (Go/no-go for BIF+DF interaction head and joint fine-tuning.)

## Protocol
- Data: PDBBind refined 2020, 5316 complexes, cluster-disjoint split train/val/test = 4049/444/823.
- Features: ligand element marginal (48) + pocket element marginal (12) + ligand-pocket cross-molecule element-pair distance histogram (1152), bins 0,2,3,4,5,6,8,10,inf. total 1212.
- All rows included, incl. 676 aromatic-to-single recovery rows.
- Arms: full / permuted (ligand_i + pocket_j, restricted within-split, different cluster, no fixed point) / ligand_only / pocket_only.
- Seed 20260915. Remote: /workspace/ayb/experiments/dfe-unified-p3/geometry_probe/causal_control_v2/metrics.json

## Permutation integrity (all passed)
identity_count=0, cross_split_count=0, same_cluster_count=0, n_permuted=5316, parse_failures=0.

## Results (test Pearson)
| arm | test Pearson | test RMSE |
|---|---:|---:|
| full | 0.3195 | 2.292 |
| permuted | 0.2240 | 2.378 |
| ligand_only | 0.5105 | 2.052 |
| pocket_only | 0.1013 | 2.430 |

Deltas: full-permuted = +0.0955 ; full-ligand_only = -0.1909 ; full-pocket_only = +0.2182.

## Go/no-go criterion (locked before result)
1. full - permuted >= 0.05 : PASS (0.0955)
2. full > pocket_only : PASS
3. full > ligand_only : FAIL (0.320 < 0.510), direction reversed

## Decision: NO-GO
- Affinity predictability is dominated by the LIGAND MARGINAL (0.510), a known PDBBind ligand bias.
- Adding pocket + pair-geometry features LOWERS performance (full 0.320 < ligand_only 0.510); full early-stops at epoch 12 = overfitting from added noise dims.
- full-permuted +0.0955 is most plausibly pocket-marginal information, not genuine paired-geometry interaction.
- Independent confirmation of the earlier P3 conclusion (frozen encoder gives only per-modality marginal priors), now via real atomic geometry.

## Consequence
- Do NOT start BIF+DF joint fine-tuning.
- Do NOT build a frozen interaction head (failed even the weak-signal band; ligand_only reversal is a hard veto).
- Shelve the BIF pairwise-interaction sell-point under the current geometry representation; stop spending GPU on it.
- Redirect to the generation track: Plan A (freeze encoder+DF, train output heads only) 12-pocket pilot from DF-500k 500000.pt, as the CrossDocked-100 SOTA push backbone.

## Not claimed
Not a CrossDocked SOTA number. Histogram geometry is simplified (no directionality/chemical environment). 676 fallback rows included without a separate sensitivity analysis (per user instruction). full-permuted delta is descriptive intervention evidence, not a complete causal proof.
