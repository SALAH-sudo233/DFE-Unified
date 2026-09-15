# Causal geometry control v1

## Question
Does real ligand-pocket paired geometry carry signal beyond ligand-only and pocket-only marginal information?

## Data
PDBBind refined 2020; 5316 complexes; split train/val/test = 4049/444/823; cluster-disjoint split. All rows included, including 676 aromatic-to-single recovery rows.

## Features
- ligand element marginal;
- pocket element marginal;
- ligand-pocket cross-molecule element-pair distance histogram with bins 0,2,3,4,5,6,8,10,inf.

## Interventions
- full: real ligand_i + pocket_i;
- permuted: ligand_i + pocket_j, j restricted to the same split and a different protein cluster, with j != i;
- ligand-only;
- pocket-only.

The permutation is recorded in `permutation.json`; identity, cross-split, and same-cluster counts are hard-checked as zero.

## Decision
The primary interaction evidence is `full.test_pearson - permuted.test_pearson`, with ligand-only/pocket-only as marginal controls. This is intervention evidence, not by itself a complete causal claim. No generator joint fine-tuning is allowed based only on this probe.

## Output
Remote: `/workspace/ayb/experiments/dfe-unified-p3/geometry_probe/causal_control_v1/`
