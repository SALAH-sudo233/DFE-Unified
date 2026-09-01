# Architecture

## Baseline

The source layout follows Pocket2Mol at upstream commit
`836a0c4ce487297ad24bc54ac2ebd163de13242c`. Pocket2Mol generates a ligand
autoregressively inside a protein pocket. Its encoder and field modules maintain
scalar and vector features, predict focal atoms, propose positions, classify
atom types, and predict bonds.

## Direction-field extension

`models/df_module.py` adds `AnalyticalDirectionField`. For every query point it
computes distances to valid pocket atoms and forms an eight-value feature vector:

1. distance to the nearest pocket atom;
2. the three components of the normalized direction to that atom;
3. a sum of fixed charge-like values divided by distance plus one;
4. a sum of fixed hydrophobicity-like values divided by distance plus one;
5. squared nearest distance;
6. inverse nearest distance.

Three linear layers with SiLU activations project these values to a learned
hidden representation. `models/maskfill.py` creates the module and injects its
output into the model's scalar features. The integration does not introduce a
separate vector representation for the raw direction components.

## Training and evaluation path

`configs/train_df.yml` selects the DF dimensions and the usual Pocket2Mol
training objective. `train_resume.py` records the resume path used for the
retained run. Sampling scripts generate candidate coordinates and atom/bond
predictions; conversion reconstructs RDKit molecules and writes SDF. The
evaluation scripts compute descriptors, docking scores, and optional
PoseBusters checks from molecules that reached that retained SDF stage.

## Evidence boundary

The original upstream source, remote overlay, checkpoint, log, and results are
separate evidence classes. `evidence/download-records.json` records source-server
file metadata. Generated manifests recompute local hashes but retain the remote
path and modification time observations. The clean Git history intentionally
does not import the research server's dirty worktree or operational scripts.
