# Reproduction guide

## Verify before running

Install Git LFS before cloning so the checkpoint and log are materialized. From
the repository root, run:

```bash
git lfs pull
python scripts/generate_manifests.py
python -m unittest discover -s tests -v
python scripts/verify_repository.py
```

The verifier requires PyTorch to inspect the checkpoint. RDKit is optional for
the core checks; when present, all retained SDF files are parsed and counted.

## Supply external data

Place the processed CrossDocked Pocket10 dataset and observed split at the paths
documented in `data/README.md`. Compare the split hash with
`data/crossdocked-manifest.json`. Dataset licenses and access terms remain with
their owners.

## Environment boundary

`env_cuda113.yml` is inherited from upstream and is historical guidance, not a
claim about the exact DF run. `environment/provenance.md` lists what was and was
not observed. Build a dedicated environment and record package, CUDA, driver,
docking executable, and RDKit versions before new experiments.

## Sampling and evaluation

`configs/sample_df_500k.yml` and `pocket_centers_30.json` preserve the intended
partial evaluation configuration. The shell and Python scripts at repository
root show the observed orchestration. Review all absolute data/tool paths and
replace them with paths in your isolated environment. Do not treat the old
server paths as portable configuration.

For a meaningful next experiment, run an unmodified Pocket2Mol checkpoint and
the DF checkpoint on the same held-out pockets, with identical seeds, sampling
budgets, reconstruction, docking, and PoseBusters settings. Record attempted
generations before reconstruction so end-to-end validity has a valid
denominator. Multiple seeds and uncertainty estimates are required before an
effect claim.
