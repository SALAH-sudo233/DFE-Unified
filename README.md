# DFE-Unified

DFE-Unified is an evidence-first snapshot of an experimental direction-field
extension to Pocket2Mol. It preserves the source actually present for the run,
the observed 500,000-iteration checkpoint, and the completed portion of a
30-pocket evaluation. The repository supports an honest review of what was
built and measured before further model development.

## Verified snapshot

| Item | Observed value |
| --- | --- |
| Pocket2Mol baseline | `836a0c4ce487297ad24bc54ac2ebd163de13242c` |
| Checkpoint iteration | 500,000 |
| Checkpoint SHA-256 | `34b2e8cadb7351c884e36cbfdee23de2a6ac2cb6fdd213612e401fd36c9d1fc0` |
| Evaluation coverage | 21 completed of 30 requested pockets |
| Evaluated records | 2,331 |
| Macro mean Vina score | -7.304 |
| Macro mean QED | 0.579 |
| PoseBusters pass rate | 986 / 2,331 (42.3%) |

These values are descriptive evidence, not a matched comparison against the
unmodified Pocket2Mol baseline. The interrupted run is not presented as a
completed 30-pocket benchmark.

## What the direction field is

The retained implementation derives eight heuristic features from pocket atoms:
nearest distance, a three-component nearest-atom direction, charge-weighted and
hydrophobicity-weighted distance terms, squared distance, and inverse distance.
An MLP projects them into the model's scalar feature channel. The charge and
hydrophobicity constants are fixed lookup values.

This implementation has not been shown to be a calibrated physical field or an
E(3)/SE(3)-equivariant construction. The evidence does not establish
state-of-the-art performance or causal improvement over Pocket2Mol.

## Repository map

- `models/`, `configs/`, and root scripts contain the pinned upstream tree plus
  the hash-bound files observed for the DF run.
- `artifacts/` contains the checkpoint and training log, stored with Git LFS.
- `results/df-500k-21-pocket/` contains the 21 completed per-pocket outputs.
- `evidence/` binds source, artifacts, and conclusions to machine-readable
  hashes and records the invalidated ADF/BIF experiment.
- `docs/` explains architecture, limitations, and reproduction boundaries.
- `dfe/diagnostics/` and `scripts/*phase0*` implement the create-only DF 500K
  Phase 0 diagnostic protocol; the operator workflow is in
  `docs/experiments/phase0-operator-runbook.md`.
- `data/` records external dataset provenance; datasets are not redistributed.

## Verify the snapshot

```bash
python scripts/generate_manifests.py
python -m unittest discover -s tests -v
python scripts/verify_repository.py
```

The final command verifies hashes, result scope, checkpoint structure, JSON,
credential exclusions, and Git LFS attributes. It also parses retained SDF files
when RDKit is installed.

See `docs/methodology-limitations.md` before interpreting the metrics and
`docs/reproduction.md` before attempting to resume training or sampling.
New DF 500K diagnostics must follow the Phase 0 operator runbook and keep the
500K checkpoint immutable.

## Upstream and license

This repository derives from
[PengXingang/Pocket2Mol](https://github.com/PengXingang/Pocket2Mol), associated
with the ICML 2022 paper *Pocket2Mol: Efficient Molecular Sampling Based on 3D
Protein Pockets*. See `UPSTREAM.md` for pinned source provenance. The upstream
MIT license is retained in `LICENSE`.
